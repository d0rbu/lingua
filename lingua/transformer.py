# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Self

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)
from xformers.ops import AttentionBias, fmha

from lingua import probe

flex_attention_comp = torch.compile(flex_attention)


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class AttentionImpl(Enum):
    SDPA = "sdpa"
    FMHA = "fmha"
    FLEX_ATTENTION = "flex_attention"
    XFORMERS = "xformers"


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    hidden_dim: int | None = None  # by default hidden_dim = 4 * dim * 2/3 because 4x usual expansion and 2/3 to adjust for gate projection
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024


def cross_entropy(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def causal_mask(
    b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
) -> torch.Tensor:
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths: torch.Tensor) -> torch.Tensor:
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert lengths.ndim == 1

    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: torch.Tensor | None = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(
        self: Self, theta: float, head_dim: int, max_seqlen: int = 1024
    ) -> None:
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            self.precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self: Self) -> None:
        self.freqs_cis[...] = self.precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self: Self, seqlen: int | None = None, tok_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
        and the end index 'end'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials.
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        cos, sin = freqs.cos(), freqs.sin()

        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self: Self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self: Self) -> None:
        torch.nn.init.ones_(self.weight)  # type: ignore

class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)

class Attention(nn.Module):
    def __init__(
        self: Self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )
    
    @staticmethod
    def __flex_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        assert block_mask is None or isinstance(block_mask, BlockMask)

        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        output = flex_attention_comp(queries, keys, values, block_mask=block_mask)
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        return output

    @staticmethod
    def __fmha(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_bias: AttentionBias | None = None,
    ) -> torch.Tensor:
        assert attn_bias is None or isinstance(attn_bias, AttentionBias)

        return fmha.memory_efficient_attention(queries, keys, values, attn_bias=attn_bias)
        # This uses B S H D instead of B H S D of pytorch

    @staticmethod
    def __sdpa(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: str | torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert mask is None or isinstance(mask, (str, torch.Tensor))

        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        is_causal = isinstance(mask, str) and mask == "causal"
        mask = None if isinstance(mask, str) else mask
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=is_causal,
            attn_mask=mask,
        )
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        return output

    ATTN_IMPLS = {
        AttentionImpl.SDPA: __sdpa,
        AttentionImpl.FMHA: __fmha,
        AttentionImpl.FLEX_ATTENTION: __flex_attention,
    }

    def forward(
        self: Self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: torch.Tensor | None = None,
        mask: BlockMask | AttentionBias | str | None = None,
        attn_impl: AttentionImpl = AttentionImpl.SDPA,
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        queries = self.q_proj(x.view_as(x))
        keys = self.k_proj(x.view_as(x))
        values = self.v_proj(x.view_as(x))

        output_shape = queries.shape
        # B S D -> B S H D
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        queries, keys = self.apply_rotary_emb(queries, keys, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            keys, values = self.kv_cache.update(keys, values, tok_idx)

        keys = self.repeat_kv(keys, self.heads_per_group, dim=2)
        values = self.repeat_kv(values, self.heads_per_group, dim=2)

        attn_impl = self.ATTN_IMPLS.get(attn_impl, None)

        if attn_impl is None:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = attn_impl(queries, keys, values, mask)

        output = self.o_proj(output.reshape(output_shape))

        return output

    def reset_parameters(self: Self, init_std=None, factor=1.0) -> None:
        init_std = init_std or (self.dim ** (-0.5))

        for linear in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.o_proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

    @staticmethod
    def reshape_for_broadcast(
        freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int
    ) -> torch.Tensor:
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
        for the purpose of broadcasting the frequency tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.
            seq_dim (int): Sequence dimension index.

        Returns:
            torch.Tensor: Reshaped frequency tensor.
        """
        ndim = x.ndim
        assert 0 <= seq_dim < ndim
        assert freqs_cis.shape == (
            x.shape[seq_dim],
            x.shape[-3],
            2,
            2,
        ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
        shape = [
            d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
        ] + [2, 2]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        queries: torch.Tensor,
        keys: torch.Tensor,
        seq_dim: int,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries_ = queries.reshape(*queries.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        keys_ = keys.reshape(*keys.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
        freqs_cis = Attention.reshape_for_broadcast(
            freqs_cis, queries_, seq_dim
        ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
        queries_out = (queries_ * freqs_cis).sum(5).flatten(3)
        keys_out = (keys_ * freqs_cis).sum(5).flatten(3)
        return queries_out.type_as(queries), keys_out.type_as(keys)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."

        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


class FeedForward(nn.Module):
    def __init__(
        self: Self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ) -> None:
        super().__init__()

        # snap hidden_dim to the next highest multiple of multiple_of
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.down_proj = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        # x of shape B S D
        h = self.up_proj(x.view_as(x))
        h_gate = self.gate_proj(x.view_as(x))
        output = self.down_proj(F.silu(h_gate) * h)

        return output

    def reset_parameters(self: Self, init_std=None, factor=1.0) -> None:
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))

        in_init_std = in_init_std
        out_init_std = out_init_std / factor

        for linear, init_std in zip(
            (self.up_proj, self.gate_proj, self.down_proj),
            (in_init_std, in_init_std, out_init_std),
        ):
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


class TransformerBlock(nn.Module):
    def __init__(self: Self, args: BaseTransformerArgs) -> None:
        super().__init__()

        assert (args.head_dim is not None) or (args.n_heads is not None), (
            "Should specify at least head_dim or n_heads"
        )
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0, "n_heads should be a multiple of n_kv_heads"
        assert args.dim % args.n_heads == 0, "dim should be a multiple of n_heads"

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim or args.dim * 4 * 2 // 3,
            multiple_of=args.multiple_of,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self: Self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: torch.Tensor | None = None,
        mask: BlockMask | AttentionBias | str | None = None,
        attn_impl: AttentionImpl = AttentionImpl.SDPA,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )

        out = h + self.feed_forward(self.ffn_norm(h))

        return out

    def init_weights(self: Self, init_std=None, factor=1.0) -> None:
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseTransformer(nn.Module):
    def __init__(self: Self, args: BaseTransformerArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = args.init_std_factor
        self.init_std_factor_fn = partial(self.INIT_STD_FACTOR_FNS[self.init_std_factor], self)

        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

    def forward(
        self: Self,
        h: torch.Tensor,
        tok_idx: torch.Tensor | None = None,
        mask: BlockMask | AttentionBias | str | None = None,
        attn_impl: AttentionImpl = AttentionImpl.SDPA,
    ) -> torch.Tensor:
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        return h

    def reset_parameters(self: Self) -> None:
        self.rope_embeddings.reset_parameters()

    INIT_STD_FACTOR_FNS = {
        InitStdFactor.CURRENT_DEPTH: lambda self, depth: (2 * (depth + 1)) ** 0.5,
        InitStdFactor.GLOBAL_DEPTH: lambda self, depth: (2 * (len(self.layers) + 1)) ** 0.5,
        InitStdFactor.DIM_RATIO: lambda self, depth: self.dim / 4096,
        InitStdFactor.DISABLED: lambda self, depth: 1.0,
    }

    def init_weights(self: Self) -> None:
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = self.init_std_factor_fn(depth)

            layer.init_weights(self.init_base_std, factor)
