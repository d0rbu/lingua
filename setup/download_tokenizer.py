import argparse
import os
from dataclasses import dataclass
from typing import Optional

from requests.exceptions import HTTPError


@dataclass
class TokenizerConfig:
    repo_id: str
    path: str = "tokenizer.model"


TOKENIZER_CONFIGS = {
    "llama": TokenizerConfig(
        repo_id="meta-llama/Llama-3.1-8B", path="original/tokenizer.model"
    ),
    "llama-1b": TokenizerConfig(
        repo_id="meta-llama/Llama-3.2-1B", path="original/tokenizer.model"
    ),
    "llama-3b": TokenizerConfig(
        repo_id="meta-llama/Llama-3.2-3B", path="original/tokenizer.model"
    ),
    "gemma": TokenizerConfig(repo_id="google/gemma-2-9b"),
    "gemma-2b": TokenizerConfig(repo_id="google/gemma-2-2b"),
}


def main(tokenizer_name: str, path_to_save: str, api_key: Optional[str] = None):
    config = TOKENIZER_CONFIGS.get(tokenizer_name, None)

    if config is not None:
        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=config.repo_id,
                filename=config.path,
                local_dir=path_to_save,
                local_dir_use_symlinks=False,
                token=api_key if api_key else None,
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                print(
                    "You need to pass a valid `--hf_token=...` to download private checkpoints."
                )
            else:
                raise e
    else:
        from tiktoken import get_encoding

        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            os.environ["TIKTOKEN_CACHE_DIR"] = path_to_save
        try:
            get_encoding(tokenizer_name)
        except ValueError:
            print(
                f"Tokenizer {tokenizer_name} not found. Please check the name and try again."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    main(
        tokenizer_name=args.tokenizer_name,
        path_to_save=args.tokenizer_dir,
        api_key=args.api_key,
    )
