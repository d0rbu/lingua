# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Type, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.optim.optimizer
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)

from lingua.distributed import is_master
from lingua.train import TrainState

logger = logging.getLogger("CHECKPOINT")

SAVE_DIR_NAME_TEMPLATE = "{:010d}"
RE_SAVE_DIR = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATED_FOLDER = "consolidated"
CONSOLIDATED_CKPT_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME_TEMPLATE = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")


@dataclass
class SaveConfig:
    every: int = 1000
    keep_last: int = 0
    # Keep the last n checkpoints
    # If 0, keep all


@dataclass
class CheckpointArgs:
    dump: SaveConfig = field(default_factory=SaveConfig)
    eval: SaveConfig = field(default_factory=SaveConfig)
    path: str | None = None
    init_ckpt_path: str | None = None


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    config_path = Path(ckpt_dir) / CONFIG_NAME
    consolidated_dir = Path(ckpt_dir) / CONSOLIDATED_FOLDER
    consolidated_ckpt_path = consolidated_dir / CONSOLIDATED_CKPT_NAME
    consolidated_config_path = consolidated_dir / CONFIG_NAME

    if not consolidated_ckpt_path.exists():
        consolidated_dir.mkdir(exist_ok=True)

        logger.info(f"Consolidating to: {consolidated_dir}")
        dcp_to_torch_save(ckpt_dir, str(consolidated_ckpt_path))

        consolidated_config_path.write_text(config_path.read_text())
        logger.info("Consolidated !")

    return consolidated_dir

T = TypeVar("T", bound="CheckpointManager")


class CheckpointManager:
    def __init__(self: Self, args: CheckpointArgs) -> None:
        self.path = args.path
        self.dump = args.dump
        self.eval = args.eval
        self.init_ckpt_path = args.init_ckpt_path

        assert os.path.exists(self.path), (
            f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"
        )

        self.existing_saves = self.get_existing_saves()

    def get_existing_saves(self: Self) -> list[Path]:
        save_dirs = [
            subdirectory
            for subdirectory in Path(self.path).iterdir()
            if subdirectory.is_dir() and re.match(RE_SAVE_DIR, subdirectory.name)
        ]
        save_dirs.sort(key=lambda save_dir: self._extract_step(save_dir.name))

        return save_dirs

    def clean_up(self: Self) -> None:
        logger.info("Cleaning up checkpoints...")

        dump_folders = []
        eval_folders = []
        other_folders = []
        for save_dir in self.existing_saves:
            is_dump_dir = self._extract_step(save_dir.name) % self.dump.every == 0
            is_eval_dir = self._extract_step(save_dir.name) % self.eval.every == 0

            if is_dump_dir:
                dump_folders.append(save_dir)
            if is_eval_dir:
                eval_folders.append(save_dir)
            if not is_dump_dir and not is_eval_dir:
                other_folders.append(save_dir)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump.keep_last > 0:
            dump_folders = dump_folders[-self.dump.keep_last :]
        if self.eval.keep_last > 0:
            eval_folders = eval_folders[-self.eval.keep_last :]

        folders_to_keep = set(other_folders + dump_folders + eval_folders)
        folders_to_remove = set(self.existing_saves) - folders_to_keep

        logger.info(f"Removing folders: {folders_to_remove}")

        if dist.get_rank() == 0:
            for save_dir in folders_to_remove:
                for file in save_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        consolidated_dir = file
                        assert consolidated_dir.name == CONSOLIDATED_FOLDER, (
                            f"{consolidated_dir} is not a consolidated directory"
                        )

                        for f in consolidated_dir.iterdir():
                            f.unlink()
                        consolidated_dir.rmdir()
                save_dir.rmdir()

        dist.barrier()

        self.existing_saves = list(folders_to_keep)
        self.existing_saves.sort(key=lambda save_dir: self._extract_step(save_dir.name))

    def get_last_step_path(self: Self, dp_rank: int = 0) -> Path | None:
        for save_dir in reversed(self.existing_saves):
            train_state_path = save_dir / TRAIN_STATE_NAME_TEMPLATE.format(dp_rank)

            if train_state_path.is_file():
                return save_dir

        return None

    @staticmethod
    def _create_folder(base_path: Path, folder_name: str) -> Path:
        folder_path = base_path / folder_name
        if is_master():
            folder_path.mkdir(parents=False, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        return folder_path

    @staticmethod
    def _extract_step(dir_name: str) -> int:
        return int(re.findall(RE_DIGITS, dir_name)[-1])

    @staticmethod
    def _get_dp_tp_mesh(device_mesh: DeviceMesh | None = None) -> tuple[int, int]:
        dp_rank = 0
        tp_rank = 0

        if device_mesh is None:
            return dp_rank, tp_rank

        if "dp_replicate" in device_mesh.mesh_dim_names:
            dp_rank = device_mesh.get_local_rank("dp_replicate")

            if "dp_shard" in device_mesh.mesh_dim_names:
                dp_rank = dp_rank * device_mesh["dp_shard"].size() + device_mesh.get_local_rank("dp_shard")

        if "tp" in device_mesh.mesh_dim_names:
            tp_rank = device_mesh.get_local_rank("tp")

        return dp_rank, tp_rank

    @torch.no_grad()
    @staticmethod
    def get_state_dict(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, dict]:
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return dict(model=model_sd, optim=optim_sd)

    def save(
        self: Self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_state: TrainState,
        config: dataclass,
        device_mesh: DeviceMesh | None = None,
    ) -> bool:
        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        current_save_dir = self._create_folder(
            path, SAVE_DIR_NAME_TEMPLATE.format(train_state.step)
        )
        logger.info(f"Saving to: {str(current_save_dir)}")

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saving...")
        state_dict = self.get_state_dict(model, optimizer)
        dcp.save(state_dict, checkpoint_id=current_save_dir)
        logger.info("State dict saved!")

        if dist.is_initialized():
            dist.barrier()

        if is_master():
            with open(current_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME_TEMPLATE.format(dp_rank)
            train_state_path = current_save_dir / train_state_name
            logger.info(f"Saving train state to: {train_state_path}")
            with open(train_state_path, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(current_save_dir)

        self.clean_up()

        if dist.is_initialized():
            dist.barrier()

        return True

    @torch.no_grad()
    def load(
        self: Self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_state: TrainState | None,
        device_mesh: DeviceMesh,
        path: Path | None = None,
    ) -> bool:
        dp_rank, _ = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available then the last saved step and finally from the init path
        path = (
            path
            or self.get_last_step_path(dp_rank=dp_rank)
            or Path(self.init_ckpt_path)
        )
        loading_from_init = path == Path(self.init_ckpt_path)

        if path is None:
            logger.info("No checkpoint found, skipping load")
            return False
        elif loading_from_init:
            logger.info(
                "Found initialization path, loading model (no optimizer or train state)"
            )

        # Only load train state if it's provided, the files exist, and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME_TEMPLATE.format(dp_rank)
        train_state_path = path / train_state_name
        if (
            train_state is not None
            and train_state_path.is_file()
            and not loading_from_init
        ):
            logger.info("Reloading train state")
            with open(train_state_path, "r") as train_state_file:
                train_state_dict = json.load(train_state_file)
            train_state.load_state_dict(train_state_dict)
            logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")

        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        state_dict_to_load = state_dict["model"] if loading_from_init else state_dict
        dcp.load(state_dict_to_load, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")

        if loading_from_init:
            model.rope_embeddings.reset_parameters()  # For RoPe initialization since it's a buffer it might not be loaded

        return True

    @classmethod
    def instantiate_and_make_dir(cls: Type[T], args: CheckpointArgs) -> T:
        if is_master():
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()

        return cls(args)
