# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import torch.distributed
import logging
from typing import Self, Generator

from torch.profiler.profiler import profile
import xformers.profiler
from xformers.profiler import (
    _Profiler,
    MemSnapshotsProfiler,
    PyTorchProfiler,
)

from lingua.distributed import is_master

import wandb


@dataclass
class ProfilerArgs:
    run: bool = False
    trace_folder: str = "profiling"
    mem_warmup: int = 100
    mem_steps: int = 2
    profile_warmup: int = 102
    profile_steps: int = 2

TRACER_VIEWER_EMBEDDER_PATH = "html/trace_viewer_embedder.html"
TRACER_VIEWER_FULL_PATH = "html/trace_viewer_full.html"


logger = logging.getLogger()


def perfetto_to_html(json_filepath: Path, html_filepath: Path) -> None:
    import viztracer
    import gzip
    import string

    root = os.path.dirname(viztracer.__file__)
    substitutions = {}

    tracer_viewer_embedder_path = os.path.join(root, TRACER_VIEWER_EMBEDDER_PATH)
    with open(tracer_viewer_embedder_path, encoding="utf-8") as f:
        template = string.Template(f.read())

    tracer_viewer_full_path = os.path.join(root, TRACER_VIEWER_FULL_PATH)
    with open(tracer_viewer_full_path, encoding="utf-8") as f:
        substitutions["trace_viewer_full"] = f.read()

    json_file = gzip.open(json_filepath) if ".gz" in str(json_filepath) else open(json_filepath)
    with json_file as j:
        content = j.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        substitutions["json_data"] = content.replace("</script>", "<\\/script>")

    with open(html_filepath, "w+", encoding="utf-8") as output_file:
        output_file.write(template.substitute(substitutions))


class PyTorchProfilerWandb(PyTorchProfiler):
    def __init__(self: Self, main_profiler: _Profiler) -> None:
        self.main_profiler = main_profiler
        self.num_steps = 0
        self.pytorch_profiler = torch.profiler.profile(
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            # With stack gives huge profile traces
            # and bugs out because of some non ascii
            # character somewhere in pytorch
            with_stack=False,
            with_flops=True,
            activities=self.ACTIVITIES,
        )

    def _analyze_trace(self: Self, prof: profile) -> None:
        logger.info("Begin analyze trace")
        super()._analyze_trace(prof)
        logger.info("End analyze trace")

    def _on_trace(self: Self, prof: profile) -> None:
        super()._on_trace(prof)

        if is_master() and wandb.run is not None:
            filename = list(
                Path(self.main_profiler.output_dir).glob(
                    "profile_CPU_CUDA*/*.pt.trace.json*"
                )
            )[0]
            html_path = str(filename).replace(".json", ".html")
            perfetto_to_html(filename, html_path)
            wandb.log({"profile_trace": wandb.Html(html_path)})


class MemSnapshotsProfilerWandb(MemSnapshotsProfiler):
    def __exit__(self: Self, exc_type, exc_val, exc_tb) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)

        if is_master() and wandb.run is not None:
            filename = list(
                Path(self.main_profiler.output_dir).glob("memory_trace_plot/*.html")
            )[0]
            wandb.log({"memory_trace": wandb.Html(open(filename), inject=False)})


@contextlib.contextmanager
def maybe_run_profiler(dump_dir: str, module: torch.nn.Module, config: ProfilerArgs) -> Generator
    # get user defined profiler settings

    if config.run:
        trace_dir = os.path.join(dump_dir, config.trace_folder)

        logger.info(f"Profiling active.  Traces will be saved at {trace_dir}")

        if is_master() and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        with xformers.profiler.profile(
            output_dir=trace_dir,
            module=module,
            schedule=[
                (
                    MemSnapshotsProfilerWandb,
                    config.mem_warmup,
                    config.mem_warmup + config.mem_steps,
                ),
                (
                    PyTorchProfilerWandb,
                    config.profile_warmup,
                    config.profile_warmup + config.profile_steps,
                ),
            ],
        ) as profiler:
            yield profiler

    else:
        torch_profiler = contextlib.nullcontext()
        yield None
