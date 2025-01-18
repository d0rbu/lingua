# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass

import requests
from huggingface_hub import snapshot_download


@dataclass
class DatasetConfig:
    repo_id: str
    orig_extension: str = ".jsonl"
    cat_command: str = "cat"
    allow_patterns: str | None = None


DATASET_CONFIGS = {
    "fineweb_edu": DatasetConfig(
        repo_id="HuggingFaceFW/fineweb-edu",
    ),
    "fineweb_edu_10bt": DatasetConfig(
        repo_id="HuggingFaceFW/fineweb-edu",
        allow_patterns="sample/10BT/*",
    ),
    "dclm_baseline_1.0": DatasetConfig(
        repo_id="mlfoundations/dclm-baseline-1.0",
        orig_extension=".jsonl.zst",
        cat_command="zstdcat",
        allow_patterns="*.jsonl.zst",
    ),
    "dclm_baseline_1.0_10prct": DatasetConfig(
        repo_id="mlfoundations/dclm-baseline-1.0",
        orig_extension=".jsonl.zst",
        cat_command="zstdcat",
        allow_patterns="global-shard_01_of_10/*.jsonl.zst",
    ),
    "meta_math_qa": DatasetConfig(
        repo_id="meta-math/MetaMathQA",
    ),
}


def run_command(command: str) -> None:
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(
    repo_id: str, local_dir: str, allow_patterns: str | None = None
) -> None:
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,  # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(
    dataset: str, work_dir: str, src_dir: str, tgt_dir: str, ntasks: int = 64
) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir: str) -> str:
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")

    return terashuf_dir


def json_to_jsonl(src_dir: str, tgt_dir: str) -> None:
    json_filenames = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    json_filepaths = {
        os.path.splitext(f)[0]: os.path.join(src_dir, f) for f in json_filenames
    }
    json_files = {name: open(path, "r") for name, path in json_filepaths.items()}
    json_contents = {name: json.load(file) for name, file in json_files.items()}

    # close files
    for file in json_files.values():
        file.close()

    json_lists = {
        name: content
        for name, content in json_contents.items()
        if isinstance(content, list)
    }
    jsonl_contents = {
        name: "\n".join(json.dumps(item) for item in content)
        for name, content in json_lists.items()
    }

    for name, content in jsonl_contents.items():
        with open(os.path.join(tgt_dir, f"{name}.jsonl"), "w") as f:
            f.write(content)


def convert_to_jsonl(dataset: str, work_dir: str, data_dir: str) -> None:
    if "fineweb" in dataset:
        parquet_to_jsonl(dataset, work_dir, data_dir, data_dir)
        return

    if "meta_math_qa" in dataset:
        json_to_jsonl(data_dir, data_dir)
        return


def main(dataset: str, memory: float, data_dir: str, seed: int = 42, nchunks: int = 32):
    # Configuration
    config = DATASET_CONFIGS.get(dataset, None)

    assert config is not None, f"Dataset {dataset} not found in DATASET_CONFIGS"

    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)

    work_dir = src_dir  # Directory of this Python file

    prefix = f"{dataset}.chunk."
    suffix = ".jsonl"

    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    download_dataset(config.repo_id, src_dir, config.allow_patterns)

    # Convert to JSONL if needed
    convert_to_jsonl(dataset, work_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = str(memory)
    os.environ["SEED"] = str(seed)

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{config.orig_extension}' -print0 | xargs -0 {config.cat_command} | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )

    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8.0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks)
