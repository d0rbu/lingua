import argparse
import os
import json
from tqdm import tqdm, trange

from lingua.tokenizer import build_tokenizer
from lingua.data import QA_PROMPT


QUESTION_KEYS = {"question", "query"}
ANSWER_KEYS = {"answer", "response"}


def main(tokenizer_type: str, tokenizer_path: str, dataset_path: str) -> None:
    tokenizer = build_tokenizer(tokenizer_type, tokenizer_path)

    filenames = os.listdir(dataset_path)
    jsonl_filenames = [f for f in filenames if f.endswith(".jsonl")]

    max_len = 0

    with trange(len(jsonl_filenames)) as t:
        for file_idx, filename in zip(t, jsonl_filenames):
            t.set_description(f"{filename} ({file_idx + 1}/{len(jsonl_filenames)})")
            with open(os.path.join(dataset_path, filename), "r") as f:
                num_lines = sum(1 for _ in f)
                f.seek(0)

                for line_idx, line in zip(trange(num_lines), f):
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    try:
                        question_key = QUESTION_KEYS.intersection(data.keys()).pop()
                        answer_key = ANSWER_KEYS.intersection(data.keys()).pop()
                    except KeyError as e:
                        print(f"Line {line_idx} in {filename} does not have a question or answer key")
                        raise e

                    question = data[question_key]
                    answer = data[answer_key]

                    formatted_question = QA_PROMPT.format(instruction=question)
                    example = f"{formatted_question}{answer}"

                    example_tokens = tokenizer.encode(example, add_bos=False, add_eos=False)
                    current_len = len(example_tokens)

                    if current_len > max_len:
                        t.set_postfix(max_len=current_len)
                        max_len = current_len

    print(f"Max length: {max_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_path", type=str, default="tokenizer/llama-1b/tokenizer.model")
    parser.add_argument("dataset_path", type=str, default="data/meta_math_qa")
    parser.add_argument("--tokenizer_type", type=str, default="tiktoken")
    args = parser.parse_args()

    main(
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path
    )
