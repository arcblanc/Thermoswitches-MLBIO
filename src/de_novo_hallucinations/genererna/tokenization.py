import argparse
import os
import random
from typing import cast

import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for GenerRNA tokenization."""
    parser = argparse.ArgumentParser(
        description="Process the text data for tokenization."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory of the raw data."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the trained AutoTokenizer.",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Directory of output files."
    )
    parser.add_argument("--file_name", type=str, default="data.txt", required=True)
    parser.add_argument("--block_size", type=int, default=512, help="Max token length.")
    parser.add_argument(
        "--is_start_with_eos",
        type=bool,
        default=False,
        help="Whether each line starts with `eos_token`.",
    )
    parser.add_argument(
        "--is_end_with_eos",
        type=bool,
        default=False,
        help="Whether each line ends with `eos_token`.",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.99, help="Train-validation split ratio."
    )
    return parser.parse_args()


def tokenize_and_save_lines(
    tokenizer: PreTrainedTokenizerBase,
    input_file: str,
    train_txt_file: str,
    val_txt_file: str,
    train_bin_file: str,
    val_bin_file: str,
    is_start_with_eos: bool,
    is_end_with_eos: bool,
    block_size: int,
    split_ratio: float,
) -> None:
    """Tokenize shuffled lines, split train/val, and write text and binary files."""
    train_ids = []
    val_ids = []
    train_lines = []
    val_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_at = int(split_ratio * len(lines))
    train_lines_list = lines[:split_at]
    val_lines_list = lines[split_at:]

    for i, line in enumerate(train_lines_list):
        ids = tokenizer.encode(line)
        if not is_end_with_eos:
            ids.append(0)
        elif not is_start_with_eos:
            ids.insert(0, 0)

        if len(ids) < block_size:
            train_ids.extend(ids)
            train_lines.append(line.strip())
        if i % 1000000 == 0:
            print(f"now processing {i}...")

    for i, line in enumerate(val_lines_list):
        ids = tokenizer.encode(line)
        if not is_end_with_eos:
            ids.append(0)
        elif not is_start_with_eos:
            ids.insert(0, 0)

        if len(ids) <= block_size:
            val_ids.extend(ids)
            val_lines.append(line.strip())

    # Save tokenized data
    save_tokenized_data(train_ids, train_bin_file)
    save_tokenized_data(val_ids, val_bin_file)
    print("Tokenized data saved...")

    # Save text data
    save_text_data(train_lines, train_txt_file)
    save_text_data(val_lines, val_txt_file)
    print("Text data saved...")


def save_tokenized_data(tokenized_data: list[int], file_path: str) -> None:
    """Write token ids to a uint16 binary file, creating parent directories."""
    np_data = np.array(tokenized_data, dtype=np.uint16)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np_data.tofile(file_path)


def save_text_data(text_data: list[str], file_path: str) -> None:
    """Write text lines to a UTF-8 file, creating parent directories."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for line in text_data:
            f.write(line + "\n")


def main() -> None:
    """Tokenize a raw text corpus into GenerRNA train and validation files."""
    args = parse_arguments()

    # Paths setup
    raw_data_path = os.path.join(args.data_dir, args.file_name)
    train_txt_path = os.path.join(args.out_dir, "train.txt")
    val_txt_path = os.path.join(args.out_dir, "val.txt")
    train_bin_path = os.path.join(args.out_dir, "train.bin")
    val_bin_path = os.path.join(args.out_dir, "val.bin")
    print("Paths setup complete...")

    # Tokenization
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(args.tokenizer_path),
    )
    tokenize_and_save_lines(
        tokenizer,
        raw_data_path,
        train_txt_path,
        val_txt_path,
        train_bin_path,
        val_bin_path,
        args.is_start_with_eos,
        args.is_end_with_eos,
        args.block_size,
        args.split_ratio,
    )
    print("Tokenization and data saving")


if __name__ == "__main__":
    main()
