import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import find_chunk_bounds


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess text data for training")
    parser.add_argument(
        "--input",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Input text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.dat",
        help="Output binary file",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="tokenizer/vocab.json",
        help="Vocabulary file",
    )
    parser.add_argument(
        "--merges",
        type=str,
        default="tokenizer/merges.txt",
        help="Merges file",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: CPU count)",
    )
    return parser.parse_args()


# Global variable for worker processes to reuse tokenizer
_worker_tokenizer = None


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str] | None):
    """
    Initialize worker process by loading tokenizer once.
    This is called once per worker process, not per chunk.
    """
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)


def _process_chunk_worker(args: tuple[str, tuple[int, int]]) -> list[int]:
    """
    Worker function to tokenize a single chunk.
    Reuses the tokenizer loaded in _init_worker.
    """
    input_path, (start, end) = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return _worker_tokenizer.encode(chunk)


def tokenize_file(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None = None,
    chunk_size: int = 5 * 1024 * 1024,
    max_workers: int | None = None,
):
    """Tokenize a text file and save as binary file using parallel processing."""
    file_size = Path(input_path).stat().st_size
    print(f"Processing {input_path} ({file_size / 1024 / 1024:.2f} MB)")

    with open(input_path, "rb") as f:
        bounds = find_chunk_bounds(
            f,
            special_tokens[0].encode("utf-8"),
            chunk_size,
        )

    args_list = [(input_path, chunk) for chunk in zip(bounds[:-1], bounds[1:])]

    total_tokens = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens),
    ) as executor:
        with open(output_path, "wb") as f_out:
            for chunk_tokens in tqdm(
                executor.map(_process_chunk_worker, args_list),
                total=len(args_list),
                desc="Tokenizing chunks",
                unit="chunk",
            ):
                tokens_chunk = np.array(chunk_tokens, dtype=np.uint16)
                tokens_chunk.tofile(f_out)
                total_tokens += len(chunk_tokens)

    print(f"Saved {total_tokens:,} tokens to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {file_size / Path(output_path).stat().st_size:.2f}x")


def main():
    args = parse_args()

    tokenize_file(
        args.input,
        args.output,
        args.vocab,
        args.merges,
        args.special_tokens,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
