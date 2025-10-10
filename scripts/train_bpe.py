import argparse

from cs336_basics.train_bpe import save_bpe, train_bpe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="data/TinyStoriesV2-GPT4-train.txt"
    )
    parser.add_argument("--save-dir", type=str, default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--special-tokens", type=str, nargs="*", default=["<|endoftext|>"]
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vocab, merges = train_bpe(
        input_path=args.data_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    save_bpe(vocab, merges, args.save_dir)

    # 统计最长 token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token}, Length: {len(longest_token)}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
