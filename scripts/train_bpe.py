from pathlib import Path

from cs336_basics.train_bpe import save_bpe, train_bpe

# 数据集路径和保存路径
INPUT_PATH = Path("data") / "TinyStoriesV2-GPT4-train.txt"
SAVE_DIR = Path("tokenizer")

# 训练参数
vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

# 训练
vocab, merges = train_bpe(
    input_path=INPUT_PATH,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)

save_bpe(vocab, merges, SAVE_DIR)

# 统计最长 token
longest_token = max(vocab.values(), key=len)
print("最长token:", longest_token, "长度:", len(longest_token))
