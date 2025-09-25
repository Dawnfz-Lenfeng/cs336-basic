import os
from collections import Counter, defaultdict

import regex as re

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        text = re.sub(split_pattern, " ", text)

    pretokens = PAT.finditer(text)
    return Counter(
        pretoken
        for match in pretokens
        if len(pretoken := tuple(match.group().encode("utf-8"))) >= 2
    )


def get_most_frequent_pair(
    pair_counts: Counter[tuple[int, int]],
    vocab: dict[int, bytes],
) -> tuple[int, int]:
    return max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], [vocab[token] for token in pair]),
    )


def merge_pair(
    pretoken_counts: Counter[tuple[int, ...]],
    pair_counts: Counter[tuple[int, int]],
    pair_to_merge: tuple[int, int],
    new_token: int,
):
    items_to_update = []
    for pretoken, count in pretoken_counts.items():
        new_pretoken = []
        need_update = False

        i = 0
        while i < len(pretoken):
            if (
                i + 1 < len(pretoken)
                and (pretoken[i], pretoken[i + 1]) == pair_to_merge
            ):
                need_update = True
                new_pretoken.append(new_token)

                if i > 0:
                    adj_pair = (pretoken[i - 1], pretoken[i])
                    pair_counts[adj_pair] -= count
                    if pair_counts[adj_pair] <= 0:
                        del pair_counts[adj_pair]
                    pair_counts[(pretoken[i - 1], new_token)] += count
                if i + 2 < len(pretoken):
                    adj_pair = (pretoken[i + 1], pretoken[i + 2])
                    pair_counts[adj_pair] -= count
                    if pair_counts[adj_pair] <= 0:
                        del pair_counts[adj_pair]
                    pair_counts[new_token, pretoken[i + 2]] += count

                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        if need_update:
            items_to_update.append((pretoken, tuple(new_pretoken), count))

    for pretoken, new_pretoken, count in items_to_update:
        del pretoken_counts[pretoken]
        pretoken_counts[new_pretoken] += count
