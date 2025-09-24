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
    return Counter(tuple(match.group().encode("utf-8")) for match in pretokens)


def get_most_frequent_pair(
    pretoken_counts: Counter[tuple[int, ...]],
    vocab: dict[int, bytes],
) -> tuple[int, int]:
    pair_counts = defaultdict(int)

    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue

        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_counts[pair] += count

    return max(
        pair_counts,
        key=lambda pair: (pair_counts[pair], [vocab[token] for token in pair]),
    )


def merge_pair(
    pretoken_counts: Counter[tuple[int, ...]],
    pair_to_merge: tuple[int, int],
    new_token: int,
):
    items_to_update = [
        (pretoken, new_pretoken, count)
        for pretoken, count in pretoken_counts.items()
        if (new_pretoken := _merge_pretoken(pretoken, pair_to_merge, new_token))
        != pretoken
    ]

    for pretoken, new_pretoken, count in items_to_update:
        del pretoken_counts[pretoken]
        pretoken_counts[new_pretoken] += count


def _merge_pretoken(
    pretoken: tuple[int, ...], pair_to_merge: tuple[int, int], new_token: int
) -> tuple[int, ...]:
    new_pretoken = []

    i = 0
    while i < len(pretoken):
        if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == pair_to_merge:
            new_pretoken.append(new_token)
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1

    return tuple(new_pretoken)
