import os
from collections import Counter

import regex as re
from .lazy_heap import LazyHeap

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    """Transform text into a list of pretokens"""
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        chunks = re.split(split_pattern, text)
    else:
        chunks = [text]

    counter = Counter()
    for chunk in chunks:
        pretokens = [
            tuple(match.group().encode("utf-8")) for match in PAT.finditer(chunk)
        ]
        counter.update(
            Counter(pretoken for pretoken in pretokens if len(pretoken) >= 2)
        )

    return counter


def pretoken2pair(
    pretoken_counts: Counter[tuple[int, ...]],
) -> Counter[tuple[int, int]]:
    """Convert pretoken counts to pair counts"""
    pair_counts: Counter[tuple[int, int]] = Counter()

    for pretoken, count in pretoken_counts.items():
        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_counts[pair] += count

    return pair_counts


def pop_most_frequent_pair(
    pair_heap: LazyHeap,
    vocab: list[bytes],
) -> tuple[int, int]:
    """Get the most frequent pair in the vocabulary"""
    pair, max_count = pair_heap.pop()
    pairs = [pair]

    while pair_heap:
        top, top_count = pair_heap.top()
        if top_count < max_count:
            break
        pairs.append(top)
        pair_heap.pop()

    if len(pairs) == 1:
        return pairs[0]
    else:
        max_pair = max(pairs, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        for pair in pairs:
            if pair != max_pair:
                pair_heap[pair] = max_count
        return max_pair


def merge_pair(
    pretoken_counts: Counter[tuple[int, ...]],
    pair_heap: LazyHeap,
    pair_to_merge: tuple[int, int],
    new_token: int,
):
    """Merge a pair of tokens in the pretoken counts, updating the counts of the new and adjacent pairs"""
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

                # left adjacent pair
                if i > 0:
                    # if left pretoken has been merged,
                    # its right adjacent pair is just current left adjacent pair
                    if new_pretoken[-1] != new_token:
                        pair_heap[(pretoken[i - 1], pretoken[i])] -= count
                    pair_heap[(pretoken[i - 1], new_token)] += count

                # right adjacent pair
                if i + 2 < len(pretoken):
                    pair_heap[(pretoken[i + 1], pretoken[i + 2])] -= count
                    pair_heap[(new_token, pretoken[i + 2])] += count

                new_pretoken.append(new_token)
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1

        if need_update:
            items_to_update.append((pretoken, tuple(new_pretoken), count))

    for pretoken, new_pretoken, count in items_to_update:
        del pretoken_counts[pretoken]
        pretoken_counts[new_pretoken] += count
