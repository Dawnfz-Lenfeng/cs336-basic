import os
from collections import Counter, defaultdict

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
        parts = re.split(split_pattern, text)
    else:
        parts = [text]

    return Counter(
        pretoken
        for part in parts
        for match in PAT.finditer(part)
        if len(pretoken := tuple(match.group().encode("utf-8"))) >= 2
    )


def pretoken2pair(
    pretoken_counts: Counter[tuple[int, ...]],
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[tuple[int, ...]]]]:
    """Convert pretoken counts to pair counts and build pair-to-pretoken mapping"""
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair2pretoken: defaultdict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for pretoken, count in pretoken_counts.items():
        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_counts[pair] += count
            pair2pretoken[pair].add(pretoken)

    return pair_counts, dict(pair2pretoken)


def pop_most_frequent_pair(
    pair_heap: LazyHeap,
    vocab: list[bytes],
) -> tuple[int, int]:
    """Pop the most frequent pair in the vocabulary"""
    max_pair, max_count = pair_heap.pop()
    vocab_order = (vocab[max_pair[0]], vocab[max_pair[1]])

    pairs_to_restore: list[tuple[int, int]] = []
    while pair_heap:
        top, top_count = pair_heap.top()
        if top_count < max_count:
            break
        # if count is the same, compare their lex order
        if (new_order := (vocab[top[0]], vocab[top[1]])) > vocab_order:
            pairs_to_restore.append(max_pair)
            max_pair, vocab_order = top, new_order
        else:
            pairs_to_restore.append(top)
        pair_heap.pop()

    for pair in pairs_to_restore:
        pair_heap[pair] = max_count

    return max_pair


def merge_pair(
    pretoken_counts: Counter[tuple[int, ...]],
    pair_heap: LazyHeap,
    pair_to_merge: tuple[int, int],
    new_token: int,
    pair2pretoken: dict[tuple[int, int], set[tuple[int, ...]]],
):
    """Merge a pair of tokens in the pretoken counts, updating the counts of the new and adjacent pairs"""
    items_to_merge = [
        (pretoken, pretoken_counts[pretoken])
        for pretoken in pair2pretoken[pair_to_merge]
    ]

    for pretoken, count in items_to_merge:
        new_pretoken, pair_delta = _merge_pretoken(
            pretoken, count, pair_to_merge, new_token
        )

        del pretoken_counts[pretoken]
        # filter len(pretoken) < 2
        if len(new_pretoken) >= 2:
            pretoken_counts[new_pretoken] += count
            _update_pair2pretoken(new_pretoken, pretoken, pair2pretoken)

        for pair, delta_count in pair_delta:
            pair_heap[pair] += delta_count


def _update_pair2pretoken(
    new_pretoken: tuple[int, ...],
    old_pretoken: tuple[int, ...],
    pair2pretoken: dict[tuple[int, int], set[tuple[int, ...]]],
):
    """Update pair2pretoken mapping when a pretoken is replaced"""
    # Remove old pretoken from all its pairs
    for pair in zip(old_pretoken[:-1], old_pretoken[1:]):
        # A pretoken may have mutiple same pair
        # So when one is removed from pair2pretoken, another does not need to be removed
        if pair not in pair2pretoken:
            continue

        pair2pretoken[pair].discard(old_pretoken)
        if not pair2pretoken[pair]:  # Remove empty sets
            del pair2pretoken[pair]

    # Add new pretoken to its pairs
    for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
        if pair not in pair2pretoken:
            pair2pretoken[pair] = set()
        pair2pretoken[pair].add(new_pretoken)


def _merge_pretoken(
    pretoken: tuple[int, ...],
    count: int,
    pair_to_merge: tuple[int, int],
    new_token: int,
) -> tuple[tuple[int, ...], list[tuple[tuple[int, int], int]]]:
    new_pretoken = []
    pair_delta = []

    i = 0
    while i < len(pretoken):
        if i + 1 < len(pretoken) and (pretoken[i], pretoken[i + 1]) == pair_to_merge:
            # left adjacent pair
            if i > 0:
                # if left pretoken has been merged,
                # its right adjacent pair is just current left adjacent pair
                if new_pretoken[-1] != new_token:
                    pair_delta.append(((pretoken[i - 1], pretoken[i]), -count))
                pair_delta.append(((pretoken[i - 1], new_token), count))

            # right adjacent pair
            if i + 2 < len(pretoken):
                pair_delta.append(((pretoken[i + 1], pretoken[i + 2]), -count))
                pair_delta.append(((new_token, pretoken[i + 2]), count))

            new_pretoken.append(new_token)
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1

    return tuple(new_pretoken), pair_delta
