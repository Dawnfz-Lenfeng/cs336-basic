import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm

from .lazy_heap import LazyHeap

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_gpt2_bytes_to_str() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


GPT2_BYTES_TO_STR = get_gpt2_bytes_to_str()
GPT2_STR_TO_BYTES = {v: k for k, v in GPT2_BYTES_TO_STR.items()}


def bytes_to_gpt2_str(bytes_: bytes) -> str:
    return "".join(GPT2_BYTES_TO_STR[byte] for byte in bytes_)


def gpt2_str_to_bytes(str_: str) -> bytes:
    return bytes(GPT2_STR_TO_BYTES[s] for s in str_)


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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = [bytes([i]) for i in range(256)] + [
        token.encode("utf-8") for token in special_tokens
    ]
    merges = []

    pretoken_counts = pretokenize(input_path, special_tokens)  # (token, ...) -> count
    pair_counts, pair2pretoken = pretoken2pair(pretoken_counts)
    pair_heap = LazyHeap(dict(pair_counts))

    num_merges = vocab_size - len(vocab)
    for _ in tqdm(range(num_merges)):
        pair = pop_most_frequent_pair(pair_heap, vocab)
        byte1, byte2 = map(lambda x: vocab[x], pair)

        merges.append((byte1, byte2))
        vocab.append(byte1 + byte2)

        merge_pair(pretoken_counts, pair_heap, pair, len(vocab) - 1, pair2pretoken)

    return {i: token for i, token in enumerate(vocab)}, merges


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    save_path: str | os.PathLike,
    gpt2_style: bool = True,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
        if gpt2_style:
            vocab = {bytes_to_gpt2_str(v): k for k, v in vocab.items()}
        else:
            vocab = {v.decode("utf-8"): k for k, v in vocab.items()}
        json.dump(vocab, f)

    with open(save_path / "merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            if gpt2_style:
                f.write(" ".join(bytes_to_gpt2_str(m) for m in merge) + "\n")
            else:
                f.write(" ".join(m.decode("utf-8") for m in merge) + "\n")
