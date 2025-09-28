import json
from typing import Iterable, Iterator

import regex as re

from .train_bpe import PAT


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            # Ensure longer tokens are matched first
            special_tokens.sort(key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
            )
            self.special_token_set = set(special_tokens)
        else:
            self.special_pattern = None
            self.special_token_set = set()

        self.encoder = {v: k for k, v in vocab.items()}
        self.ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Return a Tokenizer from a serialized vocabulary
        and list of merges(in the same format that your BPE training code output)
        and (optionally) a list of special tokens."""
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)

        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.read().split("\n")

        merges = [
            tuple(token.encode("utf-8") for token in line.split()) for line in lines
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        if not text:
            return []

        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_token_set:
                ids.append(self.encoder[part.encode("utf-8")])
            elif part:
                pretokens = [
                    [bytes([b]) for b in match.group().encode("utf-8")]
                    for match in PAT.finditer(part)
                ]
                ids.extend(
                    self.encoder[token]
                    for pretoken in pretokens
                    for token in self._merge_pretoken(pretoken)
                )

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle)
        return a generator that lazily yields token IDs
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        if not ids:
            return ""

        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

    def _merge_pretoken(self, pretoken: list[bytes]) -> list[bytes]:
        """Apply merging to pretoken in the order of creation during BPE training"""
        if len(pretoken) <= 1:
            return pretoken

        while len(pretoken) > 1:
            pair_to_merge = min(
                zip(pretoken[:-1], pretoken[1:]),
                key=lambda pair: self.ranks.get(pair, float("inf")),
            )

            if pair_to_merge not in self.ranks:
                break

            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if (
                    i + 1 < len(pretoken)
                    and (pretoken[i], pretoken[i + 1]) == pair_to_merge
                ):
                    new_pretoken.append(pretoken[i] + pretoken[i + 1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            pretoken = new_pretoken

        return pretoken
