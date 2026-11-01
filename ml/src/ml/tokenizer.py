import regex as re
from collections import Counter

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    group = tuple(zip(ids,ids[1:]))
    counts = {}
    for pair in group:
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    i = 0
    new_list = []
    while i < len(ids):
        if i < len(ids) - 1 and ((ids[i], ids[i + 1]) == pair):
            new_list.append(new_id)
            i += 2
        else:
            new_list.append(ids[i])
            i += 1
    return new_list

class BasicTokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the single bytes)")
        ids = list(text.encode("utf-8"))

        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            new_id = 256 + i

            pair = max(stats, key=lambda p : stats[p])

            ids = merge(ids, pair, new_id)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break

            ids = merge(ids,pair,self.merges[pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        chunks = [self.vocab[id] for id in ids]
        data = b"".join(chunks)
        return data.decode('utf-8', errors="replace")

# GPT-2's split rule: words (with their leading space), numbers, punctuation runs, whitespace.
# \p{L} = any letter in any language, \p{N} = any digit. (Needs the `regex` package, not `re`.)
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicTokenizer):
    """BPE that first splits text into chunks, so merges never cross a chunk boundary."""

    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN) -> None:
        super().__init__()  # reuse merges, vocab and decode() from BasicTokenizer
        self.pattern = re.compile(pattern)
        # special string -> its reserved id, e.g. {"<|endoftext|>": 1000}
        self.special_tokens: dict[str, int] = {}

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the single bytes)")
        
        chunk_counts = Counter(self.pattern.findall(text))
        # chunks: list[str] = self.pattern.findall(text)
        chunks = [(list(chunk.encode("utf-8")), freq) for chunk,freq in chunk_counts.items()]
        
        for i in range(vocab_size - 256):
            stats: dict[tuple[int, int], int] = {}

            for ids, freq in chunks:
                for pair, count in get_stats(ids).items():
                    stats[pair] = stats.get(pair,0) + count * freq

            if not stats:
                break

            new_id = 256 + i
            pair = max(stats, key=lambda p: stats[p])

            chunks = [(merge(ids, pair, new_id), freq) for ids, freq in chunks]

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]


    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating EVERYTHING as plain text (no special tokens). Your old encode()."""
        ids: list[int] = []
        chunks = self.pattern.findall(text)
        for chunk in chunks:
            ids.extend(super().encode(chunk))
        return ids

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Reserve ids for special strings, e.g. {"<|endoftext|>": 1000}."""
        for token, token_id in special_tokens.items():
            if token_id in self.vocab:
                raise ValueError(f"id {token_id} for {token!r} is already a normal token")
        self.special_tokens.update(special_tokens)

    def encode(self, text: str, allowed_special: frozenset[str] = frozenset()) -> list[int]:
        """A special string becomes its special id ONLY if it's in `allowed_special`.
        Otherwise it's encoded as plain text: the safe default for untrusted input like logs."""
        specials = {s: self.special_tokens[s] for s in allowed_special}
        print("-...>", specials)
        if not specials:
            return self.encode_ordinary(text)

        # Split around the allowed special strings and KEEP them (that's what the "( … )" group does):
        #   "hi<|endoftext|>there"  →  ["hi", "<|endoftext|>", "there"]
        split_pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
        parts = re.split(split_pattern, text)

        ids: list[int] = []
        for part in parts:
            if part in specials:
                ids.append(specials[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        inverse = {token_id: token for token, token_id in self.special_tokens.items()}
        chunks: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                chunks.append(self.vocab[i])
            elif i in inverse:
                chunks.append(inverse[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {id}")
        return b"".join(chunks).decode("utf-8", errors="replace")