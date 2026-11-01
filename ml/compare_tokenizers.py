import time

import tiktoken

from ml.logs import generate_logs
from ml.tokenizer import RegexTokenizer

lines = generate_logs(6000, seed=42)
train_text = "\n".join(lines[:5000])  # the tokenizer LEARNS from these
test_text = "\n".join(lines[5000:])  # ...and is SCORED on these, which it has never seen

print("example line:", lines[1])


def ratio(text: str, n_tokens: int) -> float:
    """Compression ratio = how many bytes one token covers on average. Higher = fewer tokens = cheaper."""
    return len(text.encode("utf-8")) / n_tokens


# baseline: one token per byte
n_bytes = len(test_text.encode("utf-8"))
print(f"raw bytes             ratio {ratio(test_text, n_bytes):.2f}")

# ours: trained on the logs
t0 = time.perf_counter()
ours = RegexTokenizer()
ours.train(train_text, vocab_size=512)
train_secs = time.perf_counter() - t0
n_tokens_ours = len(ours.encode(test_text))
print(f"ours (512 tokens)     ratio {ratio(test_text, n_tokens_ours):.2f}   trained in {train_secs:.1f}s")

# GPT-4's tokenizer (~100,000 tokens in its vocabulary)
t1 = time.perf_counter()
enc = tiktoken.get_encoding("cl100k_base")
train_secs_GPT = time.perf_counter() - t1
n_tokens_gpt4 = len(enc.encode(test_text))
print(f"GPT-4 (100k tokens)   ratio {ratio(test_text, n_tokens_gpt4):.2f}  trained in {train_secs_GPT:.1f}s")

# how much does one request ID cost?
uid = "32e70629-8fad-c1a6-06cb-0fb39a1de644"
print(f"\nUUID: {len(uid)} chars → {len(enc.encode(uid))} GPT-4 tokens, {len(ours.encode(uid))} of ours")
