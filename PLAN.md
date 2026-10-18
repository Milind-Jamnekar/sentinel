# Sentinel: the plan

**The goal:** build an autonomous on-call SRE agent from first principles. You start from the math that makes neural networks learn and end with a multi-tenant, crash-safe agent platform.

**How we work:** one step at a time. I explain the idea, write half the code, and you write the other half. You run it and send it to me, and I review it. Predictions and brain-challenges are part of each step, not optional extras.

Legend: ✅ done · 🟡 partly done · 👉 current · ⬜ not started

```
Part I  — How models work      (Python)   Modules 1–4    ← YOU ARE HERE
Part II — Retrieval            (TS)       Modules 5–7
Part III — LLM platform         (NestJS)   Modules 8–9
Part IV — Agents                (NestJS)   Modules 10–14  → Sentinel ships
```

---

## Part I: How models work (Python, `ml/`)

### Module 1: Autograd engine 👉
*What you're learning: derivatives, the chain rule, backprop, gradient descent. This is the mechanism that trains every model.*

| Step | What | Status |
|---|---|---|
| 1 | `Value` node: `data`, `grad`, `_prev`, `_op`, `_backward` | ✅ |
| 2 | Operators build the graph (`__add__`, `__mul__`) | ✅ |
| 3 | Local derivatives: the `_backward` closures | ✅ |
| 4 | `backward()` with a topological sort; the diamond-bug lesson | ✅ |
| 5 | Complete `Value`: plain numbers, `- / **`, `exp`, `tanh`, `relu` | ✅ |
| 6 | Gradient-checker tests | ✅ 8/8 green |
| 7 | `Neuron`, `Layer`, `MLP` in `nn.py` | ✅ 15/15 green |
| 8 | Training loop: forward, loss, `zero_grad`, backward, update. Reach loss < 0.01 | ✅ 18/18 green, final loss 0.0029 |
| 9 | Hard extras: iterative topological sort for a 100k-node chain, why central difference is more accurate, find the learning rate where training diverges | 👉 |
| 10 | Review and theory: how PyTorch scales this to billions of parameters (tensors, vectorized operations, GPUs) | ⬜ |

**Done when:** an MLP trained only by your engine reaches loss < 0.01, and every gradient passes the numerical check.

### Module 2: BPE tokenizer ⬜
*What you're learning: what tokens are, and why they drive cost, latency and the context limit.*
1. Strings, Unicode and UTF-8 bytes: why byte-level encoding.
2. Count adjacent pairs and merge the most frequent one; repeat for N merges.
3. `encode()` and `decode()` that round-trip any string, including emoji.
4. Special tokens (`<|endoftext|>`) and a regex pre-split, the way GPT-style tokenizers work.
5. Train it on log data and measure the compression ratio. Compare with `tiktoken`.

**Done when:** `decode(encode(s)) == s` for random Unicode, and your tokenizer compresses logs better than raw bytes do.

### Module 3: GPT from scratch, trained on logs ⬜
*What you're learning: tensors, embeddings, attention, residual connections, layernorm, training loops in PyTorch.*
1. PyTorch tensors: rebuild Module 1's MLP with `torch` and compare the gradients with yours.
2. Generate the dataset: a synthetic log corpus (the start of the Arena).
3. Bigram model, then embeddings and positional embeddings.
4. Self-attention by hand (Q, K, V, causal mask, softmax), then multi-head attention.
5. Transformer block (attention + MLP + residual + layernorm), stacked N times.
6. Training loop on MPS (your M2's GPU): train/validation split, loss curves, overfitting.

**Done when:** your model generates realistic-looking log lines and its validation loss is clearly better than the bigram model's.

### Module 4: Inference engine and anomaly detector ⬜
*What you're learning: how LLMs are served. Prefill vs decode, the KV cache, time-to-first-token (TTFT), throughput.*
1. Sampling: greedy, temperature, top-k, top-p.
2. KV cache, then benchmark the speedup.
3. Perplexity: how surprised the model is by a log line.
4. Serve it as an API (FastAPI): `POST /score` returns an anomaly score per line.
5. Benchmark TTFT, tokens/sec, and the effect of batching.

**Done when:** injected anomalous log lines score clearly higher than normal ones. This becomes **Sentinel's first tool**.

---

## Part II: Retrieval (TypeScript, `platform/`)

### Module 5: Embeddings and an HNSW vector index from scratch ⬜
1. Embeddings and cosine similarity. Brute-force k-nearest-neighbour search as the baseline.
2. HNSW: a layered graph, greedy search, inserting nodes.
3. Benchmark recall@10 against latency while tuning `M` and `efSearch`.

**Done when:** recall@10 ≥ 0.95 while running at least 10× faster than brute force on 100k vectors.

### Module 6: Hybrid RAG on Postgres ⬜
1. Arena runbooks and past incident reports as the corpus. Chunking strategies.
2. Idempotent ingestion pipeline: BullMQ, content hashing, re-embed only what changed.
3. pgvector plus Postgres full-text search, combined with reciprocal rank fusion (RRF).
4. Reranking. Metadata filters.
5. Multi-tenancy with row-level security (RLS). Drizzle ORM.

**Done when:** a tenant can never retrieve another tenant's chunks (proven by a test), and hybrid search beats vector-only search on your eval set.

### Module 7: Evals ⬜
1. Golden dataset: questions, the expected chunks, the expected answers.
2. Retrieval metrics: recall@k, MRR, nDCG.
3. LLM-as-judge for faithfulness, and calibrating the judge against your own labels.
4. A CI gate that fails the build when quality drops.

**Done when:** every change to the RAG pipeline produces a score diff in CI.

---

## Part III: The LLM platform (NestJS)

### Module 8: LLM gateway ⬜
1. One provider interface over Claude and Ollama.
2. Streaming, normalized into SSE.
3. Structured output: Zod schema, then validate, then a repair loop.
4. Resilience: retries on 429/529 with backoff and jitter, a circuit breaker, fallback to Ollama.
5. Prompt caching. Token and cost accounting per tenant. Token-bucket budgets in Redis.

**Done when:** you kill the Claude connection mid-stream and the request still completes on Ollama, with cost recorded.

### Module 9: Security and caching ⬜
1. Prompt injection: logs are attacker-controlled. Build a red-team suite.
2. Defenses: trust boundaries, spotlighting untrusted input, tool allow-lists, output checks.
3. PII redaction.
4. Semantic cache: threshold tuning, tenant isolation, invalidation.

**Done when:** the red-team suite triggers zero unauthorized actions, and the cache never serves an answer across tenants.

---

## Part IV: Agents (NestJS)

### Module 10: Agent loop from scratch ⬜
Tool calling, parallel tools, stop conditions, step and cost budgets, context compaction when the window fills. No frameworks.

### Module 11: Durable runtime (a small Temporal) ⬜
Event-sourced steps in Postgres, resume after `kill -9`, idempotency keys for side-effecting tools, human-approval gates that can pause for hours, cancellation.

### Module 12: MCP and sandboxed tools ⬜
Sentinel tools exposed as an MCP server, consuming external MCP servers, per-tool permissions, sandboxed execution.

### Module 13: Fine-tuning vs prompting ⬜
LoRA fine-tune a small model for alert triage on your M2. Use Module 7's evals to prove whether it beats prompting.

### Module 14: Capstone, Sentinel end-to-end ⬜
- The Arena: 3 NestJS services + Postgres + Redis + Prometheus/Loki + a chaos injector.
- A planner agent plus investigator sub-agents.
- A benchmark of 30 simulated incidents with known root causes.
- OpenTelemetry traces and a docker-compose deploy.

**Final boss: Sentinel is done when all of these hold.**
- [ ] Root cause correct on ≥ 80% of 30 simulated incidents
- [ ] `kill -9` the worker mid-investigation: it resumes and never repeats a side-effecting tool call
- [ ] Zero unauthorized actions on the prompt-injection red-team suite
- [ ] < $0.50 per investigation, time-to-first-token < 1.5s at p95
- [ ] With the Claude API blocked, it falls back to Ollama
- [ ] A full trace of every LLM call, tool call, token and cent

---

## Visual explainers

`visuals/` is a local-only Remotion project with animated explanations (git-ignored, not published). Run `cd visuals && npm run dev`.
- `Module1-Autograd`: computation graph, forward pass, backward pass, the diamond bug, the neuron, the MLP, the training loop.
- `Neuron-DeepDive` (~3 min, from zero): weights, the dot product, bias, why we squash, tanh, tanh's slope and saturation, a worked example, learning by gradient descent, layers.
- `Training-Loop` (~3.5 min, for Step 8): the data, forward, loss (sum of squared errors) and why we square, one graph for all examples, zero_grad, backward, the update rule, learning rate, and a full 40-round run. Every number comes from a real run of `ml/` (seed 3, lr 0.1).
