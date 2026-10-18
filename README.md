# Sentinel

Learning AI engineering from first principles by building **Sentinel**: an autonomous on-call SRE agent that investigates incidents, finds root causes, and asks a human before doing anything risky.

The path goes from the math that makes neural networks learn, up to a multi-tenant, crash-safe agent platform. See [PLAN.md](PLAN.md) for all 14 modules and current progress.

## What's here so far

`ml/` is Module 1: an **autograd engine and neural network, written from scratch** in pure Python (no numpy, no PyTorch).

| File | What it is |
|---|---|
| `ml/src/ml/engine.py` | `Value`: a scalar that remembers how it was computed, plus `backward()` (reverse-mode autodiff, with a topological sort) |
| `ml/src/ml/nn.py` | `Neuron`, `Layer`, `MLP`, built on `Value` |
| `ml/src/ml/train.py` | A training loop: forward → loss → zero_grad → backward → update |
| `ml/tests/` | Gradient checks against numerical derivatives, plus network and training tests |

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd ml
uv run pytest                    # all tests
uv run python src/ml/train.py    # train an MLP(3, [4, 4, 1]) until the loss is below 0.01
```
