import random, time, torch
from ml.engine import Value
from ml.nn import MLP

random.seed(0)
N_IN, H, BATCH = 50, 200, 32
xs = [[random.uniform(-1, 1) for _ in range(N_IN)] for _ in range(BATCH)]
ys = [random.choice([-1.0, 1.0]) for _ in range(BATCH)]

# your engine: one training step
mlp = MLP(N_IN, [H, H, 1])
t = time.perf_counter()
loss = sum(((mlp(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0))
mlp.zero_grad()
loss.backward()
t_engine = time.perf_counter() - t
print(f"your engine: {t_engine * 1000:.0f} ms per step")

# PyTorch: the same network, the same step
X, Y = torch.tensor(xs), torch.tensor(ys)
Ws = [torch.randn(H, N_IN, requires_grad=True), torch.randn(H, H, requires_grad=True), torch.randn(1, H, requires_grad=True)]
bs = [torch.zeros(H, requires_grad=True), torch.zeros(H, requires_grad=True), torch.zeros(1, requires_grad=True)]

def step():
    h = X
    for W, b in zip(Ws, bs):
        h = torch.tanh(h @ W.T + b)
    l = ((h.squeeze(1) - Y) ** 2).sum()
    for p in Ws + bs:
        p.grad = None
    l.backward()

for _ in range(5):          # warm-up
    step()
t = time.perf_counter()
for _ in range(200):
    step()
t_torch = (time.perf_counter() - t) / 200
print(f"pytorch:     {t_torch * 1e6:.0f} µs per step  → ×{t_engine / t_torch:,.0f} faster")