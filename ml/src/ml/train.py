import random
from ml.nn import MLP
from ml.engine import Value

# ---- the data: 4 examples with the right answers ----
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # +1 = "page", -1 = "all fine"


def train(steps: int = 100, lr: float = 0.1, seed: int = 3) -> float:
    """Train a fresh MLP on xs/ys. Returns the final loss."""
    random.seed(seed)  # same seed = same random starting knobs = same numbers as the video
    mlp = MLP(3, [4, 4, 1])

    loss = None
    last_loss = float("inf")
    for step in range(steps):
        # 1. FORWARD: one prediction per example
        #    hint: call mlp(x) for every x in xs, collect them in a list
        ypred = [mlp(x) for x in xs] # TODO

        # 2. LOSS: sum of squared errors, one number for "how wrong"
        #    hint: pair each prediction with its target (zip),
        #          compute (prediction - target) ** 2, then sum them all

        loss = sum(((y - pred) ** 2 for y, pred in zip(ys,ypred)),Value(0.0))

        # 3. ZERO_GRAD: wipe old gradients (your _backward uses +=)
        mlp.zero_grad()  # TODO

        # 4. BACKWARD: fill .grad on all 41 knobs
        loss.backward()

        # 5. UPDATE: turn every knob a little, against its gradient
        #    hint: loop over mlp.parameters(), and for each p:  p.data -= lr * p.grad
        for p in mlp.parameters():
            p.data -= lr * p.grad

        last_loss = loss.data
        print(f"step {step:3d}  loss {loss.data:.4f}")

    return last_loss


if __name__ == "__main__":
    final = train()
    print("final loss:", final, "✓" if final < 0.01 else "✗ not below 0.01 yet")
