import random
import pytest
import torch
from ml.engine import Value
from ml.nn import MLP
from ml.train import xs, ys

def test_mlp_gradients_match_pytorch():
    random.seed(3)
    mlp = MLP(3,[4,4,1])

    # My engine
    loss = sum(((mlp(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0))
    mlp.zero_grad()
    loss.backward()

    # the same network in PyTorch, copying MY weights ---
    Ws, bs = [], []
    for layer in mlp.layers:
        # Adding wights from neurons 
        W = torch.tensor(
            [[w.data for w in n.w] for n in layer.neurons],
            dtype=torch.float64,
            requires_grad=True
        )

        # adding data of neuron 
        b = torch.tensor(
            [n.b.data for n in layer.neurons], 
            dtype=torch.float64, 
            requires_grad=True
        )
        Ws.append(W)
        bs.append(b)

    # inputs and expected outputs to tune  
    X = torch.tensor(xs, dtype=torch.float64)
    Y = torch.tensor(ys, dtype=torch.float64)

    h = X
    for W, b in zip(Ws,bs):
        h = torch.tanh(h @ W.T + b)
    t_loss = ((h.squeeze(1) - Y) ** 2).sum()
    t_loss.backward()

    assert t_loss.item() == pytest.approx(loss.data)

    for k, layer in enumerate(mlp.layers):
        for i, neuron in enumerate(layer.neurons):
            assert bs[k].grad[i].item() == pytest.approx(neuron.b.grad)
            for j, w in enumerate(neuron.w):
                assert Ws[k].grad[i][j].item() == pytest.approx(w.grad)

