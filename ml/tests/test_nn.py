import math
import pytest
from ml.engine import Value
from ml.nn import Neuron, Layer, MLP


def test_neuron_matches_hand_calculation():
    n = Neuron(3)
    n.w = [Value(2.0), Value(1.0), Value(0.1)]
    n.b = Value(-1.5)
    out = n([1.5, 0.8, 0.3])
    print(out.data)
    assert abs(out.data - math.tanh(2.33)) < 1e-9      # ≈ 0.98, the video example


def test_neuron_rejects_wrong_input_size():
    with pytest.raises(ValueError):
        Neuron(3)([1.0, 2.0])


def test_layer_returns_list_or_single_value():
    assert len(Layer(3, 4)([1.0, 2.0, 3.0])) == 4
    assert len(Layer(3, 1)([1.0, 2.0, 3.0])) == 1


def test_mlp_returns_single_value_in_tanh_range():
    out = MLP(3, [4, 4, 1])([2.0, 3.0, -1.0])
    assert isinstance(out, Value)
    assert -1 < out.data < 1


def test_gradients_reach_the_weights():
    mlp = MLP(3, [4, 4, 1])
    mlp([2.0, 3.0, -1.0]).backward()
    assert any(p.grad != 0.0 for p in mlp.parameters())


def test_zero_grad_resets_everything():
    mlp = MLP(3, [4, 4, 1])
    mlp([2.0, 3.0, -1.0]).backward()
    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in mlp.parameters())


def test_parameter_count():
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.parameters()) == 41  # 👉 count it by hand and put your number here