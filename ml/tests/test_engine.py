import math
from ml.engine import Value

def numerical_grads(f, inputs, h=1e-6):
    """Computes numerical gradients for a list of Value inputs."""
    grads = []
    for x in inputs:
        orig = x.data
        x.data = orig + h
        fx1 = f().data
        x.data = orig - h
        fx2 = f().data
        x.data = orig  # reset
        grads.append((fx1 - fx2) / (2 * h))
    return grads

def numerical_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def test_tanh_grad():
    x = Value(0.7)
    x.tanh().backward()
    expected = numerical_grad(math.tanh, 0.7)
    assert abs(x.grad - expected) < 1e-6

def test_exp_grad():
    x = Value(0.7)
    x.exp().backward()
    expected = numerical_grad(math.exp, 0.7)
    assert abs(x.grad - expected) < 1e-6

def test_pow_grad():
    x = Value(0.7)
    z = x**3
    z.backward()

    f = lambda a: a ** 3
    expected = numerical_grad(f, 0.7)
    assert abs(x.grad - expected) < 1e-6

def test_relu_grad():
    x_val = 1.5
    x = Value(x_val)
    z = x.relu()
    z.backward()
    assert x.grad == 1.0

    # Test negative ReLU gradient
    x_val2 = -1.5
    x2 = Value(x_val2)
    z2 = x2.relu()
    z2.backward()
    assert x2.grad == 0.0

def test_division_grad():
    x_val,y_val = 6.0, 3.0
    x, y = Value(x_val), Value(y_val)
    z = x / y
    z.backward()

    f = lambda: x / y
    num_grads = numerical_grads(f, [x, y])
    assert abs(x.grad - num_grads[0]) < 1e-6
    assert abs(y.grad - num_grads[1]) < 1e-6

def test_messy_expression():
    a = Value(1.5)
    b = Value(-2.0)
    c = Value(3.0)
    d = Value(4.0)
    
    def calc():
        return ((a * b + c) ** 2 / d).tanh() - a.exp() + 3 - (c / 2).relu()
    
    out = calc()
    out.backward()
    
    inputs = [a, b, c, d]
    num_grads = numerical_grads(calc, inputs)
    
    for i, inp in enumerate(inputs):
        assert abs(inp.grad - num_grads[i]) < 1e-5

def test_plain_number_operations():

    a = Value(3.0)
    z1 = 1 - a
    z1.backward()
    assert a.grad == -1.0

    b = Value(3.0)
    z2 = 1 / b
    z2.backward()
    assert abs(b.grad - (-1.0 / (3.0 ** 2))) < 1e-6
    
    c = Value(2.0)
    x1 = c + 1
    x1.backward()
    assert c.grad == 1.0

    d = Value(2.0)
    x2 = 1 + d
    x2.backward()
    assert d.grad == 1.0

    e = Value(2.0)
    x3 = e * 2
    x3.backward()
    assert e.grad == 2.0

    f = Value(2.0)
    x4 = 2 * f
    x4.backward()
    assert f.grad == 2.0

def test_diamond_and_accumulation():
    # 1. Test a + a = 2.0 gradient
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0

    # 2. Test diamond graph = 27.0
    a = Value(2.0); b = Value(3.0); c = Value(4.0); d = Value(5.0)
    x = a * b
    y = x * c
    z = x * d
    L = y + z
    L.backward()
    assert a.grad == 27.0

def test_backward_handles_very_deep_graph():
    x = Value(1.0)
    y = x
    for _ in range(100_000):
        y = y + 1
    y.backward()
    assert x.grad == 1