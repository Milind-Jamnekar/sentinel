import random
from typing import Any, Sequence
from ml.engine import Value

# if Neuron(3) given that's means give me 
# Neuron of 3 inputs 
class Neuron:
    def __init__(self,n_in) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x: Sequence[Value | float]) -> Value:
        if len(self.w) != len(x):
            raise ValueError("Need same number of x value as weight define")
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

# if Layer(3, 4) given that's means give me
# 4 neurons of 3 inputs
class Layer:
    def __init__(self, n_in:int, n_out:int) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x:Sequence[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

# If (3, [4,4,1]) given that's means give me 
# 3 layers with 1st layer of 4, 2nd of 4 and 3rd of 1
class MLP:
    def __init__(self, n_in:int, layer_sizes:list[int]) -> None:
        sizes = [n_in] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x:Sequence[Value | float]) -> Value:
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        if len(out) != 1:
            raise ValueError(f"MLP must end with 1 neuron to return one value, got {len(out)}")
        return out[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0