import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data 
        self.grad = 0.0
        self._prev = tuple(dict.fromkeys(_children))
        self._op = _op
        self._backward = lambda:None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out 

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")

        # self -> a and other -> b => grad  
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self,other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        if self.data < 0 and isinstance(other,float) and not other.is_integer():
            raise ValueError(f"Negative base ({self.data}) with a fractional power ({other}) is not supported.")
            
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # Power rule: d/dx (x^n) = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        return other * (self ** -1.0)

    def exp(self):
        out = Value(math.exp(self.data),(self,),"exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

if __name__ == "__main__":
    a = Value(2.0); b = Value(3.0); c = Value(4.0); d = Value(5.0)
    x = a * b
    y = x * c
    z = x * d
    L = y + z
    L.backward()