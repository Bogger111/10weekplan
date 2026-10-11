import numpy as np
import matplotlib.pyplot as plt
import math
import random

from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"} 
    )
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }"
                  % (n.label, n.data, n.grad),
            shape="record"
        )
        if n._op:
            dot.node(
                name=uid + n._op,
                label=n._op
            )

            dot.edge(
                uid + n._op,
                uid
            )

    for n1, n2 in edges:

        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot

class Value:
    def __init__(self, data, _chilren = (), _op = '', label = ''):
        self.data = data
        self.grad = 0
        self._prev = set(_chilren)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value:data={self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):

        return self * (other ** -1)

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        self.grad = 1
        build_topo(self)

        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(3, [4, 4, 2, 1])

xs= [[1.0, 2.0, 4.5],
     [2.0, 1.0, 5.5],
     [1.0, 3.0, 10.0]]
ytrue = [1, 0, 1]
for k in range(200):
    ypred = [n(x) for x in xs]

    loss = sum((yg - yp) ** 2 for yg, yp in zip(ytrue, ypred))

    for p in n.parameters():
        p.grad = 0.0

    loss.backward()

    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)
print([n(x).data for x in xs])