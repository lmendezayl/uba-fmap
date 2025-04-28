import numpy as np

def relu(x):
    return np.maximum(x, 0)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def _init(d: list):
    x = np.random.rand(d[0], 1)
    W: list[np.ndarray] = []
    B: list[np.ndarray] = []
    i: int = 0
    while i < len(d)-1:
        w = np.random.rand(d[i], d[i+1])
        b = np.random.rand(d[i+1])
        W.append(w)
        B.append(b)
        i = i + 1
    return x, W, B

def forward(x, W, B, activacion):
    X: list = [x]
    i: int = 0
    while i < len(W):
        s = W[i].T @ x - B[i]
        x = activacion(s)
        X.append(x)
        i = i + 1
    return X

# d0 = 1
# d1 = 3
# dL = d3 = 1
d = [1,3,1] 
x, W, b = _init(d) 
X = forward(x,W,b,sigmoide)
print(X)
