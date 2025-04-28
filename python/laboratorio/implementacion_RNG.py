import numpy as np

# Funciones de activacion

def relu(x):
    return np.maximum(x, 0)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Red neuronal de una capa oculta

def init1(N: int):
    w = np.random.rand(N)
    b = np.random.rand(N)
    k = np.random.rand(N)
    return w, b, k

def forward1(x, w, b, k):
    s = w*x - b # capa entrada -> capa oculta
    x1 = relu(s) # aplico funcion de activacion
    y = k @ x1 # capa oculta -> capa salida
    return y

# Red neuronal de 2 capas ocultas

def init2(N1: int, N2: int):
    W1 = np.random.rand(1, N1)
    b1 = np.random.rand(N1, 1)
    W2 = np.random.rand(N1, N2)
    b2 = np.random.rand(N2, 1)
    k = np.random.rand(N2, 1)
    return W1, b1, W2, b2, k

def forward2(x, W1, b1, W2, b2, k) -> float:
    s1 = W1.T @ x - b1
    x1 = sigmoide(s1)
    s2 = W2.T @ x1 - b2
    x2 = sigmoide(s2)
    y = k.T @ x2
    return y

# Red neuronal general de 1 neurona de entrada, 6 capas ocultas de 4 neuronas, 1 neurona de salida
d = [1,4,4,4,4,4,4,1]

def init(d):
    W = []
    B = []
    for i in range(len(d)):
        w = np.random.rand(d[i], d[i+1])
        b = np.random.rand(d[i])
        w.appends
    return W, B

def forward(x, W, b, activacion) -> float:
    return NotImplemented

W, b = init(d)
print(W,b)