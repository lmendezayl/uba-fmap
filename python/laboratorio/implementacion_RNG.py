import numpy as np
import matplotlib.pyplot as plt

# Funciones de activacion y sus derivadas
def relu(x):
    return np.maximum(x, 0)
def d_relu(x):
    return (x > 0).astype(float)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x)**2


# Inicializacion de pesos
def _init(d: list, seed=None):
    if seed is not None:
        np.random.seed(seed)
    W: list = []
    B: list = []
    for l in range(len(d)-1):
        alpha = 2*np.sqrt(3/d[l+1])
        w = alpha*(np.random.rand(d[l], d[l+1])-0.5)
        b = alpha*(np.random.rand(d[l+1],1)-0.5)
        W.append(w)
        B.append(b)
    return W, B


# Forward
def forward(x, W, B, activacion):
    X: list = [x]
    ss: list = []
    for i in range(len(W) - 1):
        s = W[i].T @ x - B[i]
        ss.append(s)
        x = activacion(s)
        X.append(x)
    y = W[-1].T @ x - B[-1]
    X.append(y)
    ss.append(y)
    return y, (X, ss)


def graficar_red(W, b, activacion):
    x_vals = np.linspace(-5, 5, 100).reshape(1,100)
    y_vals, _ = forward(x_vals, W, b, relu)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals.flatten(), y_vals.flatten())
    plt.xlabel("x")
    plt.ylabel("y_pred")
    plt.title(f'Red neuronal con {len(d)-2} capas ocultas')
    plt.grid(True)
    plt.show()


def generar_datos(n):
    x = np.random.rand(1,n)
    # z = f(x)
    # la f puede ser CUALQUIER FUNCION CONTINUA que yo quiera aproximar
    z = np.sin(2 * np.pi * x) + 0.1 * np.random.rand(1,n)
    return x,z

# ECM
def C(y,z):
    return 0.5* np.sum(y-z)**2
def dC_dy(y,z):
    return y-z

# Retropropagacion
# Todo
def backward(x, z, W, b, cache, deriv_activacion, dC_dy):
    # dC_dW = [dC_dW(0), ... , dC_dW(L-1)]
    #         d(0) x d(1)     d(L-1) x d(L)
    X, s = cache
    dC_dW = []
    dC_db = []
    # Ultimo
    L = len(W)
    delta_l = dC_dy(X[L],z) * deriv_activacion(s[L])
    for l in range(len(W)-1,-1,-1):
        delta_l = (W[l] @ delta_l) * deriv_activacion(s[l-1])
        dC_dW.append(X[l-1]@ delta_l.T)
        dC_db.append(-delta_l)
    return dC_dW, dC_db

# Todo
def train(x_train, z_train, W, b, activacion, d_activacion, epocas, learning_rate):
    n = x_train.shape[0]
    costos = []
    for epoca in range(epocas):
        costo_total = 0
        for i in range(n):
            x = x_train[:,i:i+1]
            z = z_train[:,i:i+1]
            

    return NotImplementedError 

d=[1,100,50,1]
W, b = _init(d, seed=42) 
x, z = generar_datos(100)

y, cache = forward(x, W, b, relu)
#graficar_red(W, b, relu)
print(backward(x, z, W, b, cache, d_relu, dC_dy))