"""
Notas generales del codigo:
- Elimine un par de variables no usadas en las funciones creadas por Bonder
- No tiene la mejor estructura, hay MUCHO que optimizar, pero no funciona a fecha de 5/5/25 01:51
"""
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

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
     
    # Control de semilla
    if seed is not None:
        np.random.seed(seed)
        
    # Inicializamos listas de pesos y sesgos por capa l-esima
    W: list = []
    B: list = []
    
    for l in range(len(d)-1):
        alpha = 2*np.sqrt(3/d[l+1])
        w = alpha*(np.random.rand(d[l], d[l+1])-0.5)
        b = alpha*(np.random.rand(d[l+1],1)-0.5)
        W.append(w)
        B.append(b)
        
    return W, B

# Forward pass - ingresa la entrada x, pesos W, sesgos b y devuelve salida y, señales ss y salidas de las capas xx
def forward(x, W, b, 
            activacion: Callable[[list[float]], list[float]]):
    
    # Inicializamos listas 
    xx: list = [x]
    ss: list = []
    
    for i in range(len(W) - 1):
        s = W[i].T @ x - b[i]
        ss.append(s)
        x = activacion(s)
        xx.append(x)
        
    y = W[-1].T @ x - b[-1]
    xx.append(y)
    ss.append(y)
    
    return y, (xx, ss)

def graficar_red(W, b, activacion: Callable):
    x_vals = np.linspace(-5, 5, 100).reshape(1,100)
    y_vals, _ = forward(x_vals, W, b, activacion=activacion)
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals.flatten(), y_vals.flatten())
    plt.xlabel("x")
    plt.ylabel("y_pred")
    plt.title(f'Red neuronal con {len(d)-2} capas ocultas')
    plt.grid(True)
    plt.show()

def generar_datos(n: int):
    x = np.random.rand(1,n)
    # z = f(x)
    # la f puede ser CUALQUIER FUNCION CONTINUA que yo quiera aproximar
    z = np.sin(2 * np.pi * x)
    return x, z

# Error cuadratico medio
def C(y,z): return 0.5* np.sum(y-z)**2
def dC_dy(y,z): return y - z

# Backpropagation - ingresa entrada x, objetivo z, pesos W, sesgos b, cache = (xx, ss)
def backward(z, W, 
             cache, # xx = salidas de las capas y ss = señales heredados de forward
             deriv_activacion: Callable,
             dC_dy: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    
    # Inicializamos variables
    xx: list = cache[0]
    s: list = cache[1]
    L: int = len(W) # numero de capas

    # Inicializamos listas de matrices de derivadas de pesos y sesgos por capa 
    dC_dW: list = [None] * L
    dC_db: list = [None] * L
    
    # Delta de la ultima capa, al parecer son lineales, preguntar porque hace esto
    delta = dC_dy(xx[L],z) # aca Bonder no usa la d_act(s)
    dC_dW[-1] = xx[-2] @ delta.T
    dC_db[-1] = 0
    
    # Deltas de las capas ocultas
    for l in reversed(range(L - 1)): 
        delta = (W[l+1] @ delta) * deriv_activacion(s[l])
        dC_dW[l] = xx[l] @ delta.T
        dC_db[l] = -delta
        
    return dC_dW, dC_db

def descenso_grad(W, b, dC_dW, dC_db, eta: float):
    for l in range(len(W)):
        W[l] = W[l] - eta * dC_dW[l]
        b[l] = b[l] - eta * dC_db[l]
    return W, b

# Entrenamiento --  asumimos que los datos de entrenamiento x_train y z_train son de la forma (d, N).
# Es decir cada columna es un dato d-dimensional

def train(x_train, z_train, W, b, 
          activacion: Callable, deriv_activacion: Callable, 
          epocas: int, eta: float):
    
    #tomo el nro. de columnas. [0] es el número de filas.
    n = x_train.shape[1]
    costos = []
    
    # Itero para cada epoca 
    for epoca in range(epocas):
        costo_total = 0
        # Para cada capa (?) preguntar
        for i in range(n):
            # tomar xi del conjunto de entrenamiento
            x = x_train[:,i:i+1]
            z = z_train[:,i:i+1]
            # evaluar el forward en xi para obtener la predicción de la red
            y, cache = forward(x, W, b, activacion)
            # calculamos el costo de la predicción de la red, yi con el dato de entrenamiento zi
            costo = C(y,z)
            costo_total = costo_total + costo
            # usamos el backward con esta predicción para calcular el gradiente del costo
            dC_dW, dC_db = backward(z, W, cache, deriv_activacion, dC_dy)
            # realizamos el update de los parámetros de la red
            W, b = descenso_grad(W, b, dC_dW, dC_db, eta)
        promedio_costo = costo_total / n
        costos.append(promedio_costo)
        if epoca % 100 == 0:
            print(f"Epoca {epoca}, Costo: {promedio_costo:.4f}")
    return W, b, costos

# Gráfico del costo
# esta funcion me la robe impunemente, alta magia (no le se a matplotlib)
def graficar_costo(costos, fraccion=0.5):
    start = int(len(costos) * (1 - fraccion))
    plt.plot(range(start, len(costos)), costos[start:])
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title(f"Evolución del costo (último {int(fraccion * 100)}%)")
    plt.grid(True)
    plt.show()

# Hiperparametros y arquitectura
d=[1,100,100,100,1]
W, b = _init(d, seed=42) 

x_train, z_train = generar_datos(100)
W, b, costos = train(x_train, z_train, W, b, activacion=sigmoide, deriv_activacion=d_sigmoide, epocas=1000, eta=0.01)
#graficar_costo(costos, fraccion=0.4)

# Graficar la aproximación
x_test = np.linspace(0, 1, 200).reshape(1, 200)
y_test = np.sin(2 * np.pi * x_test)+ 0.5*np.cos(3*np.pi*x_test)
y_pred, _ = forward(x_test, W, b, activacion=sigmoide)

plt.plot(x_test.flatten(), y_test.flatten(), label="Target")
plt.plot(x_test.flatten(), y_pred.flatten(), label="Predicción")
plt.legend()
plt.title("Aproximación de la red")
plt.grid(True)
plt.show()

plt.scatter(x_train.flatten(), z_train.flatten(), label="Datos de entrenamiento", color='r', s=10)
plt.plot(x_test.flatten(), y_pred.flatten(), label="Predicción")
plt.legend()
plt.title("Aproximación de la red")
plt.grid(True)
plt.show()