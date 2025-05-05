import numpy as np
import matplotlib.pyplot as plt
from _collections_abc import Callable

class NeuralNetwork:
    def __init__(self, layer_dims, activation, activation_deriv, cost_fn: Callable, cost_deriv, obj_fn, sample_size, seed=None):
        self.layer_dims = layer_dims
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.cost_fn = cost_fn 
        self.cost_deriv = cost_deriv  
        self.obj_fn = obj_fn
        self.seed = seed
        self.W, self.b = self._init_params()
        self.x_train = np.random.rand(1,sample_size)
        # z = f(x)
        # la f puede ser CUALQUIER FUNCION CONTINUA que yo quiera aproximar
        self.z_train = self.obj_fn(self.x_train)
        
    def _init_params(self):
        # Control de semilla
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Inicializamos listas de pesos y sesgos por capa l-esima
        W: list = []
        B: list = []
        
        for l in range(len(self.layer_dims)-1):
            alpha = 2*np.sqrt(3/self.layer_dims[l+1])
            w = alpha*(np.random.rand(self.layer_dims[l], self.layer_dims[l+1])-0.5)
            b = alpha*(np.random.rand(self.layer_dims[l+1],1)-0.5)
            W.append(w)
            B.append(b)
            
        return W, B
    
    def forward(self):
    
        # Inicializamos listas 
        a = x
        xx: list = [x]
        ss: list = []
        
        for i in range(len(W) - 1):
            s = W[i].T @ a - b[i]
            ss.append(s)
            a = self.activation(s)
            xx.append(a)
            
        y = W[-1].T @ a - b[-1]
        xx.append(y)
        ss.append(y)
        
        return y, (xx, ss)        
    
    # Backpropagation - ingresa entrada x, objetivo z, pesos W, sesgos b, cache = (xx, ss)
    def backward(self, z, W, cache, deriv_activacion: Callable, dC_dy: Callable):
        
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

    def descenso_grad(self, W, b, dC_dW, dC_db, eta: float):
        for l in range(len(W)):
            W[l] = W[l] - eta * dC_dW[l]
            b[l] = b[l] - eta * dC_db[l]
        return W, b

    # Entrenamiento --  asumimos que los datos de entrenamiento x_train y z_train son de la forma (d, N).
    # Es decir cada columna es un dato d-dimensional

    def train(self, x_train, z_train, W, b, 
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
                y, cache = self.forward(x, W, b, activacion)
                # calculamos el costo de la predicción de la red, yi con el dato de entrenamiento zi
                costo = self.cost_fn(y,z)
                costo_total = costo_total + costo
                # usamos el backward con esta predicción para calcular el gradiente del costo
                dC_dW, dC_db = self.backward(z, W, cache, deriv_activacion, self.cost_deriv)
                # realizamos el update de los parámetros de la red
                W, b = descenso_grad(W, b, dC_dW, dC_db, eta)
            promedio_costo = costo_total / n
            costos.append(promedio_costo)
            if epoca % 100 == 0:
                print(f"Epoca {epoca}, Costo: {promedio_costo:.4f}")
        return W, b, costos






