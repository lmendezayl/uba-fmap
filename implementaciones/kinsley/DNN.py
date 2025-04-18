import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Esto es importante, sobrescribe funciones de NumPy
# Define una semilla fija de randomizacion
nnfs.init()

# Creamos una clase Layer_Dense para trabajar con datasets
class Layer_Dense:
    
    def __init__(self, n_inputs: int, n_neurons: int):
        # Inicializo pesos aleatorios desde distribucion normal con media 0 y varianza 1.
        # Mulplicamos por 0.01 para generar pesos de magnitud menor.
        # Evita que el modelo ajuste los datos al inicio ya estos serian demasiado grandes 
        # comparados a las actualizaciones hechas durante el entrenamiento
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Iniciamos los sesgos con 0 para evitar activacion neuronal al inicio
        self.biases = np.zeros((1, n_neurons), dtype=int)
    
    # Propagacion hacia delante
    def forward(self, inputs):
        # Calculo valores de salida desde entradas, pesos y sesgos
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    
    # Feedforward
    def forward(self, inputs):
        # Calculo los valores de salida desde los valores de entrada
        self.output = np.maximum(0, inputs)
      
class Activation_Softmax:
    
    # Feedorward
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        

# Creamos dataset 
X, y = spiral_data(samples=100, classes=3)

# Creamos capa densa con 2 features de entrada y 3 valores de salida
dense1 = Layer_Dense(2, 3)

# Creamos funcion de activacion ReLU a ser usada con capa densa
activation1 = Activation_ReLU()

# Forwardprop
dense1.forward(X)

# Feedforward a traves de la funcion de activacion
# Toma las salidas de la capa previa
activation1.forward(dense1.output)

print("Primeras 5 salidas:\n", activation1.output[:5])

# Primeras 5 salidas:
# [[0.         0.         0.        ]
# [0.         0.00011395 0.        ]
# [0.         0.00031729 0.        ]
# [0.         0.00052666 0.        ]
# [0.         0.00071401 0.        ]]
