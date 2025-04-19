"""
--------------------------------------------------------------------------------------
Lautaro Evaristo Mendez | 2025

Implementacion de red neuronal de clasificacion del libro Neural Networks from Scratch, 
Kinsley & Kukiela.

El objetivo es implementar una red neuronal que pueda clasificar correctamente a un
punto dado un dataset dividido en clusteres. En este caso, el dataset es 
nnfs.datasets.spiral_data, un dataset que simula una galaxia espiral, dividida por
brazos (clusteres). 

En este archivo uso el archivo DNN.py donde se encuentran las implementaciones de las 
clases usadas en esta celda.

Para una justificacion matematica del uso de una capa adicional que implementa la 
funcion de activacion softmax en problemas de clasificacion, ver Deep Learning 
Architecture, O. Calin, ch.18.8.1, p. 578. 
--------------------------------------------------------------------------------------
"""
import numpy as np
from nnfs.datasets import spiral_data
from DNN import Activation_ReLU, Activation_Softmax, Layer_Dense, Loss_CategoricalCrossEntropy

# Creamos el dataset de 100 puntos y 3 clusteres (galaxia de tres brazos)
# X es la matriz que contiene las coordenadas de los puntos
# y es el vector que contiene las etiquetas categoricas de cada punto mediante 
# una correspondencia uno a uno: punto X[i] pertenece a cluster y[i]
X, y_true = spiral_data(samples=100, classes=3)

# Creamos las capas de la red neuronal
# Capa oculta 1: 2 entradas hacia 3 neuronas
dense1 = Layer_Dense(2, 3)

# Capa oculta 2: 3 entradas hacia 3 neuronas
dense2 = Layer_Dense(3, 3)

# Funciones de activacion
# Usamos ReLU para la primer capa oculta
activation1 = Activation_ReLU()

# Usamos Softmax para la segunda capa oculta
activation2 = Activation_Softmax()

# Creamos funcion de costo usando entropia cruzada
loss_function = Loss_CategoricalCrossEntropy()

# Propagacion hacia adelante hacia primer capa oculta y aplicamos ReLU
dense1.forward(X)
activation1.forward(dense1.output)

# Propagacion hacia adelante hacia segunda capa oculta y aplicamos Softmax
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Le hacemos un head() al vector de salida
print(activation2.output[:5])

# Calculo de funcion de costo entre valores de salida de la segunda capa oculta y valor objetivo.
loss = loss_function.calculate(activation2.output, y_true)
print("Costo: ", loss)

# Calculamos la precision entre salida de activation2 y el vector objetivo y_true
y_pred = np.argmax(activation2.output, axis=1)
# Si y_true son one-hot vectors, aplicamos transformacion de matriz a vector 
# tomando argmax a cada one-hot
if len(y_true.shape) == 2:
    y_true = np.argmax(y_true, axis=1)
accuracy = np.mean(y_pred == y_true)
print("acc: ", accuracy)

