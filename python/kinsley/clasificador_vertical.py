from nnfs.datasets import vertical_data
from DNN import Activation_ReLU, Activation_Softmax, Layer_Dense, Loss_CategoricalCrossEntropy

# Creamos el dataset de 100 puntos y 3 grupos
X, y_true = vertical_data(samples=100, classes=3)

# Creamos modelo
dense1 = Layer_Dense(2, 3) # Capa oculta de 2 entradas hacia 3 neuronas
dense2 = Layer_Dense(3, 3) # Capa oculta de 3 entradas hacia 3 neuronas
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

# Creamos funcion de costo 
loss_function = Loss_CategoricalCrossEntropy()

# Propagacion hacia adelante hacia primer capa oculta y aplicamos ReLU
dense1.forward(X)
activation1.forward(dense1.output)

# Propagacion hacia adelante hacia segunda capa oculta y aplicamos Softmax
dense2.forward(activation1.output)
activation2.forward(dense2.output)
