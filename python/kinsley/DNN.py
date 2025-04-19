import numpy as np
import nnfs

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
        # Guardamos inputs para retropropagacion
        self.inputs = inputs
        
    def backward(self, dvalues):
        # Gradientes de los parametros
        self.dweigths = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradiente de los valores
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Activation_ReLU:
    # Feedforward
    def forward(self, inputs):
        # Calculo los valores de salida desde los valores de entrada
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Gradiente 0 donde valor es negativo
        self.dinputs[self.inputs <= 0] = 0 #type: ignore 
      
class Activation_Softmax:
    # Feedorward
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    # Retropropagacion
    def backward(self, dvalues):
        
        # Creamos array vacio
        self.dinputs = np.empty_like(dvalues)
    
        # Enumeramos salidas y gradientes
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): # algun dia entendere que hace enum..
            # Aplastamos array de salida
            single_output = single_output.reshape(-1, 1)
            # Calculamos el jacobiano de las salidas
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculamos gradiente por samples, y lo agregamos al array de gradientes por sample
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
        
class Loss:
    # Calcula los datos y costo regularizado 
    # dada la salida del modelo y valores objetivo
    def calculate(self, output, y):
        # Calcula costo 
        sample_losses = self.forward(output, y) # type: ignore
        data_loss = np.mean(sample_losses)
        return data_loss 
    
class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        # Numero de samples en un bache
        samples = len(y_pred)
        # Acotamos datos para prevenir division por 0
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)
        # Probabilidades valor objetivo:
        # Si trabajamos con etiquetas categoricas
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clip[
                range(samples),
                y_true
            ]
        # Si trabajamos con vectores one-hot (matriz)
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clip * y_true,
                axis=1
            )
        
        verisimilitud_log_negativa = -np.log(correct_confidences) #type: ignore
        return verisimilitud_log_negativa
    
    def backwards(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])
        
        # Si los labels son ralos, los convertimos en one-hot 
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculo gradiente
        self.dinputs = -y_true / dvalues
        # Normalizamos
        self.dinputs = self.dinputs / samples
    
            
        