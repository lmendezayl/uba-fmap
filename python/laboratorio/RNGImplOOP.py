import numpy as np
import matplotlib.pyplot as plt
from _collections_abc import Callable

class NeuralNetwork:
    def __init__(self, layer_dims, activation, activation_deriv, cost_fn, cost_deriv, seed=None):
        self.layer_dims = layer_dims
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.cost_fn = cost_fn 
        self.cost_deriv = cost_deriv
        self.seed = seed
        self.W, self.b = self._init_params()
        
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
    
    def forward(self, x, W, b, activation):
    
        # Inicializamos listas 
        a = x
        xx: list = [x]
        ss: list = []
        
        for i in range(len(W) - 1):
            s = W[i].T @ a - b[i]
            ss.append(s)
            a = activation(s)
            xx.append(a)
            
        y = W[-1].T @ a - b[-1]
        xx.append(y)
        ss.append(y)
        
        return y, (xx, ss)        
    







