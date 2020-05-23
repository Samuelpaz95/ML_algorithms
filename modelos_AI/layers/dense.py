import numpy as np
from modelos_AI import Regresion

class Dense(Regresion):
    def __init__(self, num_neurons, activation = lambda y:y):
        """Esta es una capa densa, es decir una capa de neuronas que estan totalmente
        conectadas con las de la capa anterior.
        
        Args:
        - num_neurons: la cantidad de neuronas que contiene la capa
        - activation: la funcion que decidira si la capa se activa o retornando valores de 0 a 1 ó -1 a 1
        """
        self.__neurons = num_neurons
        self.__activation = activation
        # registramos la entrada y salida mas reciente de la capa para usarlos en el entrenamiento
        self.__inputs = [] 
        self.__outputs = []
    
    def __call__(self, _inputs):
        _inputs = np.array(_inputs)
        if not len(self.__outputs) > 0 :
            # inicializamos la capa en una matriz de pesos de forma (input_shape, neurons)
            self.w = np.random.rand(_inputs.shape[-1] + 1, self.__neurons)
            # quedaría asi, cada columna es una neurona
            # [w1,w1,w1,....,w1]
            # [w2,w2,w2,....,w2]
            # [w3,w3,w3,....,w3]
            # [w4,w4,w4,....,w4]
            #    --------
            # [wn,wn,wn,....,wn]
            
        self.__inputs = _inputs # guarnamos el input
        # Hacemos la predccion, lo hace el padre.
        self.__outputs = super().__call__(_inputs)
        # luego la pasamps por una funcion de activacion
        self.__outputs = self.__activation(self.__outputs)
        return self.__outputs
    
    @property
    def outputs(self):
        return self.__outputs
    
    @property
    def inputs(self):
        return self.__inputs