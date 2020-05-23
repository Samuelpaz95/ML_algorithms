import numpy as np

class Regresion:
    
    def __init__(self, num_inputs):
        # Iniciaizamos los pesos como un array random de "num_inputs + 1" elementos
        self.w = np.random.rand(num_inputs + 1, 1)
        
    def predict(self, _inputs):
        # forma de los inputs:
        # [[x1,x2,...xn],
        #  [x1,x2,...xn],
        #  [x1,x2,...xn],
        #      .....   
        #  [x1,x2,...xn]]
        
        b_coef = np.ones((len(_inputs), 1)) # crea una columna de 1's
        _inputs = np.concatenate([_inputs, b_coef], axis=1) # se concatena la columna de 1's a los inputs
        
        # forma de los inputs despues de concatenar una columna de unos:
        # [[x1,x2,...xn, 1],
        #  [x1,x2,...xn, 1],
        #  [x1,x2,...xn, 1],
        #      .....   
        #  [x1,x2,...xn, 1]]
        
        return np.matmul(_inputs, self.w)
        
    def __call__(self, _inputs):
        """
        Este metodo hace que una instancia de esta clase puede ser llamada como una función.
        """
        return self.predict(_inputs)