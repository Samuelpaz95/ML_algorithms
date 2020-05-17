import random
import numpy as np

class SimpleRegresion:
    def __init__(self):
        self.learning_rate = 0.00006
        
        self.w = random.random()
        self.b = random.random()
    
    def predict(self, _input):
        # y = w*x + b
        return self.w * _input + self.b
    
    def train_step(self, feature, label):
        result = self.predict(feature)
        error = label - result
        # y = x*w + b
        
        # error cuadratico medio
        # E = (1/n)(d - y)²
        # E = (1/n)(d - (x*w + b))²
        
        # La deribada del error es:
        # x(x*w -d + b)
        # x(-d + xw + b)
        # x(-d + y)
        # - x(d - y) -> El gradiente
        # + x(d - y) -> El gradiente Negativo
        
        self.w = self.w + self.learning_rate * error * feature
        self.b = self.b + self.learning_rate * error
        
        return error
    
    def train(self, features, labels, epoch=1):
        for i in range(epoch):
            errores = []
            # recorre los datos uno a uno.
            for feature, label in zip(features, labels):
                error = self.train_step(feature, label)
                errores.append(error)
            print('Error del modelo:', np.mean(errores))
                
        