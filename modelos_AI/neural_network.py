class NNModel:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
        
    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs