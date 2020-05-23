import numpy as np

### Error Cuadratico Medio
### Mean Square Error
def mse(_y, labels):
    # donde n es el numero batch_size, referenciada en la funcion "Train" ↓
    # la formula (1/n) * sum(d - y)²
    return 1/len(_y) * np.sum((labels - _y)**2)

### Derivada
def d_mse(features, _y, labels):
    # sum(2x(-d + y))/n
    b_coef = np.ones((len(features), 1))
    features = np.concatenate([features, b_coef], axis=1)
    gradient = (np.sum(2 * features * (- labels + _y), axis=0) / len(features)).reshape(-1, 1)# de (num_inputs,) -> (num_inputs, 1)
    return gradient

## Entrenamiento
def train_step(model, features, labels, lr):
    # y = w*x + b
    # 2x (xw - d + b) / n
    # 2x (y - d) / n
    _y = model(features)
    #print(d_mse(features, _y, labels))
    gradient = d_mse(features, _y, labels)
    model.w = model.w - lr * gradient
    loss = mse(_y, labels)
    
    return loss

def train(model, features, labels, batch_size, epochs, lr):
    """
    Este metodo realiza los ciclos de entrenamiento determinados por el epochs y 
    la cantidad de datos del dataset
    
    Args:
    - model: un modelo de machine learning.
    - features: caracteristicas de entrada.
    - labels: etiquetas, el resultado que se espera obtener del modelo.
    - batch_size: El tamaño del mini-lote, la cantidad de ejemplos que se procesará simultaneamente.
    - epochs: la cantidad de veces que vamos a repetir el ciclo de entrenamiento.,
    """
    for _i in range(epochs):
        errors = []
        # en este for se tomaran los datos segun el batch_size.
        for _j in range(len(features) // batch_size):
            index_init = _j * batch_size
            index_end = index_init + batch_size
            
            feature = features[index_init : index_end]
            label = labels[index_init : index_end]
            
            error = train_step(model, feature, label, lr)
            errors.append(error)
            
        print('Error is:', np.mean(errors))
        
def train_nn_step(model, x, y, batch_size, lr):
    _y = model(x)
    d_error = 2*(_y - y) # derivada del error con respecto a _y
    d_sig = _y * (1 -_y) # resultado de la derivada de la sigmoide
    # una forma de expresar de que parte del error se pasara a la capa anterior
    culpa = d_error * d_sig 
    #print(d_error.shape, d_sig.shape)
    gradients = []
    for layer in reversed(model.layers):
        """El procedimiento que se hace aqui es basicamente para obtener los gradientes
        para optimizar la red nuronal, las librerias conocidas como tensorflow, pytorch y
        otros utilizan un algoritmo de autodiferenciación para calcular las derivadas parciales,
        necesarias para optener las gradientes.
        """
        # completamos el input de la capa con el coeficiente el bias = 1
        b_coef = np.ones((batch_size, 1))
        inputs = np.concatenate([layer.inputs, b_coef], axis=1)
        # muchas de las operaciones transpuestas son para compatibilizar las
        # operaciones matriciales
        gradient = np.transpose([culpa])
        gradients.append(np.transpose(np.sum(gradient * inputs, axis=1)) / batch_size)
        # aqui se agrega los siguientes factores de la derivada parcial den decenso del gradiente
        # y se redimensionan sin cambar los valores para operar corecamente el decenso de fradiente
        d_sig = inputs * (1 - inputs)
        culpa = gradient * np.transpose(layer.w).reshape(gradient.shape[0], 1, -1) * d_sig
        culpa = np.sum(culpa, axis=0)
        culpa = culpa[:,:-1]
        
    for layer, gradient in zip(reversed(model.layers), gradients):
        layer.w = layer.w - lr * gradient
        
    return mse(_y, y)

def train_nn(model, features, labels, batch_size, epochs, lr):
    """
    Este metodo realiza los ciclos de entrenamiento determinados por el epochs y 
    la cantidad de datos del dataset
    
    Args:
    - model: un modelo de machine learning.
    - features: caracteristicas de entrada.
    - labels: etiquetas, el resultado que se espera obtener del modelo.
    - batch_size: El tamaño del mini-lote, la cantidad de ejemplos que se procesará simultaneamente.
    - epochs: la cantidad de veces que vamos a repetir el ciclo de entrenamiento.,
    """
    for _i in range(epochs):
        errors = []
        # en este for se tomaran los datos segun el batch_size.
        for _j in range(len(features) // batch_size):
            index_init = _j * batch_size
            index_end = index_init + batch_size
            
            feature = features[index_init : index_end]
            label = labels[index_init : index_end]
            
            error = train_nn_step(model, feature, label, batch_size, lr)
            errors.append(error)
            
        print('Error is:', np.mean(errors))