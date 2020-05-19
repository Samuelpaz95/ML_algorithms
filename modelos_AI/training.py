import numpy as np

### Error Cuadratico Medio
### Mean Square Error
def mse(_y, labels):
    # donde n es el numero batch_size, referenciada en la funcion "Train" ↓
    # la formula (1/n) * sum(d - y)²
    return 1/len(_y) * np.sum((labels - _y)**2)

### Derivada
def d_mse(features, _y, labels):
    # sum(-2x(-d + y))/n
    b_coef = np.ones((len(features), 1))
    features = np.concatenate([features, b_coef], axis=1)
    
    return np.sum(2 * features * (- labels + _y), axis=0) / len(features)


## Entrenamiento
def train_step(model, features, labels, lr):
    # y = w*x + b
    # 2x (xw - d + b) / n
    # 2x (y - d) / n
    _y = model(features)
    #print(d_mse(features, _y, labels))
    model.w = model.w - lr * d_mse(features, _y, labels)
    return mse(_y, labels)

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