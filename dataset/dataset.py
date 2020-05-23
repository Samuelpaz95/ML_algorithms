import numpy as np
import pandas as pd

"""
En este modulo se extraen datos de un archivo train-data.csv
y purgamos datos no validos o vacios que pueda contener el dataset
"""


dftrain = pd.read_csv('train-data.csv')

# elegimos solamente estas columnas
colums = ['Year', 'Mileage', 'Engine', 'Power', 'Price']
features = dftrain[colums]

#print(features)

# localizamos donde podria haber filas con valores nulos para quitarlos del dataset
# de entrenamiento.
features = features.loc[features['Power'] != 'null bhp']

# qutamos las strings que denptan la unidad de medida para tener solo el numero
# para entrenar al modelo
# 123.4 bhp --> 123.4
features['Power'] = features['Power'].map(lambda data: str(data)[:-3])
# quetamos si hay strings vacios
features = features.loc[features['Power'] != '']

# Hacemos lo mismo aquí
# 12412 cc --> 12412
features['Engine'] = features['Engine'].map(lambda data: str(data)[:-2])

# Y aquí
# 26.6 km/kg ---> 26.6 
# 19.67 kmpl ---> 19.67
features['Mileage'] = features['Mileage'].map(lambda data: str(data)[:-5])

# qutamos string vacios
features = features.loc[features['Mileage'] != '']

# Estraemos la columnas "Price" y lo comvertimos en un array de Numpy
labels = np.array(features['Price']).astype(np.float64)

# Quitamos la ultima columna del dataset
features = features[colums[:-1]]

# y comvertimos el dataset en una matriz de numpy
features = np.array(features).astype(np.float64)

# esta es la columna "Powers usada por el modelo de regresion simple"
powers = features[:,3]

# Imprimimos los primeros 5 elementos para visualizar
#print(features[:5])


def shuffle(inputs, targets):
    """
    Este metodo se encarga de mezclar el oreden de las filas de dos matrizer simultaneamente.
    
    Este metodo se utiliza para mesclar los datos de entrenamiento.
    
    Ambas matrices deben de tener el mismo numero de filas.
    
    Args:
    
    - inputs: `Numpy.Array` una matris cualquiera.
    - targets: `Numpy.Array` una matris cualquiera.
    
    Returns:

    - `inputs` La misma matriz con el orden de las filas mesclado.
    - `targets` La misma matriz con el orden de las filas mesclado.
    """
    assert len(inputs) == len(targets)
    permutation = np.random.permutation(len(inputs))
    return inputs[permutation], targets[permutation]


