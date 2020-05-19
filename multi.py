from matplotlib import pyplot as plt
from modelos_AI import train, features, labels, Regresion, shuffle, mse, np

def graph(predictions, features, labels):
    plt.subplot(2,2,1)
    plt.title('For Year')
    plt.scatter(features[:,0], labels.reshape(-1), label="Datos del Dataset")
    plt.scatter(features[:,0], predictions, c='r', label="Resultados del modelo de ML")
    plt.grid(True)
    plt.legend()
    plt.ylabel('Price')

    plt.subplot(2,2,2)
    plt.title('For Mileage')
    plt.scatter(features[:,1], labels.reshape(-1), label="Datos del Dataset")
    plt.scatter(features[:,1], predictions, c='r', label="Resultados del modelo de ML")
    plt.grid(True)
    plt.legend()
    plt.ylabel('Price')

    plt.subplot(2,2,3)
    plt.title('For Engine')
    plt.scatter(features[:,2], labels.reshape(-1), label="Datos del Dataset")
    plt.scatter(features[:,2], predictions, c='r', label="Resultados del modelo de ML")
    plt.grid(True)
    plt.legend()
    plt.ylabel('Price')

    plt.subplot(2,2,4)
    plt.title('For Power')
    plt.scatter(features[:,3], labels.reshape(-1), label="Datos del Dataset")
    plt.scatter(features[:,3], predictions, c='r', label="Resultados del modelo de ML")
    plt.grid(True)
    plt.legend()
    plt.ylabel('Price')

    plt.show()
    
labels = labels.reshape(-1,1) # cambia la forma de la matriz

## mesclando dataset
features, labels = shuffle(features, labels)

# Repartiendo el dataset
n = int(round(len(features) * 0.8))
# Datos para entrenar
train_inputs = features[ :n ]
train_labels = labels[ :n ]

# Datos para testear
test_inputs = features[ n: ]
test_labels = labels[ n: ]

# creamos el modelo de regresion con un tamaño de 4 entradas
model = Regresion( features.shape[-1] ) # el input es cuatro


# hacemos una prediccion con los datos que el modelo
# NO vió duarante el entrenamiento
predictions = model(test_inputs)


graph(predictions, test_inputs, test_labels)

# entrenamos con los datos de entrenamiento
train(model, train_inputs, train_labels, batch_size=1000, epochs=1000, lr=0.0000001)

# hacemos una prediccion con los datos que el modelo
# NO vió duarante el entrenamiento
predictions = model(test_inputs)

graph(predictions, test_inputs, test_labels)

print("Error en los datos de testing", mse(predictions, test_labels))

np.save("weights", model.w)
print("Model saved.")
