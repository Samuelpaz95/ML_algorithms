from matplotlib import pyplot as plt

from dataset import ring_inputs, ring_labels

from training import train_nn

from modelos_AI import NNModel
from modelos_AI.layers import Dense
from modelos_AI.activations import sigmoid

"""Aque vamos a clasificar una nube de puntos de otra nube de puntos que la rodea
"""

inputs , labels = (ring_inputs, ring_labels)

porsentaje = int(len(inputs) * 0.8)
print(inputs.shape, porsentaje)

x_train = inputs[0 : porsentaje]
y_train = labels[0 : porsentaje].reshape(-1,1)
# y_train shape = (n, 1)
#[[1],
# [0],
# [0],
# [1],
# [1],
#  ...
# [0]]

x_test = inputs[porsentaje : ]
y_test = labels[porsentaje : ].reshape(-1,1)

# aqui creamos un modelo y le añadimos capas
model = NNModel()
model.add(Dense(3, activation=sigmoid))
model.add(Dense(3, activation=sigmoid))
model.add(Dense(1, activation=sigmoid))

# pruebe ejecutando este script con uan cantidad diferente de capas o de neuronas

results = model(x_test)

# pruebe entrenarlo con otros hiperparametres batch_size, epochs y lr
train_nn(model, x_train, y_train, batch_size=32, epochs=10, lr=2)

print("Antes de entrenar: \n", results)

plt.subplot(2,2,1)
plt.title("Las etiquetas.")
plt.scatter(x_test[y_test[:,0]  < 0.5, 0], x_test[y_test[:,0]  < 0.5, 1], c="skyblue")
plt.scatter(x_test[y_test[:,0] >= 0.5, 0], x_test[y_test[:,0] >= 0.5, 1], c="red")
plt.axis('equal')
plt.subplot(2,2,2)
plt.title("Las Predicciones")
plt.scatter(x_test[results[:,0]  < 0.5, 0], x_test[results[:,0]  < 0.5, 1], c="skyblue")
plt.scatter(x_test[results[:,0] >= 0.5, 0], x_test[results[:,0] >= 0.5, 1], c="red")
plt.axis('equal')

results = model(x_test)
print("Despues de entrenar: \n", results)

plt.subplot(2,2,3)
plt.scatter(x_test[y_test[:,0]  < 0.5, 0], x_test[y_test[:,0]  < 0.5, 1], c="skyblue")
plt.scatter(x_test[y_test[:,0] >= 0.5, 0], x_test[y_test[:,0] >= 0.5, 1], c="red")
plt.axis('equal')
plt.subplot(2,2,4)
plt.scatter(x_test[results[:,0]  < 0.5, 0], x_test[results[:,0]  < 0.5, 1], c="skyblue")
plt.scatter(x_test[results[:,0] >= 0.5, 0], x_test[results[:,0] >= 0.5, 1], c="red")
plt.axis('equal')

plt.show()




