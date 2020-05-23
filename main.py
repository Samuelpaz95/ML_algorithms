import matplotlib.pyplot as plt

from dataset import *
from modelos_AI.regresion_simple import SimpleRegresion

model = SimpleRegresion()


results = []
for power in powers:
    result = model.predict(power)
    results.append(result)

def graph():
    plt.scatter(powers, labels, label="Datos del dataset")
    plt.plot(powers, results, color='r', label="Resultado de la prediccion del modelo")
    plt.ylabel('Price')
    plt.xlabel('Power')
    plt.grid(True)
    plt.legend()
    plt.show()

# Grafica antes del entrenamiento.
graph()

model.train(powers, labels, epoch=50)

results = []
for power in powers:
    result = model.predict(power)
    results.append(result)

# Grafica despues del entrenamiento.
graph()

