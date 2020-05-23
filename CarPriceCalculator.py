import numpy as np

print("Example")
from dataset import features
from modelos_AI import Regresion


model = Regresion(4)


model.w = np.load('weights.npy')


colums = ['Year: ', 'Mileage: ', 'Engine: ', 'Power: ', 'Price: ']

year = float(input(colums[0]))
mileage = float(input(colums[1]))
engine = float(input(colums[2]))
power = float(input(colums[3]))

input_data = np.array([[year, mileage, engine, power]])

print(f'\n---> Your car is priced at $ {model(input_data)[0][0] * 1000} \n')