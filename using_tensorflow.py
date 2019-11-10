import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data = pd.read_csv('./salary-data/Salary_Data.csv')
X = data['YearsExperience']
Y = data['Salary']

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(X,Y,epochs=100)
print(model.predict([5]))
