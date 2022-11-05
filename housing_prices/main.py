# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

df= pd.read_csv (r'/Users/aishwaryaiyer/Desktop/housing.csv')
print (df)

from sklearn.model_selection import train_test_split


area = df[['area']].values
price = df[['price']].values

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=area.shape[1:]))
model.compile(keras.optimizers.Adam(learning_rate=1), 'mean_squared_error')
model.fit(area, price, epochs=1000)


df.plot(kind= 'scatter', x='area', y='price', title= 'Housing Prices Vs Area in UK')
y_predict = model.predict(area)
plt.plot(area, y_predict, color='red')
plt.show()
