from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1 layer with 1 neuron and just 1 input
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# stochastic gradient descent as optimizer and mean_squared_error as loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

#Training data
x=np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
# Relationship between x and y:  y = 2x - 1
y=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype=float)
model.fit(x,y,epochs=500)

#Test input
x1=np.array([10.0], dtype=float)
# Expected output according to the equation: y = 19
print(model.predict([x1]))
