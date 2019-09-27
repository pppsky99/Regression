# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:55:41 2018

@author: is2js
"""
import keras 
import numpy 

X = numpy.array([0, 1, 2, 3, 4, 5]) 
y= X * 2 + 1 

model = keras.models.Sequential()
model.add(keras.layers.Dense( 1, input_shape=(1,) ))
model.compile("SGD", "mse")
model.fit(X, y, epochs=1000, verbose=1)

predict = model.predict(X).flatten()

print('Target : ', y)
print('Predict : ', predict)