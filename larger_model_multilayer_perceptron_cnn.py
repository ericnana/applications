#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 22:51:45 2018

@author: enwc
"""

#Large CNN for the MNIST Dataset
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalize inputs from 0-255 to 0 -1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#define the larger model
def larger_model():
    #create model
    model = Sequential()
    model.add(Conv2D(32,(5,5), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = 0.01), metrics=['accuracy'])
    return model

#build the model
model = larger_model()

#print summary
print(model.summary())
    
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), validation_split = 0.1, epochs=10, batch_size=400, verbose=1, shuffle=1)

# Different Plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
     
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


#Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Large CNN Error: %.2f%%" %(100-scores[1]*100))


# serialize model to JSON
model_json = model.to_json()

#Save model in a json file file
with open('larger_model_multilayer_perceptron_cnn.json', 'w') as json_file:
    json_file.write(model_json)
    
#Save model in a h5 file
model.save_weights('larger_model_multilayer_perceptron_cnn.h5')
#model.load('larger_model_multilayer_perceptron_cnn.h5')

'''
# load json and create model
json_file = open("model.json" ,"r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
'''
