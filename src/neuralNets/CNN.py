from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, SpatialDropout2D, LeakyReLU, initializers
import random


def GetCNNModel(X_train):
    random.seed(a=8)
    model = Sequential()
    model.add(Conv2D(filters=2, strides=(1,1), input_shape=(X_train[0].shape[0],X_train[0].shape[1],1), kernel_size=(5,100)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1,2)))
    model.add(Conv2D(filters=3, strides=(1,1), kernel_size=(5,100)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((1,2)))
    #model.add(SpatialDropout2D(0.5))
    model.add(Conv2D(filters=3, strides=(1,1), kernel_size=(4,8)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(filters=2, strides=(1,1), kernel_size=(12,4)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(12, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
