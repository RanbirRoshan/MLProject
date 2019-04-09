from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, SpatialDropout2D
import random


def GetCNNModel(X_train):
        random.seed(a=8)
        model = Sequential()
        print (X_train[0].shape)
        model.add(Conv2D(filters=5, strides=(1,1), input_shape=(X_train[0].shape[0],X_train[0].shape[1],1), kernel_size=(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2,2)))
        #model.add(SpatialDropout2D(0.1))
        model.add(Conv2D(filters=4, strides=(1,1), kernel_size=(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2,2)))
        #model.add(Dropout(0.2))
        #model.add(Conv2D(filters=10, strides=(1,1), kernel_size=(3)))
        #model.add(Activation('relu'))
        #model.add(MaxPool2D((2,2)))
        model.add(Flatten())
        model.add(Dense(30, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
