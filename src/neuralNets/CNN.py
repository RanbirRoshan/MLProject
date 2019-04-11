from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, SpatialDropout2D, LeakyReLU, initializers
import random


def GetCNNModel(X_train):
    random.seed(a=8)
    model = Sequential()
    model.add(Conv2D(filters=5, strides=(1,1), activity_regularizer=regularizers.l2(0.001), input_shape=(X_train[0].shape[0],X_train[0].shape[1],1), kernel_size=(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2,2)))
    #model.add(SpatialDropout2D(0.5))
    #model.add(Conv2D(filters=4, kernel_regularizer=regularizers.l2(0.01), strides=(1,1), kernel_size=(3,3)))
    #model.add(Activation('relu'))
    #model.add(MaxPool2D((2,2)))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=10, strides=(1,1), kernel_size=(3)))
    #model.add(Activation('relu'))
    #model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(2, activation='linear', kernel_initializer=initializers.RandomNormal(), activity_regularizer=regularizers.l2(0.001)))
    model.add(LeakyReLU(alpha=0.3))
    #model.add(Dropout(0.5))
    #model.add(Dense(5, activation='relu', activity_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activity_regularizer=regularizers.l2(0.01), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
