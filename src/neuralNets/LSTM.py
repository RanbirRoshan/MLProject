from keras import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM


def GetLSTMModel(X_train):
    model = Sequential()
    size = X_train[0].shape[0]
    model.add(LSTM(units=X_train[0].shape[0],input_shape=X_train[0].shape))
    model.add(Dense(units=X_train[0].shape[0],activation="relu"))
    model.add(Dense(units=size>>1,activation="relu"))
    model.add(Dense(units=1,activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model