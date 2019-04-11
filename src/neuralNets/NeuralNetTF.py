from src.neuralNets import  SingleLayerNN
from sklearn.metrics import classification_report, log_loss
from src.neuralNets import LSTM
from src.neuralNets import CNN
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, batch_size, epochs, X_train, X_test, y_train, y_test, is_multi_class, lstm_batch_size, lstm_epochs, cnn_batch_size, cnn_epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_mc = is_multi_class
        self.lstm_batch_size = lstm_batch_size
        self.cnn_batch_size = cnn_batch_size
        self.cnn_epochs = cnn_epochs
        self.lstm_epochs = lstm_epochs
        self.mp = True
        self.validation_split = 0.2
        self.verbose = 2 #0 = silent, 1 = progress bar, 2 = one line per epoch.


    def Execute(self):
        print(self.X_train.shape[1])
        single_neuron_nn = []

        for j in range(64,65):#(self.X_train.shape[1], self.X_train.shape[1]-1, -1):
            for i in range (80,81):
                nn = SingleLayerNN.SingleLayerNN(self.X_train.shape[1], i, j)
                nn.buildNN(self.X_train.shape[1], i, j)
                single_neuron_nn.append(nn)
                #print (nn.get_weights())
                nn.fit(x=self.X_train,
                         y=self.y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=self.verbose,
                         validation_split=self.validation_split)
                prediction = nn.predict(x=self.X_test)
                print("*****Run:",i,",",j,"  ")
                print(prediction.reshape(prediction.shape[0]))
                print(self.y_test)
                #print(classification_report(self.y_test, prediction))
                #nn.save_weights("weightlog.txt")
                #print (nn.get_weights())


    def ExecuteLSTM(self,  X_train, X_test, y_train, y_test):
        model = LSTM.GetLSTMModel(X_train)
        model.fit(x=X_train,
               y=y_train,
               batch_size=self.lstm_batch_size,
               epochs=self.lstm_epochs,
               verbose=self.verbose,
               validation_split=self.validation_split)
        #prediction = model.predict(x=X_test)
        #print ("Validation Score", log_loss(y_test, prediction))
        #print(prediction.reshape(prediction.shape[0]))
        #print(y_test)
        prediction = model.predict(x=X_train)
        print ("Training Score", log_loss(y_train, prediction))
        print(prediction.reshape(prediction.shape[0]))
        print(y_train)

    def ExecuteCNN(self,  X_train, X_test, y_train, y_test):
        model = CNN.GetCNNModel(X_train)
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

        self.y_prev = model.predict(X_train)

        def epoch_end_activity(epoch, logs):
            res = model.predict(X_train)
            plt.plot(0,0)
            plt.plot(1,0)
            for i in range (0, y_train.shape[0]):
                if y_train[i] == 0:
                    plt.plot([self.y_prev[i][0], res[i][0]], [i / 20, i / 20], color='green')
                    plt.plot(res[i][0], i / 20, markersize=0.5, marker='o', color='yellow', )
                else:
                    plt.plot([self.y_prev[i][0], res[i][0]], [i / 20, i / 20], color='red')
                    plt.plot(res[i][0],i/20, markersize=0.5, marker='o', color='blue',)
            plt.show()
            self.y_prev = res

        testmodelcallback = LambdaCallback(on_epoch_end=epoch_end_activity)

        model.fit(x=X_train,
               y=y_train,
               batch_size=self.cnn_batch_size,
               epochs=self.cnn_epochs,
               verbose=self.verbose,
               validation_split=self.validation_split,
               callbacks=[testmodelcallback])
        prediction = model.predict(x=X_train)
        print("Training Score", log_loss(y_train, prediction))
        print(prediction.reshape(prediction.shape[0]))
        print(y_train)
