from src.neuralNets import  SingleLayerNN
from sklearn.metrics import confusion_matrix


class NeuralNet:

    def __init__(self, batch_size, epochs, X_train, X_test, y_train, y_test, is_multi_class):
        self.batch_size = batch_size
        self.epochs = epochs
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_mc = is_multi_class
        self.mp = True
        self.validation_split = 0.1
        self.verbose = 2 #0 = silent, 1 = progress bar, 2 = one line per epoch.


    def Execute(self):
        print(self.X_train.shape[1])
        single_neuron_nn = []

        for j in range(self.X_train.shape[1], 1, -1):
            for i in range (0,10):
                nn = SingleLayerNN.SingleLayerNN(self.X_train.shape[1], i, j)
                single_neuron_nn.append(nn)
                nn.fit(x=self.X_train,
                         y=self.y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=self.verbose,
                         validation_split=self.validation_split)
                prediction = nn.predict(x=self.X_test)
                print("*****Run:",i,",",j)
                print(confusion_matrix(self.y_test, prediction))
