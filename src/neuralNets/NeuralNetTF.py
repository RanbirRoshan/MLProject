from src.neuralNets import  SingleLayerNN


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
        self.verbose = 1 #0 = silent, 1 = progress bar, 2 = one line per epoch.


    def Execute(self):
        single_neuron_nn = SingleLayerNN.SingleLayerNN(26)
        single_neuron_nn.fit(x=self.X_train,
                             y=self.y_train,
                             batch_size=self.batch_size,
                             verbose=self.verbose,
                             validation_split=self.validation_split)
