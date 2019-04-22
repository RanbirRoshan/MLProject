#from src.utility import musicUtility
import sys
from src.utility import TestJson
from src.utility import dataUtility
from src.baseline_implementation import baseline
from src.neuralNets.NeuralNetTF import NeuralNet
import random
import os


#constanst
epochs = 20
batch_size = 2
lstm_epochs = 20
lstm_batch_size = 16
cnn_epochs = 5
cnn_batch_size = 16


def main():

    #data_format = NCHW

    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['KMP_SETTINGS'] = 'TRUE'
    os.environ['KMP_BLOCKTIME'] = '1'
    random.seed(a=5)
    # define constants here
    data_file_name = 'data_prof_temp.csv' #'data.csv'#
    use_prev_wait = False
    just_validate = False

    #musicUtility.generate_music_data("TrainingData", data_file_name, "music_csv_data")

    data_file_name = "./music_csv_data/" + data_file_name
    # load the data in memory
    X_train, X_test, y_train, y_test, is_multi_class = dataUtility.get_test_train_set(data_file_name, 0.8, True)
    X_train_LSTM, X_testLSTM, y_train_LSTM, y_test_LSTM, a, song_det = dataUtility.load_LSTM_data(data_file_name, 0.8, "music_csv_data")

    # baseline implementations
    #baseline.run_all_baselines(X_train, X_test, y_train, y_test, is_multi_class)

    #start neural net executions
    nn = NeuralNet(batch_size, epochs, X_train, X_test, y_train, y_test, is_multi_class, lstm_batch_size, lstm_epochs, cnn_batch_size, cnn_epochs)
    #nn.Execute()
    #nn.ExecuteLSTM(X_train_LSTM, X_testLSTM, y_train_LSTM, y_test_LSTM)
    nn.ExecuteCNN(X_train_LSTM, X_testLSTM, y_train_LSTM, y_test_LSTM, song_det, use_prev_wait, just_validate)

if __name__ == '__main__':
    main()
    sys.exit()
