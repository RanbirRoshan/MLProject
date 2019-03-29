from src.utility import musicUtility
from src.utility import dataUtility
from src.baseline_implementation import baseline
from src.neuralNets.NeuralNetTF import NeuralNet


#constanst
epochs = 128
batch_size = 32

def main():
    # define constants here
    data_file_name = 'music_csv_data\data.csv'#'music_csv_data\data_prof.csv' #'music_csv_data\data.csv'#

    # musicUtility.generate_music_data("TrainingData", data_file_name)

    # load the data in memory
    X_train, X_test, y_train, y_test, is_multi_class = dataUtility.get_test_train_set(data_file_name, 0.8, False)

    # baseline implementations
    #baseline.run_all_baselines(X_train, X_test, y_train, y_test, is_multi_class)

    #start neural net executions
    nn = NeuralNet(batch_size, epochs, X_train, X_test, y_train, y_test, is_multi_class)
    nn.Execute()


if __name__ == '__main__':
    main()
