from src.utility import musicUtility
from src.utility import dataUtility
from src.baseline_implementation import svm_implementaion


# define constants here
data_file_name = 'music_csv_data\data_prof.csv' #'music_csv_data\data.csv'#

# musicUtility.generate_music_data("TrainingData", data_file_name)

# load the data in memory
X_train, X_test, y_train, y_test = dataUtility.get_test_train_set(data_file_name, 0.8, False)


# baseline implementations
svm_implementaion.run_svm(X_train, X_test, y_train, y_test, 3, 73, 10)
