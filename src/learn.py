import numpy as np
import librosa
from src.utility import musicUtility
from src.utility import dataUtility
import pandas
from sklearn.preprocessing import Normalizer

# define constants here
data_file_name = 'music_csv_data\data.csv'

# musicUtility.generate_music_data("genres", data_file_name)

# load the data in memory
X_train, X_test, y_train, y_test = dataUtility.get_test_train_set(data_file_name, 0.8)

print(X_test.shape)
print(y_test.shape)



