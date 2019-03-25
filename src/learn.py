import numpy as np
import librosa
from src.utility import musicUtility
import pandas
from sklearn.preprocessing import Normalizer

# define constants here
data_file_name = 'music_csv_data\data.csv'

# musicUtility.generate_music_data("genres", data_file_name)


# load the data in memory

X, Y = musicUtility.load_data(data_file_name)

print(X)



