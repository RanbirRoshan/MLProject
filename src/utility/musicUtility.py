import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import csv


def display_music_waveform(data, sample_rate, label=None):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(data, sr=sample_rate)
    plt.title(label)


def display_music_spectrogram(data, sample_rate, x_axis, y_axis, label=None):
    local_data = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(local_data))
    plt.figure(figsize=(14, 5))
    # https://librosa.github.io/librosa/generated/librosa.display.specshow.html
    librosa.display.specshow(xdb, sr=sample_rate, x_axis=x_axis, y_axis=y_axis)
    plt.title(label)
    plt.colorbar()


def generate_music_data(folder_name, out_file_name):
    file = open(out_file_name, 'w', newline='')

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    for g in genres:
        for file_name in os.listdir(f'./'+folder_name+'/'+g):
            song_name = f'./'+folder_name+'/'+g+'/'+file_name
            x, sr = librosa.load(song_name, sr=44100)
            chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(x)
            mfcc = librosa.feature.mfcc(y=x, sr=sr)
            rmse = librosa.feature.rmse(y=x)
            to_append = f'{file_name} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())




