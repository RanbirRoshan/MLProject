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

    header = 'filename tempo mean_beat poly_fet0 poly_fet1 poly_fet2 spectral_flatness rms rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(0, 6):
        header += f' tonnetz{i}'
    for i in range(1, 13):
        header += f' chroma_stft{i}'
    for i in range(1, 13):
        header += f' chroma_cqt{i}'
    for i in range(1, 13):
        header += f' chroma_sens{i}'
    for i in range(1, 129):
        header += f' melspectogram{i}'
    for i in range(1, 8):
        header += f' spectral_contrast{i}'
    header += ' label'
    header = header.split()

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    genres = 'NonProg Prog'.split()

    #missing spectral contrast, tempogram

    for g in genres:
        for file_name in os.listdir(f'./'+folder_name+'/'+g):
            song_name = f'./'+folder_name+'/'+g+'/'+file_name
            x, sr = librosa.load(song_name, sr=44100)
            chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
            poly_fet0 = librosa.feature.poly_features(y=x, sr=sr, order=0)
            poly_fet1 = librosa.feature.poly_features(y=x, sr=sr, order=1)
            poly_fet2 = librosa.feature.poly_features(y=x, sr=sr, order=2)
            contrast = librosa.feature.spectral_contrast(y=x,sr=sr)
            tempo, beats = librosa.beat.beat_track(y=x, sr=sr)
            chroma_cqt = librosa.feature.chroma_cqt(y=x, sr=sr)
            chroma_sens = librosa.feature.chroma_cens(y=x, sr=sr)
            melspectrogram = librosa.feature.melspectrogram(y=x, sr=sr)
            mfcc = librosa.feature.mfcc(y=x, sr=sr)
            rms = librosa.feature.rms(y=x)
            rmse = librosa.feature.rmse(y=x)
            spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
            spec_flat = librosa.feature.spectral_flatness(y=x)
            rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(x)
            tonnetz = librosa.feature.tonnetz(y=x, sr=sr)
            to_append = f'{file_name.replace(" ", "")} {tempo} {np.mean(beats)} {np.mean(poly_fet0)} {np.mean(poly_fet1)} {np.mean(poly_fet2)} {np.mean(spec_flat)} {np.mean(rms)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            for e in tonnetz:
                to_append += f' {np.mean(e)}'
            for e in chroma_stft:
                to_append += f' {np.mean(e)}'
            for e in chroma_cqt:
                to_append += f' {np.mean(e)}'
            for e in chroma_sens:
                to_append += f' {np.mean(e)}'
            for e in melspectrogram:
                to_append += f' {np.mean(e)}'
            for e in contrast:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open(out_file_name, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
