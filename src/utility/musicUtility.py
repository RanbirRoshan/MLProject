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


def write_LSTM_data(file_name, folder_name, mfcc, tonnetz, chroma_stft):
    out_file_name = "./" + folder_name + file_name
    file = open(out_file_name, 'w', newline='')

    header = 'seq'
    for i in range(1, 21):
        header += f' mfcc{i} '
    for i in range(1, 7):
        header += f' tonnetz{i}'
    for i in range(1, 13):
        header += f' chroma_stft{i}'

    header = header.split()
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for i in range(0, mfcc.shape[1]):
        to_append = f'{i}'
        for j in range(0, mfcc.shape[0]):
            to_append += f' {mfcc[j][i]}'
        for j in range(0, tonnetz.shape[0]):
            to_append += f' {tonnetz[j][i]}'
        for j in range(0, chroma_stft.shape[0]):
            to_append += f' {chroma_stft[j][i]}'

        file = open(out_file_name, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


def generate_music_data(folder_name, out_file_name, outputFolderName):
    out_file_name = "./" + outputFolderName + './' + out_file_name
    file = open(out_file_name, 'w', newline='')

    header = 'filename tempo mean_beat poly_fet0 poly_fet1 poly_fet2 spectral_flatness rms rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'

    for i in range(1, 7):
        header += f' tonnetz_mean{i}'
        header += f' tonnetz_var{i}'
        header += f' tonnetz_std{i}'
    header += " chroma_direction_change chroma_peak_distance_avg"
    for i in range(1, 13):
        header += f' chroma_stft_mean{i}'
        header += f' chroma_stft_var{i}'
        header += f' chroma_stft_std{i}'
        
    header += f' chroma_cqt_direction_change chroma_cqt_peak_distance_avg'
    for i in range(1, 13):
        header += f' chroma_cqt_mean{i}'
        header += f' chroma_cqt_var{i}'
        header += f' chroma_cqt_std{i}'
        
    header += f' var_rolloff std_rolloff max_rolloff min_rolloff'
    header += f' var_zcr std_zcr max_zcr min_zcr'
        
    for i in range(1, 13):
        header += f' chroma_sens{i}'
    for i in range(1, 129):
        header += f' melspectogram{i}'
    for i in range(1, 8):
        header += f' spectral_contrast{i}'
    header += ' label LSTMFileLoc'
    header = header.split()

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    genres = 'Prog NonProg'.split()

    #missing spectral contrast, tempogram

    count = 0

    for g in genres:
        for file_name in os.listdir(f'./'+folder_name+'/'+g):
            count = count+1
            song_name = f'./'+folder_name+'/'+g+'/'+file_name

            print(count)

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
            #tempogram = librosa.feature.tempogram(y=x, sr=sr)
            to_append = f'{file_name.replace(" ", "")} {tempo} {np.mean(beats)} {np.mean(poly_fet0)} {np.mean(poly_fet1)} {np.mean(poly_fet2)} {np.mean(spec_flat)} {np.mean(rms)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            for e in tonnetz:
                to_append += f' {np.mean(e)}'
                to_append += f' {np.var(e)}'
                to_append += f' {np.std(e)}'
            
            pos = 0
            ans = 0
            total_peaks_distance = 0
            rising_trend = True
            for i in range (0, chroma_stft.shape[1]):
                prev_pos = pos
                pos = 0
                for j in range (0, chroma_stft.shape[0]):
                    if (chroma_stft[j][i] > chroma_stft[pos][i]):
                        pos = j
                if pos > prev_pos and not rising_trend:
                    ans = ans + 1
                    total_peaks_distance = pos - prev_pos +total_peaks_distance
                    rising_trend = True
                elif pos < prev_pos and rising_trend:
                    total_peaks_distance = prev_pos - pos + total_peaks_distance
                    ans = ans + 1
                    rising_trend = False
            to_append += f' {ans/chroma_cqt.shape[1]} {total_peaks_distance/ans}'
                    
            for e in chroma_stft:
                to_append += f' {np.mean(e)}'
                to_append += f' {np.var(e)}'
                to_append += f' {np.std(e)}'
                
            
            pos = 0
            ans = 0
            total_peaks_distance = 0
            rising_trend = True
            for i in range(0, chroma_cqt.shape[1]):
                prev_pos = pos
                pos = 0
                for j in range(0, chroma_cqt.shape[0]):
                    if chroma_cqt[j][i] > chroma_cqt[pos][i]:
                        pos = j
                if pos > prev_pos and not rising_trend:
                    ans = ans + 1
                    total_peaks_distance = pos - prev_pos + total_peaks_distance
                    rising_trend = True
                elif pos < prev_pos and rising_trend:
                    total_peaks_distance = prev_pos - pos + total_peaks_distance
                    ans = ans + 1
                    rising_trend = False

            to_append += f' {ans / chroma_cqt.shape[1]} {total_peaks_distance / ans}'
            for e in chroma_cqt:
                to_append += f' {np.mean(e)}'
                to_append += f' {np.var(e)}'
                to_append += f' {np.std(e)}'
                
            to_append += f' {np.var(rolloff)} {np.std(rolloff)} {np.max(rolloff)} {np.min(rolloff)}'
            to_append += f' {np.var(zcr)} {np.std(zcr)} {np.max(zcr)} {np.min(zcr)}'
            
            for e in chroma_sens:
                to_append += f' {np.mean(e)}'
            for e in melspectrogram:
                to_append += f' {np.mean(e)}'
            for e in contrast:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'

            name = "/SongWiseData/" + file_name.replace(" ", "") + "_" + str(sr) + ".csv"
            to_append += f' {name}'

            file = open(out_file_name, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

            write_LSTM_data(name, outputFolderName, mfcc, tonnetz, chroma_stft)