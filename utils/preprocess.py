import pandas as pd
import librosa as lib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz

sr = 48000

def export_species():
    tp = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_fp.csv')
    for row in tqdm(tp.iterrows()):
        recording_id = row[1]['recording_id']
        species = row[1]['species_id']
        songtype = row[1]['songtype_id']
        t_min = row[1]['t_min']
        t_max = row[1]['t_max']
        f_min = row[1]['f_min']
        f_max = row[1]['f_max']

        y, _ = lib.load('/home/cybercore/oldhome/datasets/rain_forest/train/' + recording_id + '.flac', sr=sr, offset=t_min, duration=t_max - t_min, res_type='kaiser_fast')
        np.save('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/fp/' + recording_id + '_' + str(species) + '_' + str(songtype) + '_' + str(t_min) + '_' +
                str(t_max) + '_' + str(f_min) + '_' + str(f_max) + '.npy', y)


def preprocess_test():
    sound_list = glob('/home/cybercore/oldhome/datasets/rain_forest/train/*.flac')
    for sound in tqdm(sound_list):
        y, _ = lib.load(sound, sr=sr, res_type='kaiser_fast')
        np.save('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/train_unlabel/' + sound.split('/')[-1].split('.')[0] +'.npy', y)


def preprocess_train():
    duration = 10
    overlap = 8
    df = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')

    sound_list = df['recording_id'].values
    for name in tqdm(sound_list):
        path = '/home/cybercore/oldhome/datasets/rain_forest/train/' + name + '.flac'
        
        infos = df.loc[df['recording_id'] == name].values.tolist()
        try:
            y, _ = lib.load(path, sr=sr, res_type='kaiser_fast')
        except:
            print(name)
            continue
        offset = 0
        while offset < 60:
            # get label
            label = np.zeros(24, dtype='int8')
            for info in infos:
                tmin = info[3]
                tmax = info[5]
                if (offset < tmin and offset + duration > tmax):
                    label[int(info[1])] += 1
            label[label > 1] = 1
                    
            # get sample
            if label.sum() > 0:
                segment = y[int(offset * sr) : int((offset + duration) * sr)]       
                np.save('data_melspec/tp/' + name + '_' + str(offset) + '_' + str(label) + '.npy', segment)
            
            offset += duration - overlap


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def export_background():
    df = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')

    sound_list = df['recording_id'].values
    for name in tqdm(sound_list):
        path = '/home/cybercore/oldhome/datasets/rain_forest/train/' + name + '.flac'
        tmin = 60
        tmax = 0
        infos = df.loc[df['recording_id'] == name].values.tolist()
        for info in infos:
            if tmin > info[3]:
                tmin = info[3]
            if tmax < info[5]:
                tmax = info[5]
        
        try:
            left, _ = lib.load(path, sr=sr, offset=0, duration=tmin, res_type='kaiser_fast')
            right, _ = lib.load(path, sr=sr, offset=tmax, duration=None, res_type='kaiser_fast')
        except:
            continue

        np.save('data_melspec/background/' + name + '_0_' + str(tmin) + '.npy', left)
        np.save('data_melspec/background/' + name + '_' + str(tmax) + '_60.npy', right)


def get_freq_range():
    df = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')
    species_list = df['species_id'].values
    # species_list.sort()
    freq_list = list()
    for s in set(species_list):
        fmin = sr/2
        fmax = 0
        infos = df.loc[df['species_id'] == s].values.tolist()
        for info in infos:
            if fmin > info[-3]:
                fmin = info[-3]
            if fmax < info[-1]:
                fmax = info[-1]

        freq_list.append((fmin, fmax))
    print()

if __name__ == '__main__':
    preprocess_test()
