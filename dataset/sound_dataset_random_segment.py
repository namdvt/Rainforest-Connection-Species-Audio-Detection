from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import cv2
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from glob import glob
import librosa as lib
from tqdm import tqdm
import librosa as lib
import numpy as np
from glob import glob
from scipy.signal import butter, lfilter, freqz
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask, AddImpulseResponse, \
    FrequencyMask, Shift
import os
import warnings
warnings.simplefilter("ignore")

sr = 48000

def normalize(sound):
    max_ = sound.max()
    min_ = sound.min()
    n_sound = (sound - min_) / (max_ - min_)

    return n_sound


def crop_pad(sound, length):
    length = int(length)
    if len(sound) < length:
        num_zeros = length - len(sound)
        num_left_zeros = random.randint(0, num_zeros)
        left_zeros = np.zeros(num_left_zeros)
        right_zeros = np.zeros(num_zeros - num_left_zeros)
        sound = np.concatenate([left_zeros, sound, right_zeros])
    if len(sound) > length:
        offset = random.randint(0, len(sound) - length)
        sound = sound[offset:offset + length]

    return sound


class SoundDataset(Dataset):
    def __init__(self, annotation, data_type, length, augment, prop_tp=None, use_spec=True):
        super().__init__()
        try:
            self.indexes = open(annotation).read().splitlines()
        except:
            self.indexes = annotation
        self.data_type = data_type
        self.use_spec = use_spec
        self.prop_tp = prop_tp
        self.tp_data_60s = glob('/home/datasets/rain_forest/data_waveform/tp_60s/*.npy')
        self.tp_data = glob('/home/datasets/rain_forest/data_waveform/tp/*.npy')
        self.species = glob('/home/datasets/rain_forest/data_waveform/species/*.npy')
        self.fp_data = glob('/home/datasets/rain_forest/data_waveform/fp/*.npy')
        self.length = length
        self.augment = augment

    def transform(self, waveform, pad=True, augment=False):
        # load, augment and crop

        if augment:
            waveform = self.augmenter(waveform, sample_rate=sr)
        if pad:
            waveform = crop_pad(waveform, self.length)


        # spectrogram
        spec = lib.feature.melspectrogram(waveform, sr=sr)
        spec = lib.power_to_db(spec)
        spec = normalize(spec)

        # expand dimension
        spec = torch.tensor(spec)
        spec = torch.stack((spec, spec, spec), dim=0).float()

        return spec


    def get_tp_60s(self, file_name, augment=False):
        while True:
            sample = random.choice(self.tp_data_60s)
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = F.one_hot(torch.tensor(int(sample.split('_')[-1].split('.')[0])), 24).float()
                waveform = np.load(sample)
                return self.transform(waveform, augment=augment), label


    def get_tp(self, file_name, augment=False):
        while True:
            sample = random.choice(self.tp_data)
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = torch.tensor(np.array(sample.split('[')[1].split(']')[0].split(' '), dtype='float32'))
                waveform = np.load(sample)
                return self.transform(waveform, augment=augment), label

    def get_species(self, file_name, augment=False):
        while True:
            sample = random.choice(self.species)
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = F.one_hot(torch.tensor(int(sample.split('/')[-1].split('_')[1])), 24).float()
                waveform = np.load(sample)
                return self.transform(waveform, pad=True, augment=augment), label


    def __getitem__(self, index):
        file_name = self.indexes[index]
        if self.data_type == 'train':
            prop = random.random()
            if prop <= self.prop_tp:  # tp
                tp_type = random.choice(['tp_data_60s', 'tp_data', 'species'])
                if tp_type == 'tp_data_60s':
                    sound, label = self.get_tp_60s(file_name, augment=self.augment)
                    label = torch.cat([torch.ones(1), label])

                if tp_type == 'species':  # species
                    sound, label = self.get_species(file_name, augment=self.augment)
                    label = torch.cat([torch.ones(1), label])

                if tp_type == 'tp_data':  # species
                    sound, label = self.get_tp(file_name, augment=self.augment)
                    label = torch.cat([torch.ones(1), label])

            else:  # fp
                sound_address = random.choice(self.fp_data)
                waveform = np.load(sound_address)
                sound = self.transform(waveform, pad=True, augment=self.augment)
                label = F.one_hot(torch.tensor(int(sound_address.split('/')[-1].split('_')[1])), 24).float()
                label = torch.cat([torch.zeros(1), label])
        else:
            sound, label = self.get_tp(file_name, augment=False)
            label = torch.cat([torch.ones(1), label])

        # normalize
        

        return sound, label

    def __len__(self):
        return len(self.indexes)


def get_loader(annotation, data_type, batch_size, num_workers, augment, prop_tp=None, use_spec=True):
    dataset = SoundDataset(annotation=annotation, data_type=data_type, length=10 * sr, augment=augment, prop_tp=prop_tp, use_spec=use_spec)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        shuffle=True,
                        )

    return loader


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    loader = get_loader(annotation='/home/rainforest/dataset/fold_split/split_1/1/train_list.txt', data_type='train', batch_size=1, num_workers=0,
                        augment=False, prop_tp=0.9, use_spec=False)
    # loader = get_loader_1type(annotation='tp_val.txt', data_type='tp', batch_size=64, num_workers=8, augment=False)
    a = 0
    for s, l in tqdm(loader):
        a += 1
