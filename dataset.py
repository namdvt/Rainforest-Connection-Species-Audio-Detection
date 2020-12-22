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
    def __init__(self, annotation, data_type, length, augment, prop_tp=None):
        super().__init__()
        self.indexes = open(annotation).read().splitlines()
        self.data_type = data_type
        self.prop_tp = prop_tp
        self.tp_data = glob('data_waveform/tp/*.npy')
        # self.background = glob('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/background/*.npy')
        self.species = glob('data_waveform/species/*.npy')
        self.fp_data = glob('data_waveform/fp/*.npy')
        # self.unlabel = glob('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/train_pseudo/*.npy')
        self.length = length
        self.augment = augment
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.005, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1, p=0.5)
        ])

    def get_spectrogram(self, sound_address):
        # load, augment and crop
        waveform = np.load(sound_address)
        waveform = crop_pad(waveform, self.length)

        # spectrogram
        spec = lib.feature.melspectrogram(waveform, sr=sr)
        spec = lib.power_to_db(spec)
        spec = normalize(spec)
        assert spec.shape == (128, 938)

        return spec

    def get_tp(self, file_name):
        while True:
            sample = random.choice(self.tp_data)
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = torch.tensor(np.array(sample.split('[')[1].split(']')[0].split(' '), dtype='float32'))

                return self.get_spectrogram(sample), label

    def get_all_tp(self, file_name):
        tps = list()
        for sample in self.tp_data:
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = torch.tensor(np.array(sample.split('[')[1].split(']')[0].split(' '), dtype='float32'))
                tps.append((sample, label))
        tps.sort()
        return tps

    def get_mixed_tp(self, file_name):
        sample1, label1 = self.get_all_tp(file_name)[-1]
        sample1 = self.get_spectrogram(sample1)

        random_name = random.choice(self.indexes)
        sample2, label2 = self.get_all_tp(random_name)[0]
        sample2 = self.get_spectrogram(sample2)

        # mix label
        label = label1 + label2
        label[label > 0] = 1

        # mix spectrogram
        t = np.arange(-469, 469, 1) / random.randint(20, 80)
        sigmoid = 1 - 1 / (1 + np.exp(-t))

        sample = sigmoid * sample1 + (1 - sigmoid) * sample2

        return sample, label

    def get_species(self, file_name):
        while True:
            sample = random.choice(self.species)
            if sample.split('/')[-1].split('_')[0] == file_name:
                label = F.one_hot(torch.tensor(int(sample.split('/')[-1].split('_')[1])), 24).float()

                return self.get_spectrogram(sample), label

    def get_mixed_species(self, file_name):
        sample1, label1 = self.get_species(file_name)
        sample2, label2 = self.get_species(random.choice(self.indexes))

        # mix label
        label = label1 + label2
        label[label > 0] = 1

        # mix spectrogram
        t = np.arange(-469, 469, 1) / random.randint(150, 300)
        sigmoid = 1 / (1 + np.exp(-t))
        if random.choice([0, 1]):
            sigmoid = 1 - sigmoid

        sample = sigmoid * sample1 + (1 - sigmoid) * sample2

        return sample, label

    def get_unlabel(self):
        sample = random.choice(self.unlabel)
        label = torch.tensor(np.array(sample.split('[')[1].split(']')[0].split(' '), dtype='float32'))

        return self.get_spectrogram(sample), label

    def __getitem__(self, index):
        file_name = self.indexes[index]
        if self.data_type == 'train':
            prop = random.random()
            if prop <= self.prop_tp:  # tp
                if random.choice([0, 1]):
                    sound, label = self.get_tp(file_name)
                    label = torch.cat([torch.ones(1), label])
                else:  # species
                    sound, label = self.get_species(file_name)
                    label = torch.cat([torch.ones(1), label])

                if self.augment:
                    if random.choice([0, 1]):
                        if random.choice([0, 1]):
                            sound, label = self.get_mixed_tp(file_name)
                            label = torch.cat([torch.ones(1), label])
                        else:  # species
                            sound, label = self.get_mixed_species(file_name)
                            label = torch.cat([torch.ones(1), label])

            else:  # fp
                sound_address = random.choice(self.fp_data)
                sound = self.get_spectrogram(sound_address)
                label = F.one_hot(torch.tensor(int(sound_address.split('/')[-1].split('_')[1])), 24).float()
                label = torch.cat([torch.zeros(1), label])
        else:
            sound, label = self.get_tp(file_name)
            label = torch.cat([torch.ones(1), label])

        # normalize
        # sound = cv2.resize(sound, dsize=(100, 700))
        sound = torch.tensor(sound)
        sound = torch.stack((sound, sound, sound), dim=0).float()
        # sound = normalize(sound)

        return sound, label

    def __len__(self):
        return len(self.indexes)


def get_loader(annotation, data_type, batch_size, num_workers, augment, prop_tp=None):
    dataset = SoundDataset(annotation=annotation, data_type=data_type, length=10 * sr, augment=augment, prop_tp=prop_tp)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        shuffle=True,
                        )

    return loader

#
# class SoundEmbeddingDataset(Dataset):
#     def __init__(self, annotation, length, augment):
#         super().__init__()
#         self.indexes = open(annotation).read().splitlines()
#         self.length = length
#         self.augment = augment
#         self.tp_data = glob('data_waveform/tp/*.npy')
#         self.species = glob('data_waveform/species/*.npy')
#         self.augmenter = Compose([
#             AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),
#             TimeStretch(min_rate=0.6, max_rate=0.9, p=0.5),
#             Shift(min_fraction=-1, max_fraction=1, p=0.5),
#         ])
#
#     def get_sample(self, file_name):
#         while True:
#             sample = random.choice(self.tp_data)
#             if sample.split('/')[-1].split('_')[0] == file_name:
#                 label = torch.tensor(np.array(sample.split('[')[1].split(']')[0].split(' '), dtype='float32'))
#
#                 return sample, label.argmax()
#
#     def get_species(self, file_name):
#         while True:
#             sample = random.choice(self.species)
#             if sample.split('/')[-1].split('_')[0] == file_name:
#                 label = torch.tensor(int(sample.split('/')[-1].split('_')[1]))
#
#                 return sample, label
#
#     def __getitem__(self, index):
#         file_name = self.indexes[index]
#         if random.choice([0, 1]):
#             sound_address, label = self.get_sample(file_name)
#         else:
#             sound_address, label = self.get_species(file_name)
#
#         sound = np.load(sound_address)
#
#         if self.augment:
#             sound = self.augmenter(sound, sample_rate=sr)
#
#         sound = crop_pad(sound, self.length)
#
#         # convert to spectrogram
#         sound = lib.feature.melspectrogram(sound, sr=sr)
#         sound = lib.power_to_db(sound)
#
#         assert sound.shape == (128, 938)
#
#         # normalize
#         sound = torch.tensor(sound)
#         sound = torch.stack((sound, sound, sound), dim=0).float()
#         sound = normalize(sound)
#
#         return sound, label
#
#     def __len__(self):
#         return len(self.indexes)
#
#
# def get_loader_embedding(annotation, batch_size, num_workers, augment):
#     dataset = SoundEmbeddingDataset(annotation=annotation, length=10 * sr, augment=augment)
#
#     loader = DataLoader(dataset=dataset,
#                         batch_size=batch_size,
#                         drop_last=True,
#                         num_workers=num_workers,
#                         pin_memory=True,
#                         shuffle=True,
#                         )
#
#     return loader


if __name__ == '__main__':
    loader = get_loader(annotation='fold/3/train_filenames.txt', data_type='train', batch_size=2, num_workers=0,
                        augment=True, prop_tp=0.9)
    # loader = get_loader_1type(annotation='tp_val.txt', data_type='tp', batch_size=64, num_workers=8, augment=False)
    a = 0
    for s, l in tqdm(loader):
        a += 1
