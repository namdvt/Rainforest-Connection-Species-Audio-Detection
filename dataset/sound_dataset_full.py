import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import cv2
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
import librosa as lib
from tqdm import tqdm
import librosa as lib
import numpy as np
from glob import glob
import os
import pandas as pd
from torch.utils.data import WeightedRandomSampler
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
    def __init__(self, annotation, data_type, length, augment, prop_tp=None, threshold=0.998, hard_mining=None, weighted_sampler=False, external=None):
        super().__init__()
        self.data_type = data_type
        self.weighted_sampler = weighted_sampler
        self.prop_tp = prop_tp
        self.data_tp = dict()
        self.data_fp = dict()
        self.tp_weight = list()
        
        if data_type == 'train':
            self.data_tp.update(self._get_tp_data(annotation, hard_mining))
            self.data_tp.update(self._get_species_data(annotation))
            self.data_tp.update(self._get_external_data(external))
            self.data_fp.update(self._get_fp_data())
            self.fp_indexes = list(self.data_fp.keys())
        else:
            self.data_tp.update(self._get_tp_data(annotation))

        self.length = length
        self.augment = augment
        self.tp_indexes = list(self.data_tp.keys())
        print('num samples for ' + data_type + ': ' + str(len(self.tp_indexes)))

    def _get_tp_data(self, annotation, hard_mining=None):
        eps = 0.000001
        tp_data = dict()
        
        # tp
        all_tp = glob('/home/datasets/rain_forest/data_waveform/tp/*.npy')
        list_filename = open(annotation).read().splitlines()

        if self.weighted_sampler:
            df = pd.read_csv(hard_mining)
            for path in all_tp:
                if path.split('/')[-1].split('_')[0] in list_filename:
                    label = torch.tensor(np.array(path.split('[')[1].split(']')[0].split(' '), dtype='float32'))
                    label = torch.cat([torch.ones(1), label])
                    tp_data[path] = label
                    self.tp_weight.append(np.log(df.loc[df['path'] == path]['loss'].to_numpy()[0] + eps))

            self.tp_weight = np.array(self.tp_weight)
            self.tp_weight = np.abs(self.tp_weight - self.tp_weight.min() + eps)
        else:
            for path in all_tp:
                if path.split('/')[-1].split('_')[0] in list_filename:
                    label = torch.tensor(np.array(path.split('[')[1].split(']')[0].split(' '), dtype='float32'))
                    label = torch.cat([torch.ones(1), label])
                    tp_data[path] = label

        # filter hard samples
        if hard_mining:
            # get all hard samples
            hard_samples = list()
            for idx, row in df.iterrows():
                if np.array(row[2], dtype='float') >= 0.0001:
                    hard_samples += [row[1]]

            tp_hard = [(k, v) for k, v in tp_data.items() if k in hard_samples]
            print('hard mining mode')
            return tp_hard

        return tp_data


    def _get_species_data(self, annotation):
        tp_data = dict()

        # species
        all_species = glob('/home/datasets/rain_forest/data_waveform/species/*.npy')
        list_filename = open(annotation).read().splitlines()
        for path in all_species:
            if path.split('/')[-1].split('_')[0] in list_filename:
                label = F.one_hot(torch.tensor(int(path.split('/')[-1].split('_')[1])), 24).float()
                label = torch.cat([torch.ones(1), label])
                tp_data[path] = label

        return tp_data

    def _get_fp_data(self):
        fp_data = dict()
        for path in glob('/home/datasets/rain_forest/data_waveform/fp/*.npy'):
            label = F.one_hot(torch.tensor(int(path.split('/')[-1].split('_')[1])), 24).float()
            label = torch.cat([torch.zeros(1), label])
            fp_data[path] = label

        return fp_data

    def _get_external_data(self, annotation, threshold=0.98):
        external_data = dict()
        df = pd.read_csv(annotation)
        for _, row in df.iterrows():
            score = np.array(row[2].replace('[','').replace(']','').replace(' ','').split(','), dtype='float').max()
            if score > threshold:
                path = row[1]
                label = torch.tensor(np.array(row[2].replace('[','').replace(']','').replace(' ','').split(','), dtype='float'))
                label[label >= threshold] = 1
                label[label < threshold] = 0
                label = torch.cat([torch.ones(1), label])
                external_data[path] = label
        
        print('load external data from ' + annotation)
        return external_data


    def _get_spectrogram(self, waveform, pad=True, augment=False):
        if pad:
            waveform = crop_pad(waveform, self.length)

        # spectrogram
        spec = lib.feature.melspectrogram(waveform, sr=sr)
        spec = lib.power_to_db(spec)
        spec = normalize(spec)

        spec = torch.tensor(spec)
        spec = torch.stack((spec, spec, spec), dim=0)

        return spec

    def __getitem__(self, index):
        # get index
        if self.data_type == 'train':
            prop = random.random()
            if prop <= self.prop_tp:
                path = self.tp_indexes[index]
                label = self.data_tp.get(path)
            else:
                path = random.choice(self.fp_indexes)
                label = self.data_fp.get(path)
        else:
            path = self.tp_indexes[index]
            label = self.data_tp.get(path)

        # load sample
        sound = np.load(path)
        if len(sound.shape) == 4:
            sound = torch.tensor(sound)
            sound = sound.squeeze()
        else:
            sound = self._get_spectrogram(sound)

        return sound.float(), label.float()

    def __len__(self):
        return len(self.tp_indexes)


def get_loader(annotation, data_type, batch_size, num_workers, augment, prop_tp=None, threshold=0.998, hard_mining=None, weighted_sampler=False, external=None):
    dataset = SoundDataset(annotation=annotation, data_type=data_type, length=10 * sr, augment=augment, prop_tp=prop_tp, threshold=threshold, hard_mining=hard_mining, weighted_sampler=weighted_sampler, external=external)

    if weighted_sampler:
        sampler = WeightedRandomSampler(dataset.tp_weight, len(dataset.tp_weight))
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            drop_last=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=sampler,
                            )
    else:
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

    loader = get_loader(annotation='dataset/fold_split/split_1/1/train_list.txt', data_type='train', batch_size=2, num_workers=0,
                        augment=False, prop_tp=0.9, hard_mining='dataset/hard_mining/hard_mining_fold1.csv', weighted_sampler=True)
    # loader = get_loader(annotation='dataset/fold_split/split_1/1/train_list.txt', data_type='train', batch_size=2, num_workers=0,
    #                     augment=False, prop_tp=0.9)
    a = 0
    for s, l in tqdm(loader):
        a += 1
