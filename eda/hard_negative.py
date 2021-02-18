import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import pandas as pd
import pywt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataset.sound_dataset import normalize
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa as lib
import torch.nn as nn

from model.model import Model

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
sr = 48000

if __name__ == '__main__':
    for fold in ['2','3','4','5']:
        backbone = 'selecsls42b'
        model = Model(backbone=backbone).to(device)
        weight = 'output/12/selecsls42b_' + fold + '.pth'
        model.load_state_dict(torch.load(weight, map_location=device), strict=True)
        model.eval()
        print('loaded :' + weight)

        num = 0
        hard_mining_fold1 = pd.DataFrame(columns=['id', 'path', 'loss'])
        for filename in tqdm(glob('/home/datasets/rain_forest/data_waveform/tp/*.npy')):
            segment = np.load(filename)
            target = torch.tensor(np.array(filename.split('[')[1].split(']')[0].split(' '), dtype='float32')).to(device)

            spec = lib.feature.melspectrogram(segment, sr=sr)
            spec = lib.power_to_db(spec)

            spec = torch.tensor(spec)
            spec = torch.stack((spec, spec, spec), dim=0).float().unsqueeze(0).to(device)
            spec = normalize(spec)

            outputs = list()
            with torch.no_grad():
                pred = model(spec)

            loss = F.binary_cross_entropy(pred, target)
            ann = (num, filename, loss.item())
            hard_mining_fold1 = hard_mining_fold1.append(pd.Series(ann, index=hard_mining_fold1.columns), ignore_index=True)

            num += 1

        hard_mining_fold1.to_csv('hard_mining_fold' + fold + '.csv', index=False)