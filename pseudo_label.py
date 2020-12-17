import cv2
import pandas as pd
import pywt
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from dataset import normalize
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa as lib
from scipy.stats.mstats import gmean

from model import Model

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
backbone = 'legacy_seresnet34'
sr = 48000
    
print('inferencing model ' + backbone)

if __name__ == '__main__':
    # load model
    models = list()
    weights = glob('output/0.891/*.pth')
    for weight in weights:
        model = Model(backbone=backbone).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()
        models.append(model)
        print('loaded ' + weight)

    # prepare data
    tp = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')
    tp_list = set()
    for row in tp.iterrows():
        tp_list.add(row[1]['recording_id'])

    list_train = glob('/home/cybercore/oldhome/datasets/rain_forest/train/*.flac')

    # pseudo_label
    for test in tqdm(list_train):
        if raw.split('/')[-1].split('.')[0] in tp_list:
            continue
        
        waveform, _ = lib.load(raw, sr=sr, res_type='kaiser_fast')
        waveform = np.load(test)

        batch = list()
        for idx in range(0, 50*sr, 2*sr):
            segment = waveform[idx:idx + 10*sr]
            spec = lib.feature.melspectrogram(segment, sr=sr)
            spec = lib.power_to_db(spec)

            spec = torch.tensor(spec)
            spec = torch.stack((spec, spec, spec), dim=0).float().unsqueeze(0).to(device)
            spec = normalize(spec)

            outputs = list()
            with torch.no_grad():
                for model in models:
                    output = model(spec)
                    outputs.append(output)

            outputs = torch.cat(outputs)
            outputs = gmean(outputs.cpu())
            prop = outputs.max()

            if prop >= 0.95:
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                np.save('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/test_pseudo/' + test.split('/')[-1].split('.')[0] + '_' + str(idx / sr) + '_' + str(prop) + '_' + str(outputs) + '.npy', segment)

    print('finish')
