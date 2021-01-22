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
sr = 48000
    
num = '21'
# backbone = 'resnext50_32x4d'


if __name__ == '__main__':
    models = list()
    weights = glob('output/' + num + '/*.pth')
    backbone = weights[0].replace('_' + weights[0].split('/')[-1].split('_')[-1], '').split('/')[-1]
    print('inferencing model ' + backbone)

    for weight in weights:
        model = Model(backbone=backbone).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()
        models.append(model)
        print('loaded ' + weight)

    submission_gmean = pd.DataFrame(
        columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])


    test_data = glob('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/test/waveform/*.npy')
    test_data.sort()

    for address in tqdm(test_data):
        full_spec = np.load(address)
        full_spec = lib.feature.melspectrogram(full_spec, sr=sr)
        full_spec = lib.power_to_db(full_spec)

        batch = list()
        for idx in range(0, 4688, 94):
            spec = full_spec[:, idx:idx+938]
            spec = torch.tensor(spec)
            spec = torch.stack((spec, spec, spec), dim=0).float().unsqueeze(0).to(device)
            spec = normalize(spec)
            batch.append(spec)
        batch = torch.cat(batch)

        outputs = list()
        with torch.no_grad():
            for model in models:
                output = model(batch)
                outputs.append(output)
        outputs = torch.stack(outputs)

        # gmean
        result_gmean = gmean(outputs.cpu().numpy(), axis=0)
        result_gmean = result_gmean.max(axis=0).tolist()
        result_gmean.insert(0, address.split('/')[-1].split('.')[0])
        submission_gmean = submission_gmean.append(pd.Series(result_gmean, index=submission_gmean.columns), ignore_index=True)
    
    submission_gmean.to_csv('submission/' + num + '.csv', index=False)
    print('finish')
