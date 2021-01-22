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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
backbone = 'legacy_seresnet34'
sr = 48000
    
# model.eval()

print('inferencing model ' + backbone)

if __name__ == '__main__':
    model = Model(backbone=backbone).to(device)
    weights = glob('output/2/*.pth')
    # weights = ['output/regnetx_064_1.pth', 'output/regnetx_064_2.pth']
    test_data = glob('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/test/waveform/*.npy')
    test_data.sort()
    final_result = list()
    submission = pd.DataFrame(
        columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
    final_results = {}
    for weight in weights:
        fold = int(weight.split('.pth')[0].split('_')[-1]) - 1
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()

        for address in tqdm(test_data):
            sample_name = address.split('/')[-1].split('.')[0]
            if sample_name not in final_results:
                final_results[sample_name] = torch.zeros((5, 50, 24)).to(device)

            full_spec = np.load(address)
            full_spec = lib.feature.melspectrogram(full_spec, sr=sr)
            full_spec = lib.power_to_db(full_spec)

            batch = list()
            for idx in range(0, 4688, 94):
                spec = full_spec[:, idx:idx + 938]
                spec = torch.tensor(spec)
                spec = torch.stack((spec, spec, spec), dim=0).float().unsqueeze(0).to(device)
                spec = normalize(spec)
                batch.append(spec)
            batch = torch.cat(batch)

            outputs = list()
            with torch.no_grad():
                output = model(batch)
                final_results[sample_name][fold] = output
                # torch.cuda.empty_cache()

    # export to csv
    for sample_name in final_results.keys():
        sample_result = gmean(final_results[sample_name].cpu()).max(axis=0).tolist()
        sample_result.insert(0, sample_name)
        submission = submission.append(pd.Series(sample_result, index=submission.columns), ignore_index=True)
    submission.to_csv('submission/2_light.csv', index=False)

    print()
