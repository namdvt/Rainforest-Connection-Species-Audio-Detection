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
    weights = glob('output/best_loss/*.pth')
    # weights = ['output/regnetx_064_1.pth', 'output/regnetx_064_2.pth']
    test_data = glob('data_waveform/test/waveform/*.npy')
    test_data.sort()
    final_result = list()
    submission_mean = pd.DataFrame(
        columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
    submission_gmean = pd.DataFrame(
        columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
    final_results = {}
    for weight in weights:
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()

        for address in tqdm(test_data):
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
                output = torch.max(output, dim=0)[0]
                torch.cuda.empty_cache()

            # get final results from all models
            file_name = address.split('/')[-1].split('.')[0]
            if file_name in final_results:
                final_results[file_name].append(output)
            else:
                final_results[file_name] = list()
                final_results[file_name].append(output)

    # mean
    for file_name in final_results.keys():
        result = torch.stack(final_results[file_name]).mean(dim=0).tolist()
        result.insert(0, file_name)
        submission_mean = submission_mean.append(pd.Series(result, index=submission_mean.columns), ignore_index=True)
    submission_mean.to_csv('submission/legacy_seresnet34_best_loss_mean.csv', index=False)

    # gmean
    for file_name in final_results.keys():
        result = gmean(torch.stack(final_results[file_name]).cpu(), axis=0).tolist()
        result.insert(0, file_name)
        submission_gmean = submission_gmean.append(pd.Series(result, index=submission_gmean.columns), ignore_index=True)
    submission_gmean.to_csv('submission/legacy_seresnet34_best_loss_gmean.csv', index=False)

    print()