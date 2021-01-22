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
import torch.nn as nn
from train_stack import StackModel
import time

from model import Model

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
sr = 48000
model_list = ['12', '14']
    
def load_model(path):
    models = list()
    for weight_path in glob('output/' + path + '/*.pth'):
        backbone = weight_path.split('/')[-1].replace('_' + weight_path.split('/')[-1].split('_')[-1], '')
        model = Model(backbone=backbone).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
        model.eval()
        print('loaded stack model from ' + weight_path)
        models.append(model)

    return models


def load_stack_model():
    models = list()
    for weight_path in glob('output/stack_' + '_'.join(model_list) + '/*.pth'):
        model = StackModel(num_models=len(model_list)).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
        model.eval()
        print('loaded stack model from ' + weight_path)
        models.append(model)

    return models


# def gmean(input_x, dim):
#     log_x = torch.log(input_x)
#     return torch.exp(torch.mean(log_x, dim=dim))


class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # load model
        self.model_1 = load_model(path='12')
        self.model_2 = load_model(path='14')
        # self.model_3 = load_model(path='16')

        # load stack model
        self.stack = load_stack_model()

    def forward_model(self, models, x):
        out = torch.stack((
            models[0](x, is_sigmoid=False),
            models[1](x, is_sigmoid=False),
            models[2](x, is_sigmoid=False),
            models[3](x, is_sigmoid=False),
            models[4](x, is_sigmoid=False)
        ))

        return out.mean(dim=0)

    def forward(self, x):
        feature_1 = self.forward_model(self.model_1, x)
        feature_2 = self.forward_model(self.model_2, x)
        # feature_3 = self.forward_model(self.model_3, x)

        feature_stacked = torch.stack((feature_1, feature_2)).permute(1,0,2)
        # feature_stacked = feature_stacked

        output = torch.stack((
            self.stack[0](feature_stacked.contiguous()),
            self.stack[1](feature_stacked.contiguous()),
            self.stack[2](feature_stacked.contiguous()),
            self.stack[3](feature_stacked.contiguous()),
            self.stack[4](feature_stacked.contiguous())
        ))

        output = torch.sigmoid(output).mean(0).max(0)[0]
        # output = gmean(torch.sigmoid(output).cpu().numpy(), axis=0).max(axis=0)

        return output


if __name__ == '__main__':
    # load weights
    ensemble_model = EnsembleModel().to(device)
    ensemble_model.eval()

    # load test data
    test_data = glob('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/test/waveform/*.npy')
    test_data.sort()
    final_result = list()

    submission = pd.DataFrame(
        columns=['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
    final_results = {}

    for path in tqdm(test_data):
        sample_name = path.split('/')[-1].split('.')[0]
        waveform = np.load(path)

        batch = list()
        for idx in range(0, 50*sr, 5*sr):
            segment = waveform[idx:idx + 10*sr]
            spec = lib.feature.melspectrogram(segment, sr=sr)
            spec = lib.power_to_db(spec)

            spec = torch.tensor(spec)
            spec = torch.stack((spec, spec, spec), dim=0).float().unsqueeze(0).to(device)
            spec = normalize(spec)

            with torch.no_grad():
                output = ensemble_model(spec)
            
            if np.asarray([1 if 0.05<=i<=0.9 else 0 for i in output]).sum() == 0 and output[output >= 0.9].shape[0] > 0:
                predicts = output.clone().cpu().numpy()
                predicts[predicts > 0.5] = 1
                predicts[predicts < 0.5] = 0

                np.save('/home/cybercore/oldhome/datasets/rain_forest/data_waveform/test_pseudo/' + sample_name.split('/')[-1].split('.')[0] + '_' + str(idx / sr) + '_' + str(predicts.astype(int)) + '.npy', segment)

            # print()

