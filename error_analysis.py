import torch
import torch.optim as optim
from utils.helper import write_log, write_figures
import numpy as np
import os
from dataset import get_loader
from bceloss import BCELoss
import cv2

from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
backbone = 'legacy_seresnet34'

model = Model(backbone=backbone).to(device)
model.load_state_dict(torch.load('output/r34_aux_0.pth', map_location=device))
model.eval()
    
if __name__ == '__main__':
    val_loader = get_loader(annotation='fold/3/val_filenames.txt', data_type='val', batch_size=1, num_workers=8, augment=False)
    bce = BCELoss()
    all_loss = list()
    idx = 0
    for a, sound, target in tqdm(val_loader):
        sound = sound.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(sound)

        loss = bce(output, target)
        if loss > 0.1:
            name = 'failure_cases/' + str(target.argmax().item()) + '_' + str(idx) + '.png'
            print(a)
        all_loss.append(loss.item())
        idx += 1

    print()

