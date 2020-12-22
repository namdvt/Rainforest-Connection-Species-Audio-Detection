import torch
import torch.optim as optim
from utils.helper import write_log_embedding, write_figures
import numpy as np
import os
from dataset import get_loader_embedding
from bceloss import CosineMarginLoss
import torch.nn as nn

from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse


# CUDA_LAUNCH_BLOCKING="1"


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    for sound, target in tqdm(data_loader):
        sound = sound.to(device)
        target = target.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            output = model(sound, classify=False)
        else:
            with torch.no_grad():
                output = model(sound, classify=False)

        # loss
        loss = criterion(output, target)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)

    if phase == 'training':
        print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
        return epoch_loss

    else:
        print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
        return epoch_loss


def train(args):
    print('start training')

    device = args.device
    model = Model(backbone=args.backbone).to(device)
    # model.load_state_dict(torch.load('output/embedding/embedding_r180.pth', map_location=device), strict=True)

    num_epochs = args.num_epochs
    train_loader = get_loader_embedding(annotation=args.tp_train, batch_size=args.batch_size, num_workers=args.num_workers, augment=True)
    val_loader = get_loader_embedding(annotation=args.tp_val, batch_size=args.batch_size, num_workers=args.num_workers, augment=False)

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    optimizer = optim.SGD(pg0, lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,
    #                       weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1)
    criterion = CosineMarginLoss(embed_dim=512, num_classes=24).to(device)
    # criterion.load_state_dict(torch.load('output/embedding/cosine_loss0.pth', map_location=device), strict=True)

    train_losses, val_losses = [], []
    num_not_improve = 0
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('---------------------------------------------------------- lr: ' + str(
            scheduler.optimizer.param_groups[0]['lr']))

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), args.output_folder + 'embedding_r18' + args.fold + '.pth')
            torch.save(criterion.state_dict(), args.output_folder + 'cosine_loss' + args.fold + '.pth')
            num_not_improve = 0
        else:
            num_not_improve += 1

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_log_embedding(args.output_folder, epoch, train_epoch_loss, val_epoch_loss)
        # write_figures(args.output_folder, train_losses, val_losses)
        # scheduler.step(val_epoch_loss)
        scheduler.step()

        if num_not_improve == args.max_num_not_improve or epoch == num_epochs - 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rainforest training')

    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--tp_train', default='fold/1/train_filenames.txt')
    parser.add_argument('--tp_val', default='fold/1/val_filenames.txt')
    parser.add_argument('--fp_train', default='fp_train.txt')
    parser.add_argument('--fp_val', default='fp_val.txt')
    parser.add_argument('--prop_tp', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--output_folder', default='output/embedding/', help='Location to save checkpoint models')
    parser.add_argument('--max_num_not_improve', type=int, default=20)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0', help='device used for training')
    parser.add_argument('--backbone', type=str, default='resnet18')

    config = parser.parse_args()

    if not os.path.exists(config.output_folder):
        os.mkdir(config.output_folder)

    train(config)
