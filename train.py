import torch
import torch.optim as optim
from utils.helper import write_log, write_figures
import numpy as np
import os
from dataset import get_loader
from bceloss import BCELoss
import torch.nn as nn

from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score

import argparse
# CUDA_LAUNCH_BLOCKING="1"


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_rank = 0.0

    for sound, target in tqdm(data_loader):
        sound = sound.to(device)
        target = target.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            output = model(sound, classify=True)
        else:
            with torch.no_grad():
                output = model(sound, classify=True)

        # loss
        loss = criterion(epoch, output, target)
        if torch.isnan(loss):
            print('nan')
            continue

        running_loss += loss.item()

        if phase == 'validation':
            rank = label_ranking_average_precision_score(target[:, 1:].detach().cpu().numpy().astype(int), output.detach().cpu().numpy())
            # rank = label_ranking_average_precision_score(target.detach().cpu().numpy().astype(int), output.detach().cpu().numpy())
            running_rank += rank

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)

    if phase == 'training':
        print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
        return epoch_loss
    else:
        epoch_rank = running_rank / len(data_loader)
        print('[%d][%s] loss: %.4f rank: %.4f' % (epoch, phase, epoch_loss, epoch_rank))
        return epoch_loss, epoch_rank


def train(args):
    print('start training fold' + args.fold)
    results = open('results.txt', 'a')

    device = args.device
    model = Model(backbone=args.backbone).to(device)
    # model.load_state_dict(torch.load('output/embedding/bb_0.pth', map_location=device), strict=False)

    num_epochs = args.num_epochs

    train_loader = get_loader(annotation=args.tp_train, data_type='train', batch_size=args.batch_size, num_workers=args.num_workers, augment=True, prop_tp=args.prop_tp)
    val_loader = get_loader(annotation=args.tp_val, data_type='val', batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
 
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
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1)
    criterion = BCELoss(alpha=args.alpha)

    train_losses, val_losses, val_ranks = [], [], []
    num_not_improve = 0
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss, val_epoch_rank = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('---------------------------------------------------------- lr: ' + str(scheduler.optimizer.param_groups[0]['lr']))

        if epoch == 0 or val_epoch_rank >= np.max(val_ranks):
            # torch.save(model.state_dict(), args.output_folder + '/best_acc/r18_' + args.fold + '.pth')
            num_not_improve = 0
        else:
            num_not_improve += 1

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), args.output_folder + '/best_loss/r35_' + args.fold + '.pth')
            num_not_improve = 0

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_ranks.append(val_epoch_rank)

        if args.plot:
            write_figures('output', train_losses, val_losses)

        write_log(args.output_folder, epoch, train_epoch_loss, val_epoch_loss, val_epoch_rank)
        # scheduler.step(val_epoch_rank)
        scheduler.step()

        if num_not_improve == args.max_num_not_improve or epoch == num_epochs-1:
            print('finish training')
            print('train loss: ' + str(np.min(train_losses)))
            print('val loss: ' + str(np.min(val_losses)) + ' val rank: ' + str(np.max(val_ranks)))
            results.write(args.backbone + ' ' + args.fold + ' ' + str(args.prop_tp) + ' ' + str(args.alpha) + ' : ' + str(np.min(val_losses)) + ' ' + str(np.max(val_ranks)) + '\n')
            results.close()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rainforest training')

    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--tp_train', default='fold/3/train_filenames.txt')
    parser.add_argument('--tp_val', default='fold/3/val_filenames.txt')
    parser.add_argument('--fp_train', default='fp_train.txt')
    parser.add_argument('--fp_val', default='fp_val.txt')
    parser.add_argument('--prop_tp', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--output_folder', default='output/', help='Location to save checkpoint models')
    parser.add_argument('--max_num_not_improve', type=int, default=35)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0', help='device used for training')
    parser.add_argument('--backbone', type=str, default='legacy_seresnet34')

    config = parser.parse_args()

    if not os.path.exists(config.output_folder):
        os.mkdir(config.output_folder)

    train(config)
