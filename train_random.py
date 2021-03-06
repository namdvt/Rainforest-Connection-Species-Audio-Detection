import torch
import torch.optim as optim
from utils.helper import write_log, write_figures
import numpy as np
import os
from dataset.sound_dataset_random import get_loader
from loss.bceloss import BCELoss

from model.model import Model
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score

import argparse
# CUDA_LAUNCH_BLOCKING="1"


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
        step = 1
    else:
        model.eval()
        step = 3

    running_loss = 0.0
    running_rank = 0.0

    for _ in range(step):
        for sound, target in tqdm(data_loader):
            sound = sound.to(device)
            target = target.to(device)

            if phase == 'training':
                optimizer.zero_grad()
                output = model(sound)
            else:
                with torch.no_grad():
                    output = model(sound)

            # loss
            loss = criterion(epoch, output, target)

            running_loss += loss.item()

            if phase == 'validation':
                rank = label_ranking_average_precision_score(target[:, 1:].detach().cpu().numpy().astype(int), output.detach().cpu().numpy())
                running_rank += rank

            if phase == 'training':
                loss.backward()
                optimizer.step()

    epoch_loss = running_loss / len(data_loader) / step

    if phase == 'training':
        print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
        return epoch_loss
    else:
        epoch_rank = running_rank / len(data_loader) / step
        print('[%d][%s] loss: %.4f rank: %.4f' % (epoch, phase, epoch_loss, epoch_rank))
        return epoch_loss, epoch_rank


def train(args):
    print('start training fold' + args.fold + ' using backbone ' + args.backbone)
    results = open('results.txt', 'a')
    device = args.device

    # create model
    model = Model(backbone=args.backbone).to(device)
    # model = ModelSincNet(sample_rate=48000).to(device)

    try:
        weight = 'output/14/' + args.backbone + '_' + args.fold + '.pth'
        model.load_state_dict(torch.load(weight, map_location=device), strict=True)
        print('loaded :' + weight)
    except:
        pass

    # train
    num_epochs = args.num_epochs

    train_loader = get_loader(annotation=args.tp_train, data_type='train', batch_size=args.batch_size, num_workers=args.num_workers, augment=False, prop_tp=args.prop_tp)
    val_loader = get_loader(annotation=args.tp_val, data_type='val', batch_size=args.batch_size, num_workers=args.num_workers, augment=False)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1)
    criterion = BCELoss(alpha=args.alpha)

    train_losses, val_losses, val_ranks = [], [], []
    num_not_improve = 0
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss, val_epoch_rank = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('---------------------------------------------------------- lr: ' + str(scheduler.optimizer.param_groups[0]['lr']))

        if epoch == 0 or (val_epoch_rank >= np.max(val_ranks) and val_epoch_loss <= np.min(val_losses)):
            torch.save(model.state_dict(), args.output_folder + args.backbone + '_' + args.fold + '.pth')
            num_not_improve = 0
        else:
            num_not_improve += 1

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_ranks.append(val_epoch_rank)

        if args.plot:
            write_figures('output', train_losses, val_losses)

        write_log(args.output_folder, epoch, train_epoch_loss, val_epoch_loss, val_epoch_rank)
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

    parser.add_argument('--fold', type=str, default='1')
    parser.add_argument('--tp_train', default='fold/1/train_list.txt')
    parser.add_argument('--tp_val', default='fold/1/val_list.txt')
    parser.add_argument('--fp_train', default='fp_train.txt')
    parser.add_argument('--fp_val', default='fp_val.txt')
    parser.add_argument('--prop_tp', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    # parser.add_argument('--beta', default=0, type=float, help='weight for fp loss')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--output_folder', default='output/', help='Location to save checkpoint models')
    parser.add_argument('--max_num_not_improve', type=int, default=35)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:1', help='device used for training')
    parser.add_argument('--backbone', type=str, default='selecsls42b')

    config = parser.parse_args()

    if not os.path.exists(config.output_folder):
        os.mkdir(config.output_folder)

    train(config)
