from glob import glob
import random
import numpy as np
from tqdm import tqdm
import pandas as pd


def write_annotation():
    f_train = open('data_melspec/tp_train.txt', 'w+')
    list_sound = glob('data_melspec/tp_10s/*.npy')

    list_sound.sort()
    for image in list_sound:
        f_train.write(image + '\n')

    f_train.close()
    # f_val.close()


def write_annotation_all_tp():
    df = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')
    all_tp = open('data_melspec/all_tp.txt', 'w+')
    sound_list = df['recording_id'].unique()
    for name in sound_list:
        all_tp.write(name + '\n')



def write_annotation_species():
    f_train = open('data_melspec/data_filtered/species.txt', 'w+')
    all_background = glob('data_melspec/data_filtered/species/*.npy')
    all_background.sort()

    # get all train file name
    train_list = open('data_melspec/tp_train.txt').read().splitlines()
    train_filenames = set()
    for line in train_list:
        train_filenames.add(line.split('/')[-1].split('_')[0])

    # filter background
    for background in all_background:
        if background.split('/')[-1].split('_')[0] in train_filenames:
            f_train.write(background + '\n')

    f_train.close()


def select_fp_val():
    # get all file name from tp val list
    file_val_list = list()
    with open('tp_val.txt') as f:
        tp_val_lines = f.readlines()

    for line in tp_val_lines:
        file_name = line.split('/')[-1].split('_')[0]
        if not file_name in file_val_list:
            file_val_list.append(file_name)

    # get corresponding file name to fp val list
    fp_all = glob('/home/cybercore/nam/rainforest/data_melspec/train_fp/*.npy')
    fp_train = open('fp_train.txt', 'w+')
    fp_val = open('fp_val.txt', 'w+')

    for line in tqdm(fp_all):
        if line.split('/')[-1].split('_')[0] in file_val_list:
            fp_val.write(line + '\n')
        else:
            fp_train.write(line + '\n')


    print()


def write_file(location, all_list, val_list):
    f_val = open(location + '/val_filenames.txt', 'w+')
    f_train = open(location + '/train_filenames.txt', 'w+')
    for v in val_list:
        f_val.write(v + '\n')
    for v in all_list:
        if v not in val_list:
            f_train.write(v + '\n')


def write_kfold():
    tp = pd.read_csv('/home/cybercore/oldhome/datasets/rain_forest/train_tp.csv')
    all_list = set()
    for row in tqdm(tp.iterrows()):
        all_list.add(row[1]['recording_id'])

    all_list = list(all_list)
    random.shuffle(all_list)

    write_file('fold/1', all_list, all_list[0:226])
    write_file('fold/2', all_list, all_list[226:452])
    write_file('fold/3', all_list, all_list[452:678])
    write_file('fold/4', all_list, all_list[678:904])
    write_file('fold/5', all_list, all_list[904:1132])

if __name__ == '__main__':
    write_kfold()
