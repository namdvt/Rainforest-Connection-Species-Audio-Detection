from glob import glob
import random

if __name__ == '__main__':
    f_train = open('../train_tp_train.txt', 'w+')
    f_val = open('../train_tp_val.txt', 'w+')

    with open('../train_tp.txt') as f:
        content = f.readlines()
    random.shuffle(content)

    for line in content[0:int(len(content) * 0.8)]:
        f_train.write(line)

    for line in content[int(len(content) * 0.8):]:
        f_val.write(line)

    f_train.close()
    f_val.close()
    print()
