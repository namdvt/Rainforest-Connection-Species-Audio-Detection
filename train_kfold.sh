rm results.txt
python train_random.py --backbone legacy_seresnet34 --fold '1' --tp_train 'dataset/fold_split/split_1/1/train_list.txt' --tp_val 'dataset/fold_split/split_1/1/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/34/'
python train_random.py --backbone legacy_seresnet34 --fold '2' --tp_train 'dataset/fold_split/split_1/2/train_list.txt' --tp_val 'dataset/fold_split/split_1/2/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/34/'
python train_random.py --backbone legacy_seresnet34 --fold '3' --tp_train 'dataset/fold_split/split_1/3/train_list.txt' --tp_val 'dataset/fold_split/split_1/3/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/34/'
python train_random.py --backbone legacy_seresnet34 --fold '4' --tp_train 'dataset/fold_split/split_1/4/train_list.txt' --tp_val 'dataset/fold_split/split_1/4/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/34/'
python train_random.py --backbone legacy_seresnet34 --fold '5' --tp_train 'dataset/fold_split/split_1/5/train_list.txt' --tp_val 'dataset/fold_split/split_1/5/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/34/'
python test_kfold.py --experiment 34
