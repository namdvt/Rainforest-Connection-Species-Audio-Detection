rm results.txt
python train.py --backbone legacy_seresnet34 --fold '1' --tp_train 'fold/1/train_filenames.txt' --tp_val 'fold/1/val_filenames.txt' --prop_tp 0.6 --alpha 0.8
python train.py --backbone legacy_seresnet34 --fold '2' --tp_train 'fold/2/train_filenames.txt' --tp_val 'fold/2/val_filenames.txt' --prop_tp 0.6 --alpha 0.8
python train.py --backbone legacy_seresnet34 --fold '3' --tp_train 'fold/3/train_filenames.txt' --tp_val 'fold/3/val_filenames.txt' --prop_tp 0.6 --alpha 0.8
python train.py --backbone legacy_seresnet34 --fold '4' --tp_train 'fold/4/train_filenames.txt' --tp_val 'fold/4/val_filenames.txt' --prop_tp 0.6 --alpha 0.8
python train.py --backbone legacy_seresnet34 --fold '5' --tp_train 'fold/5/train_filenames.txt' --tp_val 'fold/5/val_filenames.txt' --prop_tp 0.6 --alpha 0.8
# python test_kfold.py
python test_kfold_light.py