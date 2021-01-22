rm results.txt
python train.py --backbone selecsls42b --fold '1' --tp_train 'fold/1/train_list.txt' --tp_val 'fold/1/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/21/'
python train.py --backbone selecsls42b --fold '2' --tp_train 'fold/2/train_list.txt' --tp_val 'fold/2/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/21/'
python train.py --backbone selecsls42b --fold '3' --tp_train 'fold/3/train_list.txt' --tp_val 'fold/3/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/21/'
python train.py --backbone selecsls42b --fold '4' --tp_train 'fold/4/train_list.txt' --tp_val 'fold/4/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/21/'
python train.py --backbone selecsls42b --fold '5' --tp_train 'fold/5/train_list.txt' --tp_val 'fold/5/val_list.txt' --prop_tp 0.6 --alpha 0.8 --output_folder 'output/21/'
python test_kfold.py
