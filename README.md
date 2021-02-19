# Rainforest-Connection-Species-Audio-Detection
My solution for [Rainforest Connection Species Audio Detection Challenge](https://www.kaggle.com/c/rfcx-species-audio-detection/overview), which achieved rank 50 (top 5%) in private leaderboard.
Thank Quang, Tuan, Pikaman, your insights are valuable and I've learnt alot through this competition.
### Data Preprocessing
- Audio files are clipped to 10s-segment with stride=1
- Convert to mel-sepctrogram (sr=48000)
- Also use clipped data by t_max, t_min (TP and FP)
- Splitted to folds by file name, consider the number of samples in each class for every folds
### Model
- 2 best backbones are legacy_seresnet34 and selecsls42b
- Use maxpool, avgpool in time and frequency domain, respectively instead of stride
### Loss
- Use BCE loss for TP and l1 loss for FP
- Weight for BCE/l1 loss is 0.8/0.2
### Training
- Use bachsize=8, lr=0.01, CosineAnnealingWarmRestarts 30, 1
- No augmentation
- In every batches, randomly select 1 sample for 1 filename
- Probalibity of TP/FP in one batch is 0.6/0.4
### Inference
- Inference using stride=1, then max the results
- Compute gmean for outputs from different folds
### Not work
- Augmentation: mix, add noise, time stretch ...
- Add background
- Pseudo label
- Asymmetric loss, lsep loss
- 1D CNN
- Hard mining
### Must try
- Crop on frequency axis [discussion](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220304)
- Catastrophic forgetting [paper](https://arxiv.org/pdf/1612.00796.pdf)
- Fix overconfidence [paper](https://arxiv.org/pdf/2002.10118.pdf), [discussion](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220389)
