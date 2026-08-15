# CIFAR10 Supervised Baseline

82.3% test accuracy averaged over 5 runs (81.3, 82.2, 82.5, 82.7, 83.0) on Colab with a T4.

## Model
Architecture is a CNN with
```
[[conv(3x3) --> batchNorm --> ReLU]x2 --> Max Pool]
```
repeated 3 times followed by a global average pool and a fully connected layer. Channel widths go from 32 --> 64 --> 128. Convolutions use `bias=False` since batchNorm removes the effect of a bias anyway.

## Data
Data is 50K train and 10K test, taken from HuggingFace rather than UofT due to slow download speeds. Converted into PyTorch tensors then train split into 45,000 train / 5,000 val.

## Training
Trained with AdamW, constant learning rate (0.001), batch size 128, 20 epochs, cross-entropy loss.
The best validation checkpoint is kept, and the test set is evaluated exactly once from that checkpoint.

## Known limitations
Model is overfitting as seen from large gap between train and val (~98% vs ~80%). To alleviate add augmentation (light cropping and h-flips), cosine LR schedule, tuned weight decay and label smoothing. Runs also vary by about 0.6% with a fixed seed, likely because cuDNN is using non-deterministic algorithms.
