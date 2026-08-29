# DRAFT: Implementation of LeJEPA

### What this is
A from scratch implementation of [LeJEPA](https://arxiv.org/abs/2511.08544) trained on CIFAR-10: ViT encoder, SIGReg, linear probe evaluation. Everything written was from the paper without using reference implementations or tutorials.

### Setup
Dataset used was CIFAR-10 with a 45K/10K train/test split. ViT has 5 blocks, 192 dims, 6 heads and patch size 4. A batch size of 256 was used. AdamW peak 4e-4 with warmup + cosine, lambda=0.01 over 600 epochs.

The probe is a Linear(192, 10) with 20 epochs and a frozen encoder.

### Results

| encoder | probe acc (test) | internal head acc (test) | erank | \|cos\| | \|cos\| centred | std min / med / max |
|---|---|---|---|---|---|---|
| **LeJEPA (600 epochs)** | **71.85%** | - | **51.35** | **0.095** | **0.094** | **0.38 / 0.87 / 1.88** |

#### Baselines
| encoder | probe acc (test) | internal head acc (test) | erank | \|cos\| | \|cos\| centred | std min / med / max |
|---|---|---|---|---|---|---|
| Random init ViT | 33.51% | - | 3.79 | 0.482 | 0.521 | 0.17 / 0.54 / 1.82 |
| Supervised ViT | 67.27% | 65.16% | 15.80 | 0.304 | 0.288 | 0.26 / 0.49 / 1.90 |
| CNN (block 0) | - | 82.33% | - | - | - | - |
| LeJEPA (20 epochs) | 41.00% | - | 7.87 | 0.299 | ? | ? |

Within image token similarity: random init ViT 0.5671, supervised 0.4703.

### Ablations

| encoder | probe acc (test) | erank | \|cos\| |
|---|---|---|---|
| Prediction only | 19.71% | 1.02 | 0.841 |
| SIGReg only | 20.78% | 32 | 0.153 |
