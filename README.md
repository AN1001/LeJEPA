# DRAFT: Implementation of LeJEPA

### What this is
A from scratch implementation of [LeJEPA](https://arxiv.org/abs/2511.08544) trained on CIFAR-10: ViT encoder, SIGReg, linear probe evaluation. Written from the paper without using reference implementations or tutorials.

### Setup
Dataset used was CIFAR-10 with a 45K/10K train/test split. ViT has 5 blocks, 192 dims, 6 heads and patch size 4. A batch size of 256 was used. AdamW peak 4e-4 with warmup + cosine, lambda=0.01 over 600 epochs.

The probe is a Linear(192, 10) with 20 epochs and a frozen encoder.

### Results

| encoder | probe acc (test) | internal head acc (test) | erank | \|cos\| | \|cos\| centred |
|---|---|---|---|---|---|---|
| **LeJEPA (600 epochs)** | **72.4%** | - | **42.9** | **0.1059** | **0.1061** |

#### Baselines
| encoder | probe acc (test) | internal head acc (test) | erank | \|cos\| | \|cos\| centred | std min / med / max |
|---|---|---|---|---|---|---|
| Random init ViT | 34.3% | - | 3.8 | 0.482 | 0.521 | 0.17 / 0.54 / 1.82 |
| Supervised ViT | 67.3% | 65.2% | 15.8 | 0.304 | 0.288 | 0.26 / 0.49 / 1.90 |
| CNN (block 0) | - | 80.0% | - | - | - | - |
| LeJEPA (20 epochs) | 41.00% | - | 7.87 | 0.299 | ? | ? |

Within image token similarity: random init ViT 0.5671, supervised 0.4703.
