# Implementation of LeJEPA

### baselines:
| encoder | probe acc (test) | internal head acc (test) | erank | \|cos\| | \|cos\| centred | std min / med / max |
|---|---|---|---|---|---|---|
| Random init ViT | 33.51% | 65.16% | 3.79 | 0.482 | 0.521 | 0.17 / 0.54 / 1.82 |
| Supervised ViT | 67.27% |  | 15.80 | 0.304 | 0.288 | 0.26 / 0.49 / 1.90 |
| CNN (block 0) | — | 82.33% | — | — | — | — |
| LeJEPA | ? | — | ? | ? | ? | ? |

Within image token similarity: random init ViT 0.5671, supervised 0.4703.

Conditions for all: CIFAR-10 45k/5k/10k, 20 epochs, no augmentation, AdamW lr=1e-3.

Probe: frozen encoder, cached features, Linear(192,10), test set.
