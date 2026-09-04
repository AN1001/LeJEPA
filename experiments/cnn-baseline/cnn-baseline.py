"""
Supervised CNN baseline

Saves final model at experiments/cnn-baseline/

A full working example can be found at:
https://www.kaggle.com/code/arnavnagpure/lejepa-exp-cnn-baseline

Notes:
20 pt diff between train and val, overfitting occuring.
Should implement regularisation.

Improvements:
Augmentation (h-flip, slight crop) for more data
Weight decay, cosine LR, label smoothing
"""
from pathlib import Path

import torch

from lejepa.cnn import CNN
from lejepa.config import TrainConfig
from lejepa.data import get_data
from lejepa.metrics import evaluate
from lejepa.utils import batch

LR = 1e-3
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
EPOCHS = 20

out = Path(__file__).resolve().parent
config = TrainConfig(save_dir=str(out))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(config.seed)

ds = get_data(config.data_path, device)

model = CNN(ds["mean"], ds["std"]).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=LR)

n = len(ds["ytr"])


for epoch in range(1, EPOCHS + 1):
    model.train()
    correct = torch.zeros((), device=device)
    seen = 0

    for idx in batch(n, BATCH_SIZE, device=device):
        x, y = ds["xtr"][idx], ds["ytr"][idx]

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        correct += (y_pred.argmax(1) == y).sum()
        seen += len(idx)

    train_acc = (correct / seen).item()
    val_acc = evaluate(model, ds["xval"], ds["yval"], EVAL_BATCH_SIZE)

    print(f"E {epoch:<4}| train {train_acc:.4f} | val {val_acc:.4f}")


torch.save(model.state_dict(), out / "model.pt")

acc = evaluate(model, ds["xte"], ds["yte"], EVAL_BATCH_SIZE)
print(f"test accuracy: {acc:.4f}, val accuracy: {val_acc:.4f}")
