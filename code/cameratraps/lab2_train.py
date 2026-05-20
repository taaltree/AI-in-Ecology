"""
lab2_train.py

Day-vs-night classifier on Oregon Critters subset.
Modify per Lab 2 tasks (augmentation, best-epoch save, etc.).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_ROOT = "day_night"
BATCH     = 32
EPOCHS    = 5
LR        = 1e-4


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    print("Device:", device)

    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=tx)
    val_ds   = datasets.ImageFolder(f"{DATA_ROOT}/val",   transform=tx)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=LR)

    def run(loader, train):
        model.train(train)
        ls, n, c = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            c  += (logits.argmax(1) == y).sum().item()
            n  += y.size(0)
        return ls / n, c / n

    for ep in range(EPOCHS):
        tl, ta = run(train_dl, True)
        vl, va = run(val_dl, False)
        print(f"epoch {ep+1}: train_loss={tl:.3f} acc={ta:.3f}  "
              f"val_loss={vl:.3f} acc={va:.3f}")

    torch.save(model.state_dict(), "day_night.pt")
    print("Saved day_night.pt")


if __name__ == "__main__":
    main()
