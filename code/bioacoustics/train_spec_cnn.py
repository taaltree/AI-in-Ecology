"""
train_spec_cnn.py

Train a small ResNet-18 on mel-spectrograms of labeled audio clips.

Expected layout:
    clips/train/<class>/*.wav
    clips/val/<class>/*.wav
"""
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models


class SpectrogramDataset(Dataset):
    def __init__(self, root, sr=22050, n_mels=128, duration=5.0):
        self.classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        self.items   = []
        for label, cls in enumerate(self.classes):
            for wav in (Path(root) / cls).glob("*.wav"):
                self.items.append((wav, label))
        self.sr, self.n_mels, self.duration = sr, n_mels, duration

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav, label = self.items[idx]
        y, _ = librosa.load(wav, sr=self.sr, mono=True, duration=self.duration)
        target = int(self.sr * self.duration)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
        img = np.stack([img] * 3, axis=0).astype(np.float32)
        return torch.from_numpy(img), label


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    device = pick_device()
    print("Device:", device)

    train_ds = SpectrogramDataset("clips/train")
    val_ds   = SpectrogramDataset("clips/val")
    print("Classes:", train_ds.classes)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=3e-4)

    best = 0.0
    for epoch in range(10):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        c, n = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                c += (model(x).argmax(1) == y).sum().item()
                n += y.size(0)
        acc = c / n
        print(f"epoch {epoch+1}: val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "spec_cnn.pt")
            print(f"  saved (best so far)")

    print(f"Best val acc: {best:.3f}; classes = {train_ds.classes}")


if __name__ == "__main__":
    main()
