import numpy as np
import pickle
import torch
import torch.utils.data as data

from fairmotion.utils import constants


class Dataset(data.Dataset):
    def __init__(self, fp, device, mean=None, std=None):
        # Store parameters.
        self.fp = fp
        self.device = device

        # Load datasets into memory.
        src, tgt = torch.load(fp)

        # Dataset statistics.
        self.mean = mean if mean is not None else src.mean(dim=(0, 1))
        self.std = std if std is not None else src.std(dim=(0, 1))
        self.eps = constants.EPSILON

        # Save and send dataset to device.
        # ! Comment out device if dataset is large.
        self.src = ((src - self.mean) / (self.std + self.eps)).float().to(self.device)
        self.tgt = ((tgt - self.mean) / (self.std + self.eps)).float().to(self.device)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

    def __len__(self):
        return self.src.size(0)


def get_loader(
    fp,
    batch_size=100,
    device="cuda",
    mean=None,
    std=None,
    shuffle=False,
):
    return data.DataLoader(
        dataset=Dataset(fp, device, mean, std),
        batch_size=batch_size,
        shuffle=shuffle,
    )
