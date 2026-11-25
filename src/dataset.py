"""Manifest-based dataset (columns: image, mask, label, split)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import load_gray, to_tensor

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mask"] = df["mask"].fillna("")
    for col in ("image", "mask"):  # resolve paths relative to the project root
        df[col] = [str(ROOT / v) if v and not Path(v).is_absolute() else v for v in df[col]]
    return df


class SEMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], size: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.idx = {c: i for i, c in enumerate(classes)}
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = to_tensor(load_gray(row["image"]), self.size)[0]
        if self.augment:  # cheap geometric augmentation (SEM defects are orientation-agnostic)
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.5:
                x = x.flip(1)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (1, 2))
        return x, self.idx[row["label"]]
