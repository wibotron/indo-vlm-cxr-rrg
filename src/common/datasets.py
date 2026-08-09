"""
Shared dataset classes for the RRG pipeline (Pipeline Abstraction 001).

Reads the parquet files produced by `eda_and_preprocessing_chexpert-plus.ipynb`
(train_internal.parquet / dev_internal.parquet / test_official.parquet),
which already contain:
    - actual_image_path      : verified path to the PNG file on disk
    - <Pathology>_label      : 0.0/1.0 multilabel targets (Stage-B)
    - target_report_text     : "findings: ... impression: ..." (meeting-stage)
    - has_findings/has_impression, is_frontal : stratification flags

These classes are intentionally shared (`src/common/`) across experiments,
since the underlying data format is not expected to change between
experiment variants (e.g. exp_001 vs a future exp_002 ablation) — only the
model/training logic that consumes them changes.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_pathology_label_columns(df: pd.DataFrame, label_suffix: str = "_label") -> list:
    """Return pathology label columns in a stable, sorted order.

    The sort order matters: the classifier's output layer order must match
    this list exactly and consistently across train/dev/test and across
    training runs, or label-to-index alignment silently breaks.
    """
    cols = sorted(c for c in df.columns if c.endswith(label_suffix))
    if not cols:
        raise ValueError(
            f"No columns ending with '{label_suffix}' found in the given dataframe. "
            "Confirm this parquet file was produced by the build_pathology_labels "
            "step in the preprocessing notebook."
        )
    return cols


def compute_pos_weights(df: pd.DataFrame, label_cols: list) -> torch.Tensor:
    """Per-pathology pos_weight for nn.BCEWithLogitsLoss, from TRAIN prevalence only.

        pos_weight_i = num_negative_i / num_positive_i

    This upweights the loss contribution of positive examples for rare
    pathologies (e.g. Fracture, Pleural Other), preventing the classifier
    from collapsing to "always predict negative".

    IMPORTANT: compute this from the TRAIN split only. Computing it from
    dev/test would leak evaluation-set class balance into a training-time
    loss configuration.
    """
    pos_counts = (df[label_cols] == 1.0).sum()
    neg_counts = (df[label_cols] == 0.0).sum()
    pos_weight = (neg_counts / pos_counts.clip(lower=1)).to_numpy(dtype="float32")
    return torch.tensor(pos_weight)

class CXRPathologyDataset(Dataset):
    """Stage-B dataset: CXR image -> 14 pathology labels."""

    def __init__(self, parquet_path: str, label_cols: list, transform=None):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

        missing = [c for c in label_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"label columns missing from {parquet_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._load_image(row["actual_image_path"])
        labels = torch.tensor(row[self.label_cols].to_numpy(dtype="float32"))
        return image, labels

    def _load_image(self, image_path: str):
        if not Path(image_path).exists():
            # Fail loudly instead of silently skipping — a broken path here
            # means the preprocessing notebook's image-existence check was
            # skipped, or the dataset was moved without updating paths.
            raise FileNotFoundError(f"image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.transform(image) if self.transform is not None else image

class ReportGenerationDataset(Dataset):
    """Stage-A / meeting-stage dataset: CXR image -> target report text.

    Also exposes the pathology label columns, since the meeting-stage
    dataloader needs the same image tensor fed through BOTH the frozen
    Stage-A alignment pathway and the frozen Stage-B classifier pathway
    (see train_meeting_stage.py) within a single training step.
    """

    def __init__(self, parquet_path: str, label_cols: list, transform=None):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._load_image(row["actual_image_path"])
        target_text = row["target_report_text"]
        labels = torch.tensor(row[self.label_cols].to_numpy(dtype="float32"))
        return image, target_text, labels

    def _load_image(self, image_path: str):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.transform(image) if self.transform is not None else image