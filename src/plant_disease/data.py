from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class MetadataColumns:
    image_path: str = "image_path"
    split: str = "split"
    label_idx: str = "label_idx"


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Optional[str | Path],
        split: str,
        transform: Optional[Callable] = None,
        columns: MetadataColumns = MetadataColumns(),
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        if dataframe is not None:
            df = dataframe
        else:
            df = pd.read_csv(metadata_csv)
        self.df = df[df[columns.split].str.lower() == split.lower()].reset_index(drop=True)
        self.transform = transform
        self.columns = columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row[self.columns.image_path])
        label = int(row[self.columns.label_idx])

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
