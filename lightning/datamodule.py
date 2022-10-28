import os
from typing import Any, Dict, List, Optional, Tuple


import pytorch_lightning as pl
from omegaconf import DictConfig
from dataset import EmbeddedDataset
from torch.utils.data import DataLoader


class EmbeddedDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = EmbeddedDataset(
            **self.config.data.data_paths.train
        )
        self.val_dataset = EmbeddedDataset(
            **self.config.data.data_paths.val
        )

    @property
    def num_dataloader_workers(self) -> int:
        """Get the number of parallel workers in each dataloader."""
        if self.config.data.dataloader_workers >= 0:
            return self.config.data.dataloader_workers
        return os.cpu_count()
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
            shuffle=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
        )
        