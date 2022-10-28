from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

from modeling import EmbeddingClassifier, TwoLayerCls

class EmbeddingModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = TwoLayerCls(**config.model)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

    def training_step(
        self,
        emds: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(emds)
        loss = self.loss(outputs, labels)
        self.log("train/CE_loss", loss)
        return loss

    def validation_step(
        self,
        emds: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(emds)
        loss = self.loss(outputs, labels)
        
        _, pred = torch.max(outputs, 1)
        acc = torch.sum(pred==labels, dtype=torch.float32)\
            / emds.size(dim=0, dtype=torch.float32)

        self.log("val/CE_loss", loss)
        self.log("val/acc", acc)
        return acc

    def get_total_training_steps(self) -> int:
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def adjust_learning_rate(self, current_step: int) -> float:

        training_steps = self.get_total_training_steps()
        warmup_steps = int(training_steps * self.config.train.warmup_ratio)

        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0, (training_steps - current_step) / (training_steps - warmup_steps))

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(self.model.parameters(), **self.config.train.optimizer)
        scheduler = LambdaLR(optimizer, self.adjust_learning_rate)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]