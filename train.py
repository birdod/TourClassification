import argparse
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import EmbeddingModule, EmbeddedDataModule


def main(config: DictConfig):
    
    model_checkpoint = ModelCheckpoint(monitor="val/acc", save_weights_only=True)

    Trainer(
        accelerator='gpu',
        devices=2,
        strategy="ddp",
        logger=WandbLogger(project="TourCls_Linear"),
        callbacks=[model_checkpoint, LearningRateMonitor("step")],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=10,
    ).fit(EmbeddingModule(config), datamodule=EmbeddedDataModule(config))
    
    model = EmbeddingModule.load_from_checkpoint(
        model_checkpoint.last_model_path, config=config
    )
    torch.save(model.model.state_dict(), config.train.name + ".pth")


if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
