import os
from dataclasses import dataclass
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from skimage import color
from torch.utils.data import DataLoader, RandomSampler

from ..datasets.sun_rgbd import SunRGBDDataModule
from ..models_2d.segmentation_resunet_2d import ResUNet


@dataclass
class Hyperparameters:
    resize: Union[bool, float] = 0.5
    num_epochs: int = 20
    batch_size: int = 6
    num_workers: int = 0
    learning_rate: float = 0.0001

    # Will be set during initialization
    num_classes: int = None
    mask_shape = None
    rgbd_shape = None
    model_pred_shape = None
    model_pred_type = None


torch.set_float32_matmul_precision("high")


@dataclass
class Config:
    data_path: str = os.path.join(".data/nyu_depth_v2_labeled.mat")
    model_path: str = os.path.join(".models_2d")
    model_ident: str = "pResResUNet_50_sunrgbd"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    verbose: bool = True
    print_model: bool = False
    checkpoints_path: str = "model_checkpoints"
    show_all_nepochs: int = 1
    pin_memory: bool = False


class LitResUNet(pl.LightningModule):
    def __init__(self, **kwargs):
        super(LitResUNet, self).__init__()

        hp_kwargs = self.separate_kwargs(kwargs, Hyperparameters)
        config_kwargs = self.separate_kwargs(kwargs, Config)

        self.params = Hyperparameters(**hp_kwargs)
        self.config = Config(**config_kwargs)
        self.verbose = self.config.verbose

        ###<-------- SunRGBDDataModule -------->###True if self.device == "cuda" else False,
        self.data_module = SunRGBDDataModule(
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=self.config.pin_memory,
            verbose=self.verbose,
        )
        self.data_module.setup("init")
        self.params.num_classes = self.data_module.num_classes

        self.model = ResUNet(num_classes=self.params.num_classes).to(self.device)

        self._set_and_check_shapes()

        # Criterion, optimizer, checkpoint
        self.metric_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.params.num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_iou",
            dirpath=self.config.checkpoints_path,
            filename=self.config.model_ident + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
        )
        self.early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=4,
            verbose=True,
            mode="min",
        )

        if self.verbose:
            print(self)
        if self.config.print_model:
            print(self.model_summary())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        rgbds, masks = batch
        preds = self.model(rgbds)
        loss = self.criterion(preds, masks)
        iou = self.metric_iou(preds, masks)

        self.log_dict(
            {"train_iou": iou, "train_loss": loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage: str) -> None:
        self.data_module.setup(stage)

    def get_random_test_sample(self, stage="validate"):
        data_set = (
            self.data_module.val_dataset
            if stage == "validate"
            else self.data_module.test_dataset
        )

        random_sampler = RandomSampler(data_set, replacement=True, num_samples=1)
        rgbd, mask = next(
            iter(
                DataLoader(
                    dataset=data_set,
                    batch_size=1,
                    sampler=random_sampler,
                    num_workers=self.params.num_workers,
                )
            )
        )
        return rgbd, mask

    def validation_step(self, batch, batch_idx):
        rgbds, masks = batch
        preds = self.model(rgbds)
        loss = self.criterion(preds, masks)
        iou = self.metric_iou(preds, masks)
        self.log_dict(
            {"val_iou": iou, "val_loss": loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_iou": iou, "val_loss": loss}

    def on_validation_epoch_end(self):
        if self.current_epoch % self.config.show_all_nepochs == 0:
            fig = self.show_segmented(logging=True)
            self.logger.experiment.add_figure(
                "segmentation", fig, global_step=self.current_epoch
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def model_summary(self):
        return super().__repr__()

    def __repr__(self):
        return f"""
    <<<<<<<<<<<<<<<<<<<<PyTorch Lightning Model>>>>>>>>>>>>>>>>>>>>>>
    -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    {self.config}
    {self.params}
    model_pred shape: {self.params.model_pred_shape}, dtype: {self.params.model_pred_type}
    """

    def show_segmented(self, logging=False):
        """
        Display a random example of predicted segmentation and ground truth.
        """
        rgbd, mask = self.get_random_test_sample(stage="validate")

        self.model.eval()
        with torch.no_grad():
            pred = self.model(rgbd)
            pred = pred.cpu().squeeze().argmax(dim=0).numpy().astype(np.int16)

        mask = mask.cpu().squeeze().numpy().astype(np.int16)

        # print("pred.shape:", pred.shape)
        # print("mask.shape:", mask.shape)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharex=False, sharey=False, figsize=(18, 6)
        )
        fig.suptitle("Segmentation of a random sample")
        colors = list(self.data_module.class_cmap.values())
        # print("colors:", colors)
        ax1.imshow(color.label2rgb(pred, colors=colors))
        ax1.set_title("Predicted segmentation")
        ax1.set_axis_off()
        ax2.imshow(color.label2rgb(mask, colors=colors))
        ax2.set_title("Ground truth")
        ax2.set_axis_off()

        if logging:
            return fig
        plt.show()

    def _set_and_check_shapes(self):
        self.params.rgbd_shape, self.params.mask_shape = self.data_module.get_shapes()

        self.model.eval()
        with torch.no_grad():
            x = self.model.forward(
                torch.randn((1, *self.params.rgbd_shape)).to(self.device)
            )
            self.params.model_pred_shape = x.shape
            self.params.model_pred_type = x.dtype

        assert (
            self.params.mask_shape == self.params.model_pred_shape[2:]
        ), f"{self.params.mask_shape} != {self.params.model_pred_shape} "

    def separate_kwargs(self, kwargs, dataclass_instance):
        fields = {f for f in dataclass_instance.__dataclass_fields__}
        return {k: v for k, v in kwargs.items() if k in fields}


if __name__ == "__main__":
    model = LitResUNet()
    # trainer = pl.Trainer(
    #     logger=model.logger,
    #     max_epochs=model.params.num_epochs,
    #     fast_dev_run=True,
    #     log_every_n_steps=1,
    #     callbacks=[model.checkpoint_callback, model.early_stopping_callback],
    # )
    # trainer.fit(model)
