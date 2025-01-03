# Copyright © Scott Workman. 2025.

import torch
import torchmetrics
import lightning as L
from torch.utils.data import DataLoader

import loss
import models
from data import HCODataset, HCOPreDataset


class CVD(L.LightningModule):

  def __init__(self, **kwargs):
    super().__init__()

    self.save_hyperparameters()

    self.criterion = loss.Loss()
    self.net = models.build_model(self.hparams.method)

    self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
    self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

  def forward(self, inputs):
    out_overhead, out_ground = self.net(inputs)
    return out_overhead, out_ground

  def _compute_metrics(self, outputs, targets, mode):
    _, out_ground = outputs
    tar_ground, valid_ground, _, _ = targets
    valid_inds = torch.nonzero(valid_ground, as_tuple=True)
    rmse = eval(f"self.{mode}_rmse")
    rmse(out_ground.squeeze()[valid_inds], tar_ground[valid_inds])

  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets, self.current_epoch)

    self._compute_metrics(outputs, targets, "train")

    self.log_dict({
        "train_loss": loss,
        "train_rmse": self.train_rmse
    },
                  prog_bar=True,
                  on_step=True,
                  on_epoch=True,
                  logger=True,
                  sync_dist=True)

    return loss

  def validation_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.criterion(outputs, targets, self.current_epoch)

    self._compute_metrics(outputs, targets, "val")

    self.log_dict({
        "val_loss": loss,
        "val_rmse": self.val_rmse
    },
                  prog_bar=True,
                  on_step=False,
                  on_epoch=True,
                  logger=True,
                  sync_dist=True)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           patience=5)
    return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]

  def train_dataloader(self):
    if self.hparams.method == "refine":
      dataset = HCODataset('train', zoom=self.hparams.zoom)
    else:
      dataset = HCOPreDataset('train', zoom=self.hparams.zoom)
    return DataLoader(dataset,
                      batch_size=self.hparams.batch_size,
                      shuffle=True,
                      num_workers=32)

  def val_dataloader(self):
    if self.hparams.method == "refine":
      dataset = HCODataset('val', zoom=self.hparams.zoom)
    else:
      dataset = HCOPreDataset('val', zoom=self.hparams.zoom)
    return DataLoader(dataset,
                      batch_size=self.hparams.batch_size,
                      shuffle=False,
                      num_workers=32)


if __name__ == "__main__":
  m = CVD(**{"method": "refine_base", "batch_size": 1, "learning_rate": 1e-4})
  print(m)
