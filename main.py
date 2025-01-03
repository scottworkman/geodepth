# Copyright © Scott Workman. 2025.

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from cvd import CVD

import argparse

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", default=2, type=int)
  parser.add_argument('--batch_size', default=24, type=int)
  parser.add_argument('--save_dir', default='./logs/', type=str)
  parser.add_argument('--learning_rate', default=1e-4, type=float)
  parser.add_argument(
      '--method',
      default="refine_fuse",
      type=str,
      help="The method [refine_base, refine_fuse, refine, ground]")
  parser.add_argument('--zoom', default=16, type=int)
  parser.add_argument('--pretrain', default=None, type=str)
  parser.add_argument('--resume', default=None, type=str)
  args = parser.parse_args()

  L.seed_everything(args.seed, workers=True)

  if args.pretrain != None:
    model = CVD.load_from_checkpoint(args.pretrain, **vars(args), strict=False)
  else:
    model = CVD(**vars(args))

  checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                        mode="min",
                                        save_last=True)
  lr_monitor_callback = LearningRateMonitor(logging_interval='step')

  job_dir = "{}{}".format(args.save_dir, args.method)
  logger = TensorBoardLogger(job_dir)

  trainer = L.Trainer(accelerator="gpu",
                      devices=1,
                      max_epochs=25,
                      logger=logger,
                      num_sanity_val_steps=1,
                      default_root_dir=job_dir,
                      callbacks=[checkpoint_callback, lr_monitor_callback],
                      profiler="simple",
                      precision="16-mixed")
  trainer.fit(model, ckpt_path=args.resume)
