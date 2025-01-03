# Copyright © Scott Workman. 2025.

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import os
import imageio
import numpy as np
import pandas as pd
from pathlib import Path


class HCODataset(Dataset):
  """HoliCity-Overhead Dataset."""

  def __init__(self, mode='train', zoom=16, return_id=False):
    self.data_dir = f"{Path(os.path.abspath(__file__)).parent}/holicity-overhead/"
    self.mode = "valid" if mode == "val" else mode
    self.return_id = return_id

    if zoom == 17:
      self.gsd = .372
    elif zoom == 16:
      self.gsd = 0.7432
    elif zoom == 15:
      self.gsd = 1.486
    else:
      raise ValueError

    df_images = pd.read_csv("{}images.txt".format(self.data_dir),
                            names=["ground", "lat", "lon"])
    df_overhead = pd.read_csv("{}overhead/images_{}.txt".format(
        self.data_dir, zoom),
                              names=[
                                  "overhead", "lat", "lon", "min_lon",
                                  "min_lat", "max_lon", "max_lat"
                              ])
    df_combined = pd.concat((df_images, df_overhead), axis=1)

    # filter
    fnames_all = np.genfromtxt("{}/split/filelist.txt".format(self.data_dir),
                               dtype=str)
    fnames_split = np.genfromtxt("{}/split/{}-middlesplit.txt".format(
        self.data_dir, self.mode),
                                 dtype=str)

    length = len(fnames_split[0])
    fnames_split = set(fnames_split)
    fnames = [f for f in fnames_all if f[:length] in fnames_split]

    self.df = df_combined[df_combined["ground"].isin(fnames)]

  def __getitem__(self, idx):
    im = imageio.v2.imread("{}image/{}_imag.jpg".format(
        self.data_dir, self.df["ground"].iloc[idx])) / 255.
    depth = np.load("{}depth/{}_dpth.npz".format(
        self.data_dir, self.df["ground"].iloc[idx]))["depth"].squeeze()
    geo = np.load("{}geo/{}_camr.npz".format(self.data_dir,
                                             self.df["ground"].iloc[idx]))
    im_overhead = imageio.v2.imread("{}/overhead/{}".format(
        self.data_dir, self.df["overhead"].iloc[idx])) / 255.
    depth_overhead = np.load("{}/height/{}.npy".format(
        self.data_dir, self.df["overhead"].iloc[idx][:-4]))

    # extract calibration parameters
    pano_yaw = np.array(float(geo["pano_yaw"]))
    tilt_yaw = np.array(float(geo["tilt_yaw"]))
    tilt_pitch = np.array(float(geo["tilt_pitch"]))
    yaw = np.array(geo["yaw"])
    pitch = np.array(geo["pitch"])

    # identify valid depth regions
    with np.errstate(invalid='ignore'):
      valid = ~np.logical_or(depth == 0, np.isnan(depth))
      depth[valid == 0] = np.nan

    with np.errstate(invalid='ignore'):
      valid_overhead = ~np.logical_or(depth_overhead == -9999,
                                      np.isnan(depth_overhead))
      depth_overhead[valid_overhead == 0] = np.nan
      if np.any(valid_overhead):
        depth_overhead = depth_overhead - np.nanmin(depth_overhead) + 1e-8

    t_im_ground = TF.to_tensor(im).float()
    t_label_ground = torch.from_numpy(depth).float()
    t_valid_ground = torch.from_numpy(valid).long()

    t_im_overhead = TF.to_tensor(im_overhead).float()
    t_label_overhead = torch.from_numpy(depth_overhead).float()
    t_valid_overhead = torch.from_numpy(valid_overhead).long()

    t_pano_yaw = torch.from_numpy(pano_yaw).float()
    t_tilt_yaw = torch.from_numpy(tilt_yaw).float()
    t_tilt_pitch = torch.from_numpy(tilt_pitch).float()
    t_yaw = torch.from_numpy(yaw).float()
    t_pitch = torch.from_numpy(pitch).float()
    t_gsd = torch.from_numpy(np.array(self.gsd)).float()

    inputs = [
        t_im_ground, t_im_overhead, t_label_overhead, t_pano_yaw, t_tilt_yaw,
        t_tilt_pitch, t_yaw, t_pitch, t_gsd
    ]
    targets = [
        t_label_ground, t_valid_ground, t_label_overhead, t_valid_overhead
    ]

    if self.return_id:
      return inputs, targets, self.df["ground"].iloc[idx]
    else:
      return inputs, targets

  def __len__(self):
    return len(self.df)


class HCOPreDataset(Dataset):
  """HoliCity-Overhead Dataset with precomputed depth cutouts."""

  def __init__(self, mode='train', zoom=16):
    self.data_dir = f"{Path(os.path.abspath(__file__)).parent}/holicity-overhead/"
    self.intermediate_dir = f"{self.data_dir}intermediate/{zoom}/"
    self.mode = "valid" if mode == "val" else mode

    assert os.path.isdir(
        f"{self.intermediate_dir}"
    ), f"Intermediate directory '{self.intermediate_dir}' does not exist!"

    df = pd.read_csv("{}images.txt".format(self.data_dir),
                     names=["ground", "lat", "lon"])

    # filter
    fnames_all = np.genfromtxt("{}/split/filelist.txt".format(self.data_dir),
                               dtype=str)
    fnames_split = np.genfromtxt("{}/split/{}-middlesplit.txt".format(
        self.data_dir, self.mode),
                                 dtype=str)

    length = len(fnames_split[0])
    fnames_split = set(fnames_split)
    fnames = [f for f in fnames_all if f[:length] in fnames_split]

    self.df = df[df["ground"].isin(fnames)]

  def __getitem__(self, idx):
    im = imageio.v2.imread("{}image/{}_imag.jpg".format(
        self.data_dir, self.df["ground"].iloc[idx])) / 255.
    depth = np.load("{}depth/{}_dpth.npz".format(
        self.data_dir, self.df["ground"].iloc[idx]))["depth"].squeeze()
    depth_cutout = np.load("{}{}.npy".format(
        self.intermediate_dir, self.df["ground"].iloc[idx])).squeeze()

    # identify valid depth regions
    with np.errstate(invalid='ignore'):
      valid = ~np.logical_or(depth == 0, np.isnan(depth))
      depth[valid == 0] = np.nan

    t_im_ground = TF.to_tensor(im).float()
    t_label_ground = torch.from_numpy(depth).float()
    t_valid_ground = torch.from_numpy(valid).long()
    t_depth_cutout = torch.from_numpy(depth_cutout).float().unsqueeze(0)

    inputs = [t_im_ground, t_depth_cutout]
    targets = [t_label_ground, t_valid_ground, torch.empty(1), torch.empty(1)]

    return inputs, targets

  def __len__(self):
    return len(self.df)


if __name__ == "__main__":
  take = 8

  dataset = HCODataset('train')

  for n, data in zip(range(take), dataset):
    inputs, targets = [[y.numpy() for y in x] for x in data]
    im_ground, im_overhead, _, pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch, gsd = inputs
    label_ground, valid_ground, label_overhead, valid_overhead = targets
    print(n, im_overhead.shape, label_ground.shape, valid_ground.shape,
          np.nanquantile(label_ground[valid_ground == 1], [0, 1]),
          np.quantile(valid_ground, [0, 1]), yaw, pitch)

  dataset = HCOPreDataset('train')

  for n, data in zip(range(take), dataset):
    inputs, targets = [[y.numpy() for y in x] for x in data]
    im_ground, depth_cutout = inputs
    label_ground, valid_ground, _, _ = targets
    print(n, im_ground.shape, label_ground.shape, valid_ground.shape,
          depth_cutout.shape,
          np.nanquantile(label_ground[valid_ground == 1], [0, 1]),
          np.quantile(valid_ground, [0, 1]))
