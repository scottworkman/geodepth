# Copyright © Scott Workman. 2025.
""" 
  Precomputes geospatial context (in the form of a synthetic depth
  image) from ground-truth height maps. 
"""

import _init_paths

import torch
from torch.utils.data import ConcatDataset

from data import HCODataset
from nets.ops import generate_cutout
from nets.geo import depth2voxel, voxel2pano

import os
import errno
import numpy as np


def ensure_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


if __name__ == "__main__":
  zoom = 16
  dataset = ConcatDataset(
      [HCODataset("train", return_id=True),
       HCODataset("val", return_id=True)])
  dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=64)

  pano_size = [256, 512]
  out_dir = f"../holicity-overhead/intermediate/{zoom}/"
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
      print(batch_idx, len(dataloader))
      inputs, outputs, image_id = batch

      im_ground, im_overhead, depth_overhead, pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch, gsd = [
          x.to(device) for x in inputs
      ]

      b, _, h, _ = im_ground.shape

      # from depth to voxel
      voxel_depth = depth2voxel(depth_overhead.unsqueeze(1),
                                torch.unique(gsd).squeeze())

      # from voxel to pano
      orientations = torch.zeros(b).type(im_ground.dtype).to(im_ground.device)
      pano_depth = voxel2pano(voxel_depth, orientations, pano_size)

      # extract perspective cutouts
      pano_cutout = torch.cat([
          generate_cutout(pano_depth[idx, ...], yaw=yaw[idx], pitch=pitch[idx])
          for idx in range(b)
      ],
                              dim=0)

      # identify non-intersections
      pano_cutout[pano_cutout > (torch.unique(gsd) * h / 2 * .95)] = -1
      pano_cutout = pano_cutout.cpu().numpy()

      # save
      for idx in range(b):
        out_file = "{}{}.npy".format(out_dir, image_id[idx])
        if os.path.exists(out_file):
          continue

        ensure_dir(out_file)
        np.save(out_file, pano_cutout[idx, ...].squeeze())
