# Copyright © Scott Workman. 2025.
""" 
  Reference:

  https://github.com/lizuoyue/sate_to_ground
"""

import torch

import numpy as np


def generate_grid(h, w, dtype=torch.float32):
  x = torch.linspace(-1.0, 1.0, w)
  y = torch.linspace(-1.0, 1.0, h)
  xx, yy = torch.meshgrid(x, y, indexing='xy')
  grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).to(dtype)
  return grid


def depth2voxel(img_depth, gsd):
  device = img_depth.device
  dtype = img_depth.dtype

  n, c, h, w = img_depth.size()
  gsd = gsd if torch.is_tensor(gsd) else torch.tensor(gsd).to(device)

  gsize = (w * gsd).int()
  half_g = torch.true_divide(gsize, 2).int()

  site_z = img_depth[:, 0, int(h / 2), int(w / 2)] + 3.0
  voxel_sitez = site_z.view(n, 1, 1, 1).expand(n, gsize, gsize, gsize)

  # depth voxel
  grid_mask = generate_grid(gsize, gsize, dtype).to(device)
  grid_mask = grid_mask.expand(n, gsize, gsize, 2)
  grid_depth = torch.nn.functional.grid_sample(img_depth,
                                               grid_mask,
                                               align_corners=True)

  voxel_depth = grid_depth.expand(n, gsize, gsize, gsize)
  voxel_depth = voxel_depth - voxel_sitez

  # occupancy voxel
  voxel_grid = torch.arange(-half_g, half_g, 1).type(dtype).to(device)
  voxel_grid.requires_grad = True
  voxel_grid = voxel_grid.view(1, gsize, 1, 1).expand(n, gsize, gsize, gsize)
  voxel_ocupy = torch.ge(voxel_depth, voxel_grid).type(dtype)
  voxel_ocupy[:, gsize - 1, :, :] = 0

  # distance voxel
  voxel_dx = grid_mask[0, :, :, 0].view(1, 1, gsize, gsize).expand(
      n, gsize, gsize, gsize).type(dtype) * float(gsize / 2.0)
  voxel_dy = grid_mask[0, :, :, 1].view(1, 1, gsize, gsize).expand(
      n, gsize, gsize, gsize).type(dtype) * float(gsize / 2.0)
  voxel_dz = voxel_grid

  voxel_dis = voxel_dx.mul(voxel_dx) + voxel_dy.mul(voxel_dy) + voxel_dz.mul(
      voxel_dz)
  voxel_dis = voxel_dis.add(0.01)  # avoid 1/0 = nan
  voxel_dis = voxel_dis.mul(voxel_ocupy)
  voxel_dis = torch.sqrt(voxel_dis) - voxel_ocupy.add(-1.0).mul(
      float(gsize) * 0.9)

  return voxel_dis


def voxel2pano(voxel_dis, ori, size_pano):
  device = voxel_dis.device
  dtype = voxel_dis.dtype

  PI = np.pi
  r, c = [size_pano[0], size_pano[1]]
  n, s, t, tt = voxel_dis.size()
  k = int(s / 2)

  # rays
  ori = ori.view(n, 1).expand(n, c).type(dtype)
  x = torch.arange(0, c, 1).type(dtype).view(1, c).expand(n, c).to(device)
  y = torch.arange(0, r, 1).type(dtype).view(1, r).expand(n, r).to(device)
  lon = x * 2 * PI / c + ori - PI
  lat = PI / 2.0 - y * PI / r
  sin_lat = torch.sin(lat).view(n, 1, r, 1).expand(n, 1, r, c)
  cos_lat = torch.cos(lat).view(n, 1, r, 1).expand(n, 1, r, c)
  sin_lon = torch.sin(lon).view(n, 1, 1, c).expand(n, 1, r, c)
  cos_lon = torch.cos(lon).view(n, 1, 1, c).expand(n, 1, r, c)
  vx = cos_lat.mul(sin_lon)
  vy = -cos_lat.mul(cos_lon)
  vz = sin_lat
  vx = vx.expand(n, k, r, c)
  vy = vy.expand(n, k, r, c)
  vz = vz.expand(n, k, r, c)

  voxel_dis = voxel_dis.contiguous().view(1, n * s * s * s)

  # sample voxels along pano-rays
  d_samples = torch.arange(0, float(k), 1).view(1, k, 1, 1).expand(n, k, r,
                                                                   c).to(device)
  samples_x = vx.mul(d_samples).add(k).long()
  samples_y = vy.mul(d_samples).add(k).long()
  samples_z = vz.mul(d_samples).add(k).long()
  samples_n = torch.arange(0, n, 1).view(n, 1, 1, 1).expand(n, k, r,
                                                            c).long().to(device)
  samples_indices = samples_n.mul(s * s * s).add(samples_z.mul(s * s)).add(
      samples_y.mul(s)).add(samples_x)
  samples_indices = samples_indices.view(1, n * k * r * c)
  samples_indices = samples_indices[0, :].to(device)

  # get depth pano
  samples_depth = torch.index_select(voxel_dis, 1, samples_indices)
  samples_depth = samples_depth.view(n, k, r, c)
  min_depth = torch.min(samples_depth, 1)
  pano_depth = min_depth[0]
  pano_depth = pano_depth.view(n, 1, r, c)

  return pano_depth
