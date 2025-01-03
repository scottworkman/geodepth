# Copyright © Scott Workman. 2025.

import torch
import torch.nn as nn

import nets.base
from nets.ops import generate_cutout
from nets.geo import depth2voxel, voxel2pano


def build_model(method):
  print("[*] building model from {}".format(method))
  if method == "refine_base":
    model = RefineBase()
  elif method == "refine_fuse":
    model = RefineFuse()
  elif method == "refine":
    model = Refine()
  elif method == "ground":
    model = Ground()
  else:
    raise Exception('Method unrecognized.')

  return model


class RefineBase(nn.Module):
  """
  Concatenates geospatial context (in the form of a synthetic depth
  image) as an additional channel to the input image.
  """

  def __init__(self):
    super().__init__()

    self.net_ground = nets.base.DenseNet(1)
    self.pano_size = (256, 512)

  def forward(self, inputs):
    im_ground, im_depth_context = inputs

    # concatenate cutout with ground image
    fused = torch.cat((im_ground, im_depth_context), dim=1)

    depth = self.net_ground(fused)

    return None, depth


class RefineFuse(nn.Module):
  """
  Fuses geospatial context (in the form of a synthetic depth image)
  with image features inside the decoder.
  """

  def __init__(self):
    super().__init__()

    self.net_ground = nets.base.DenseNetFuse(1, num_context=1)
    self.pano_size = (256, 512)

  def forward(self, inputs):
    im_ground, im_depth_context = inputs

    depth = self.net_ground(im_ground, im_depth_context)

    return None, depth


class Refine(nn.Module):
  """
  RefineFuse + estimating the height maps. 
  """

  def __init__(self):
    super().__init__()

    self.net_ground = nets.base.DenseNetFuse(1, num_context=1)
    self.net_overhead = nets.base.LinkNet34(1)
    self.pano_size = (256, 512)

  def forward(self, inputs):
    im_ground, im_overhead, _, pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch, gsd = inputs

    b, _, h, _ = im_overhead.shape

    # estimate depth
    overhead_depth = self.net_overhead(im_overhead)

    # from depth to voxel
    voxel_depth = depth2voxel(overhead_depth, torch.unique(gsd).squeeze())

    # from voxel to pano
    orientations = torch.zeros(b).type(im_ground.dtype).to(im_ground.device)
    pano_depth = voxel2pano(voxel_depth, orientations, self.pano_size)

    # extract perspective cutouts
    pano_depth_cutout = torch.cat([
        generate_cutout(pano_depth[idx, ...], yaw=yaw[idx], pitch=pitch[idx])
        for idx in range(b)
    ],
                                  dim=0)

    # identify non-intersections
    pano_depth_cutout[pano_depth_cutout > (torch.unique(gsd) * h / 2 *
                                           .95)] = -1

    depth = self.net_ground(im_ground, pano_depth_cutout)

    return overhead_depth, depth


class Ground(nn.Module):
  """
  Baseline that omits geospatial context.
  """

  def __init__(self):
    super().__init__()

    self.net_ground = nets.base.DenseNet(1, num_channels=3)

  def forward(self, inputs):
    im_ground, _ = inputs
    return None, self.net_ground(im_ground)


if __name__ == "__main__":
  im = torch.randn([4, 3, 512, 512])
  im_context = torch.randn([4, 1, 512, 512])
  pano_yaw = tilt_yaw = tilt_pitch = yaw = pitch = torch.zeros([4])
  gsd = torch.ones([4]) * .372

  model = Refine()
  print(
      model([im, im, im, pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch,
             gsd])[-1].shape)

  for model in [RefineBase(), RefineFuse(), Ground()]:
    print(model([im, im_context])[-1].shape)
