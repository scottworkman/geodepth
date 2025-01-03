# Copyright © Scott Workman. 2025.
""" 
  References:

  https://github.com/vispy/vispy/blob/main/vispy/util/transforms.py
  https://github.com/zhou13/holicity/blob/master/panorama2perspective.py
"""

import torch
import torch.nn.functional as F

import math
import numpy as np
import numpy.linalg as LA

# 90 degree fov fixed
p = np.eye(4)
p[3, 3] = 0
p[2, 2] = -1.00002
p[2, 3] = -1
p[3, 2] = -0.0200002


def rotate(angle, axis, dtype=None):
  """The 3x3 rotation matrix for rotation about a vector.
    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.
    Returns
    -------
    M : ndarray
        Transformation matrix describing the rotation.
    """
  angle = np.radians(angle)
  assert len(axis) == 3
  x, y, z = axis / np.linalg.norm(axis)
  c, s = math.cos(angle), math.sin(angle)
  cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
  M = np.array(
      [[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
       [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
       [cx * z - y * s, cy * z + x * s, cz * z + c, 0.], [0., 0., 0., 1.]],
      dtype).T
  return M


def lookat(position, forward, up=[0, 1, 0]):
  """Computes matrix to put camera looking at look point."""
  c = np.asarray(position).astype(float)
  w = -np.asarray(forward).astype(float)
  u = np.cross(up, w)
  v = np.cross(w, u)
  u /= LA.norm(u)
  v /= LA.norm(v)
  w /= LA.norm(w)
  return np.r_[u, u.dot(-c), v,
               v.dot(-c), w,
               w.dot(-c), 0, 0, 0, 1].reshape(4, 4).T


def get_mat(pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch):
  mat = rotate(pano_yaw, [0, 0, 1]) @ rotate(
      tilt_pitch,
      np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1]),
  ) @ lookat(
      [0, 0, 0],
      [np.cos(pitch) * np.cos(yaw),
       np.cos(pitch) * np.sin(yaw),
       np.sin(pitch)],
      [0, 0, 1],
  )
  return mat


def generate_cutout(im,
                    pano_yaw=0,
                    tilt_yaw=0,
                    tilt_pitch=0,
                    yaw=0,
                    pitch=0,
                    fov=90,
                    sz=(512, 512)):
  device = im.device
  dtype = im.dtype

  pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch, fov = [
      x if torch.is_tensor(x) else torch.tensor(x).type(dtype).to(device)
      for x in [pano_yaw, tilt_yaw, tilt_pitch, yaw, pitch, fov]
  ]

  _, iimh, iimw = im.shape  # input  image size
  oimh, oimw = sz  # output image size

  hfov = fov / 180 * np.pi  # horizontal field of view (radians)
  f = oimw / (2 * torch.tan(hfov / 2))  # focal length (pixels)

  ouc = (oimw + 1) / 2  # output image center
  ovc = (oimh + 1) / 2

  # tangent plane to unit sphere mapping
  X, Y = torch.meshgrid(torch.linspace(1, oimw, oimw).to(device) - 0.5,
                        torch.linspace(1, oimh, oimh).to(device) - 0.5,
                        indexing='ij')
  X = torch.transpose(X, 0, 1)  # match with numpy default
  Y = torch.transpose(Y, 0, 1)
  X = X - ouc
  Y = Y - ovc

  Z = -f.repeat(X.shape)
  PTS = torch.stack((X.flatten(), Y.flatten(), Z.flatten(),
                     torch.ones(Z.flatten().shape[0]).to(device)))

  yaw = -yaw * np.pi / 180 + np.pi / 2
  pitch = pitch * np.pi / 180

  t2n = lambda x: x.cpu().numpy()
  T = get_mat(t2n(pano_yaw), t2n(tilt_yaw), t2n(tilt_pitch), t2n(yaw),
              t2n(pitch))
  T = torch.from_numpy(T).type(dtype).to(device)

  T = torch.matmul(torch.from_numpy(p).type(dtype).to(device), T)
  PTSt = torch.matmul(T, PTS)

  Xt = torch.reshape(PTSt[0, :], (oimh, oimw))
  Yt = torch.reshape(PTSt[1, :], (oimh, oimw))
  Zt = torch.reshape(PTSt[2, :], (oimh, oimw))

  pitch = torch.atan(Zt / torch.sqrt(Xt**2 + Yt**2))
  yaw = torch.atan2(Xt, Yt)
  U = (yaw + np.pi) / (2 * np.pi)
  V = (pitch + np.pi / 2) / np.pi
  U = U * 2 - 1
  V = V * 2 - 1

  grid = torch.stack((U, V), dim=2).type(dtype).unsqueeze(0)
  oim = F.grid_sample(im.unsqueeze(0),
                      grid,
                      mode="bilinear",
                      align_corners=True)

  return torch.flip(oim, [2])
