# Copyright © Scott Workman. 2025.

import torch
import torch.nn as nn


class Loss(nn.Module):

  def __init__(self):
    super().__init__()

  def pseudo_huber(self, output, target, valid, delta=2.0):
    valid_inds = torch.nonzero(valid, as_tuple=True)
    residual = target[valid_inds] - output[valid_inds]
    return torch.mean((delta**2) * (torch.sqrt(1 + (residual / delta)**2) - 1))

  def forward(self, outputs, targets, epoch):
    out_overhead, out_ground = outputs
    tar_ground, valid_ground, tar_overhead, valid_overhead = targets

    loss_overhead = loss_ground = 0
    weight_overhead = .1 if epoch < 5 else .01

    if out_overhead is not None:
      loss_overhead = self.pseudo_huber(out_overhead.squeeze(), tar_overhead,
                                        valid_overhead)

    if out_ground is not None:
      loss_ground = self.pseudo_huber(out_ground.squeeze(), tar_ground,
                                      valid_ground)

    loss = (weight_overhead * loss_overhead) + loss_ground

    return loss
