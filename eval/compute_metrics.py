# Copyright © Scott Workman. 2025.

import _init_paths

import torch

import cvd
from data import HCODataset, HCOPreDataset

import argparse
import numpy as np
from tqdm import tqdm


def compute_errors(gt, pred):
  """
  Computation of error metrics between predicted and ground truth depths
  
  Reference: 

  https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
  """
  thresh = np.maximum((gt / pred), (pred / gt))
  a1 = (thresh < 1.25).mean()
  a2 = (thresh < 1.25**2).mean()
  a3 = (thresh < 1.25**3).mean()

  rmse = (gt - pred)**2
  rmse = np.sqrt(rmse.mean())

  rmse_log = (np.log(gt) - np.log(pred))**2
  rmse_log = np.sqrt(rmse_log.mean())

  abs_rel = np.mean(np.abs(gt - pred) / gt)

  sq_rel = np.mean(((gt - pred)**2) / gt)

  return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == "__main__":
  MIN_DEPTH = 1e-3
  MAX_DEPTH = 80

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--method',
      default="refine_fuse",
      type=str,
      help="The method [refine_base, refine_fuse, refine, ground]")
  parser.add_argument('--zoom', default=16, type=int)
  parser.add_argument('--median_scaling', action='store_true')
  parser.add_argument('--overhead_scaling', action='store_true')
  parser.add_argument('--save_dir', default='../logs/', type=str)
  parser.add_argument('--checkpoint', default="last.ckpt", type=str)
  args = parser.parse_args()

  job_dir = "{}{}".format(args.save_dir, args.method)

  if args.method == "refine":
    dataset = HCODataset("val", zoom=args.zoom)
  else:
    dataset = HCOPreDataset("val", zoom=args.zoom)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = cvd.CVD.load_from_checkpoint(
      '{}/lightning_logs/version_0/checkpoints/{}'.format(
          job_dir, args.checkpoint))
  model.to(device).float()
  model.eval()

  errors = []

  for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    inputs, targets = data
    label_ground, valid_ground, _, _ = [x.to(device) for x in targets]

    with torch.no_grad():
      _, output_ground = model([x.to(device).unsqueeze(0) for x in inputs])
      output_ground = output_ground.squeeze()

    mask = torch.logical_and(label_ground > MIN_DEPTH, label_ground < MAX_DEPTH)
    mask = torch.logical_and(mask, valid_ground)

    valid_inds = torch.nonzero(mask, as_tuple=True)
    output = output_ground[valid_inds]
    label = label_ground[valid_inds]

    if args.overhead_scaling:
      # perform median scaling using synthetic depth image
      assert args.method == "ground"
      _, depth_cutout = [x.to(device) for x in inputs]
      tmp = torch.logical_and(mask, depth_cutout != -1)
      if torch.sum(tmp) == 0:
        continue
      cutout = depth_cutout[tmp != 0]
      cutout[cutout < MIN_DEPTH] = MIN_DEPTH
      cutout[cutout > MAX_DEPTH] = MAX_DEPTH

      ratio = torch.median(cutout) / torch.median(output)
      output *= ratio

    if args.median_scaling:
      ratio = torch.median(label) / torch.median(output)
      output *= ratio

    output[output < MIN_DEPTH] = MIN_DEPTH
    output[output > MAX_DEPTH] = MAX_DEPTH

    if output.shape[0] == 0:
      continue

    errors.append(
        compute_errors(label.detach().cpu().numpy(),
                       output.detach().cpu().numpy()))

  mean_errors = np.array(errors).mean(0)

  print("\n  " +
        ("{:>8} | " *
         7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
  print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
  print()
