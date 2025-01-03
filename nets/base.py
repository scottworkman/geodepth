# Copyright © Scott Workman. 2025.

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class DecoderBlock(nn.Module):

  def __init__(self, in_channels, n_filters):
    super().__init__()

    # B, C, H, W -> B, C/4, H, W
    self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
    self.norm1 = nn.GroupNorm(16, in_channels // 4)
    self.relu1 = nn.ReLU(inplace=True)

    # B, C/4, H, W -> B, C/4, H, W
    self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                      in_channels // 4,
                                      3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
    self.norm2 = nn.GroupNorm(16, in_channels // 4)
    self.relu2 = nn.ReLU(inplace=True)

    # B, C/4, H, W -> B, C, H, W
    self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
    self.norm3 = nn.GroupNorm(16, n_filters)
    self.relu3 = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.deconv2(x)
    x = self.norm2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.norm3(x)
    x = self.relu3(x)
    return x


class LinkNet34(nn.Module):
  """
  Modified from: https://github.com/snakers4/spacenet-three
  """

  def __init__(self, num_outputs, num_channels=3):
    super().__init__()

    filters = [64, 128, 256, 512]
    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if num_channels == 3:
      self.firstconv = resnet.conv1
    else:
      self.firstconv = nn.Conv2d(num_channels,
                                 64,
                                 kernel_size=(7, 7),
                                 stride=(2, 2),
                                 padding=(3, 3))

    self.firstbn = resnet.bn1
    self.firstrelu = resnet.relu
    self.firstmaxpool = resnet.maxpool
    self.encoder1 = resnet.layer1
    self.encoder2 = resnet.layer2
    self.encoder3 = resnet.layer3
    self.encoder4 = resnet.layer4

    self.decoder4 = DecoderBlock(filters[3], filters[2])
    self.decoder3 = DecoderBlock(filters[2], filters[1])
    self.decoder2 = DecoderBlock(filters[1], filters[0])
    self.decoder1 = DecoderBlock(filters[0], filters[0])

    self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
    self.finalrelu1 = nn.ReLU(inplace=True)
    self.finalconv2 = nn.Conv2d(32, 32, 3)
    self.finalrelu2 = nn.ReLU(inplace=True)
    self.finalconv3 = nn.Conv2d(32, num_outputs, 2, padding=1)

  def forward(self, x):
    x = self.firstconv(x)
    x = self.firstbn(x)
    x = self.firstrelu(x)
    x = self.firstmaxpool(x)
    e1 = self.encoder1(x)
    e2 = self.encoder2(e1)
    e3 = self.encoder3(e2)
    e4 = self.encoder4(e3)

    d4 = self.decoder4(e4) + e3
    d3 = self.decoder3(d4) + e2
    d2 = self.decoder2(d3) + e1
    d1 = self.decoder1(d2)

    f1 = self.finaldeconv1(d1)
    f2 = self.finalrelu1(f1)
    f3 = self.finalconv2(f2)
    f4 = self.finalrelu2(f3)
    f5 = self.finalconv3(f4)

    return f5


class UpSample(nn.Sequential):

  def __init__(self, skip_input, output_features):
    super().__init__()
    self.convA = nn.Conv2d(skip_input,
                           output_features,
                           kernel_size=3,
                           stride=1,
                           padding=1)
    self.leakyreluA = nn.LeakyReLU(0.2)
    self.convB = nn.Conv2d(output_features,
                           output_features,
                           kernel_size=3,
                           stride=1,
                           padding=1)
    self.leakyreluB = nn.LeakyReLU(0.2)

  def forward(self, x, concat_with):
    up_x = F.interpolate(x,
                         size=[concat_with.size(2),
                               concat_with.size(3)],
                         mode='bilinear',
                         align_corners=True)
    return self.leakyreluB(
        self.convB(
            self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))


class Decoder(nn.Module):

  def __init__(self,
               num_outputs,
               num_features=2208,
               decoder_width=0.5,
               num_context=0):
    super().__init__()
    features = int(num_features * decoder_width)

    self.conv2 = nn.Conv2d(num_features + num_context,
                           features,
                           kernel_size=1,
                           stride=1,
                           padding=1)

    self.up1 = UpSample(skip_input=features // 1 + 384 + num_context,
                        output_features=features // 2)
    self.up2 = UpSample(skip_input=features // 2 + 192 + num_context,
                        output_features=features // 4)
    self.up3 = UpSample(skip_input=features // 4 + 96 + num_context,
                        output_features=features // 8)
    self.up4 = UpSample(skip_input=features // 8 + 96 + num_context,
                        output_features=features // 16)

    self.deconv1 = nn.ConvTranspose2d(features // 16,
                                      features // 16,
                                      3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)

    self.conv3 = nn.Conv2d(features // 16,
                           num_outputs,
                           kernel_size=3,
                           stride=1,
                           padding=1)

  def fuse(self, feat, context):
    if context is None:
      return feat
    else:
      context_resized = F.interpolate(context,
                                      size=[feat.size(2),
                                            feat.size(3)],
                                      mode='bilinear',
                                      align_corners=True)
      return torch.cat([feat, context_resized], dim=1)

  def forward(self, features, context=None):
    x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[
        4], features[6], features[8], features[11]

    x_d0 = self.conv2(self.fuse(x_block4, context))
    x_d1 = self.up1(self.fuse(x_d0, context), x_block3)
    x_d2 = self.up2(self.fuse(x_d1, context), x_block2)
    x_d3 = self.up3(self.fuse(x_d2, context), x_block1)
    x_d4 = self.up4(self.fuse(x_d3, context), x_block0)
    d1 = F.relu(self.deconv1(x_d4))
    return self.conv3(d1)


class Encoder(nn.Module):

  def __init__(self, num_channels=4):
    super().__init__()
    self.original_model = models.densenet161(
        weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    if num_channels != 3:
      self.original_model.features[0] = nn.Conv2d(num_channels,
                                                  96,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2),
                                                  padding=(3, 3),
                                                  bias=False)

  def forward(self, x):
    features = [x]
    for k, v in self.original_model.features._modules.items():
      features.append(v(features[-1]))
    return features


class DenseNet(nn.Module):
  """
  Modified from: https://github.com/ialhashim/DenseDepth
  """

  def __init__(self, num_outputs, num_channels=4):
    super().__init__()
    self.encoder = Encoder(num_channels)
    self.decoder = Decoder(num_outputs)

  def forward(self, x):
    return self.decoder(self.encoder(x))


class DenseNetFuse(nn.Module):
  """
  Modified from: https://github.com/ialhashim/DenseDepth
  """

  def __init__(self, num_outputs, num_context, num_channels=3):
    super().__init__()
    self.encoder = Encoder(num_channels)
    self.decoder = Decoder(num_outputs, num_context=num_context)

  def forward(self, x, context):
    return self.decoder(self.encoder(x), context)


if __name__ == "__main__":
  im = torch.randn([4, 4, 512, 512])
  context = torch.randn(4, 1, 512, 512)

  model = DenseNet(8)
  print(model(im).shape)

  model = DenseNetFuse(8, num_context=1)
  print(model(im[:, :3, ...], context).shape)
