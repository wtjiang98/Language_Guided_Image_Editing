# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, MyIN
import numpy as np

def get_pad(in_, ksize, stride, atrous=1):
  out_ = np.ceil(float(in_) / stride)
  return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class SNGatedConv2dSPADEWithActivation(nn.Module):
  """
  Gated Convolution with spetral normalization
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
               batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True), opt=None):
    super(SNGatedConv2dSPADEWithActivation, self).__init__()
    self.opt = opt
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.activation = activation
    # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
    self.sigmoid = torch.nn.Sigmoid()
    self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
    self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)

    assert 'spectral' in opt.norm_G, "you are using SNGatedConv2d"
    spade_config_str = opt.norm_G.replace('spectral', '')
    self.spade = SPADE(spade_config_str, out_channels, opt.lang_dim, opt)

    # todo: 是否要init？
    # for m in self.modules():
    #   if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight)

  def gated(self, mask):
    return self.sigmoid(mask)
    # return torch.clamp(mask, -1, 1)

  def forward(self, input, condition):
    x = self.conv2d(input)
    mask = self.mask_conv2d(input)
    if self.activation is not None:
      x = self.activation(x) * self.gated(mask)
    else:
      x = x * self.gated(mask)

    # spade
    x = self.spade(x, condition)
    if self.opt.spade_attn:
      return x[0], x[1]
    else:
      return x



class SNGatedConv2dWithActivationV0(torch.nn.Module):
  """
  Gated Convolution with spetral normalization
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
               batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(SNGatedConv2dWithActivationV0, self).__init__()
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.activation = activation
    self.batch_norm = batch_norm
    self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
    self.sigmoid = torch.nn.Sigmoid()
    self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
    self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)
    # todo: 是否要init？
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

  def gated(self, mask):
    return self.sigmoid(mask)
    # return torch.clamp(mask, -1, 1)

  def forward(self, input):
    x = self.conv2d(input)
    mask = self.mask_conv2d(input)
    if self.activation is not None:
      x = self.activation(x) * self.gated(mask)
    else:
      x = x * self.gated(mask)
    if self.batch_norm:
      return self.batch_norm2d(x)
    else:
      return x


class SNGatedConv2dWithActivation(nn.Module):
  """
  Gated Convolution with spetral normalization
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
               norm_config='instance', activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(SNGatedConv2dWithActivation, self).__init__()
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.activation = activation
    # self.batch_norm = batch_norm
    # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
    self.sigmoid = torch.nn.Sigmoid()
    self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
    self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)

    if 'batch' in norm_config:
      self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
    elif 'instance' in norm_config:
      self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
    else:
      self.norm_layer = None

    # # todo: 是否要init？
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

  def gated(self, mask):
    return self.sigmoid(mask)
    # return torch.clamp(mask, -1, 1)

  def forward(self, input):
    x = self.conv2d(input)
    mask = self.mask_conv2d(input)
    if self.activation is not None:
      x = self.activation(x) * self.gated(mask)
    else:
      x = x * self.gated(mask)
    if self.norm_layer is not None:
      return self.norm_layer(x)
    # if self.batch_norm2d is not None:
    #   return self.batch_norm2d(x)
    else:
      return x



class SNGatedDeConv2dWithActivationV0(torch.nn.Module):
  """
  Gated DeConvlution layer with activation (default activation:LeakyReLU)
  resize + conv
  Params: same as conv2d
  Input: The feature from last layer "I"
  Output:\phi(f(I))*\sigmoid(g(I))
  """

  def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(SNGatedDeConv2dWithActivationV0, self).__init__()
    self.conv2d = SNGatedConv2dWithActivationV0(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias, batch_norm, activation)
    self.scale_factor = scale_factor

  def forward(self, input):
    # print(input.size())
    x = F.interpolate(input, scale_factor=2)
    return self.conv2d(x)




class SNGatedDeConv2dWithActivation(torch.nn.Module):
  """
  Gated DeConvlution layer with activation (default activation:LeakyReLU)
  resize + conv
  Params: same as conv2d
  Input: The feature from last layer "I"
  Output:\phi(f(I))*\sigmoid(g(I))
  """

  def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               bias=True, norm_config='instance', activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(SNGatedDeConv2dWithActivation, self).__init__()
    self.conv2d = SNGatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias, norm_config, activation)
    self.scale_factor = scale_factor

  def forward(self, input):
    # print(input.size())
    x = F.interpolate(input, scale_factor=2)
    return self.conv2d(x)


class Self_Attn(nn.Module):
  """ Self attention Layer"""

  def __init__(self, in_dim, activation, with_attn=False):
    super(Self_Attn, self).__init__()
    self.chanel_in = in_dim
    self.activation = activation
    self.with_attn = with_attn
    self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
    self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
    self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros(1))

    self.softmax = nn.Softmax(dim=-1)  #

  def forward(self, x):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
    """
    m_batchsize, C, width, height = x.size()
    proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
    proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
    energy = torch.bmm(proj_query, proj_key)  # transpose check
    attention = self.softmax(energy)  # BX (N) X (N)
    proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, width, height)

    out = self.gamma * out + x
    if self.with_attn:
      return out, attention
    else:
      return out


class GatedConv2dWithActivation(nn.Module):
  """
  Gated Convlution layer to replace Conv2d
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
               activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(GatedConv2dWithActivation, self).__init__()
    self.activation = activation
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.sigmoid = torch.nn.Sigmoid()

    # for m in self.modules():
    #   if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight)

  def gated(self, mask):
    # return torch.clamp(mask, -1, 1)
    return self.sigmoid(mask)

  def forward(self, input):
    x = self.conv2d(input)
    mask = self.mask_conv2d(input)
    if self.activation is not None:
      x = self.activation(x) * self.gated(mask)
    else:
      x = x * self.gated(mask)

    return x


class SPADEGatedResnetBlock(nn.Module):
  def __init__(self, fin, fout, opt, spade_condi_nc=None):
    super().__init__()
    # Attributes
    self.learned_shortcut = (fin != fout)
    self.opt = opt
    if opt.lang_dim != 0:
      if opt.use_op:
        self.spade_condi_nc = opt.lang_dim + opt.semantic_nc
      elif opt.FiveK:
        self.spade_condi_nc = opt.lang_dim
      elif opt.lang_encoder == 'bilstm':
        self.spade_condi_nc = opt.lang_dim

    if spade_condi_nc is not None:
      self.spade_condi_nc = spade_condi_nc

    fmiddle = min(fin, fout)

    # create conv layers
    # self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
    # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
    # if self.learned_shortcut:
    #   self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

    if 'spectral' in opt.norm_G:
      # 因为这里的conv后面接的是SPADE，所以不用norm
      self.conv_0 = SNGatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, norm_type=None)
      self.conv_1 = SNGatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, norm_type=None)
      if self.learned_shortcut:
        self.conv_s = SNGatedConv2dWithActivation(fin, fout, kernel_size=1, bias=False, norm_type=None)

    # apply spectral norm if specified
    # if 'spectral' in opt.norm_G:
    #   self.conv_0 = spectral_norm(self.conv_0)
    #   self.conv_1 = spectral_norm(self.conv_1)
    #   if self.learned_shortcut:
    #     self.conv_s = spectral_norm(self.conv_s)

    # define normalization layers
    spade_config_str = opt.norm_G.replace('spectral', '')
    if opt.spade_fusion:
      self.norm_0 = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
      self.norm_1 = SPADE(spade_config_str, fmiddle, self.spade_condi_nc, opt)
      if self.learned_shortcut:
        self.norm_s = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
    else:
      self.norm_0 = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
      self.norm_1 = SPADE(spade_config_str, fmiddle, self.spade_condi_nc, opt)
      if self.learned_shortcut:
        self.norm_s = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)

  # note the resnet block with SPADE also takes in |seg|,
  # the semantic segmentation map as input
  def forward(self, x, seg=None):
    attn_list = []
    x_s = self.shortcut(x, seg)

    norm_0 = self.norm_0(x, seg)
    if self.opt.spade_attn:
      attn_list.append(norm_0[1])
      norm_0 = norm_0[0]
    dx = self.conv_0(self.actvn(norm_0))

    norm_1 = self.norm_1(dx, seg)
    if self.opt.spade_attn:
      attn_list.append(norm_1[1])
      norm_1 = norm_1[0]
    dx = self.conv_1(self.actvn(norm_1))

    out = x_s + dx

    if self.opt.spade_attn:
      return out, attn_list
    else:
      return out

  def shortcut(self, x, seg):
    if self.learned_shortcut:
      norm_s = self.norm_s(x, seg)
      if self.opt.spade_attn:
        norm_s = norm_s[0]
      x_s = self.conv_s(norm_s)
    else:
      x_s = x
    return x_s

  def actvn(self, x):
    return F.leaky_relu(x, 2e-1)



# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
  def __init__(self, fin, fout, opt, spade_condi_nc=None):
    super().__init__()
    # Attributes
    self.opt = opt
    self.learned_shortcut = (fin != fout)
    if opt.lang_dim != 0:
      if opt.use_op:
        self.spade_condi_nc = opt.lang_dim + opt.semantic_nc
      elif opt.FiveK:
        self.spade_condi_nc = opt.lang_dim
      elif opt.lang_encoder == 'bilstm':
        self.spade_condi_nc = opt.lang_dim

    if spade_condi_nc is not None:
      self.spade_condi_nc = spade_condi_nc

    fmiddle = min(fin, fout)

    # create conv layers
    self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
    self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
    if self.learned_shortcut:
      self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

    # apply spectral norm if specified
    if 'spectral' in opt.norm_G:
      self.conv_0 = spectral_norm(self.conv_0)
      self.conv_1 = spectral_norm(self.conv_1)
      if self.learned_shortcut:
        self.conv_s = spectral_norm(self.conv_s)

    # define normalization layers
    spade_config_str = opt.norm_G.replace('spectral', '')
    if opt.spade_fusion:
      self.norm_0 = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
      self.norm_1 = SPADE(spade_config_str, fmiddle, self.spade_condi_nc, opt)
      if self.learned_shortcut:
        self.norm_s = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
    else:
      self.norm_0 = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)
      self.norm_1 = SPADE(spade_config_str, fmiddle, self.spade_condi_nc, opt)
      if self.learned_shortcut:
        self.norm_s = SPADE(spade_config_str, fin, self.spade_condi_nc, opt)

  # note the resnet block with SPADE also takes in |seg|,
  # the semantic segmentation map as input
  def forward(self, x, seg):
    attn_list = []
    x_s = self.shortcut(x, seg)

    norm_0 = self.norm_0(x, seg)
    if self.opt.spade_attn:
      attn_list.append(norm_0[1])
      norm_0 = norm_0[0]
    dx = self.conv_0(self.actvn(norm_0))

    norm_1 = self.norm_1(dx, seg)
    if self.opt.spade_attn:
      attn_list.append(norm_1[1])
      norm_1 = norm_1[0]
    dx = self.conv_1(self.actvn(norm_1))

    out = x_s + dx

    # if self.opt.spade_attn:
    return out, attn_list
    # else:
    #   return out

  def shortcut(self, x, seg):
    if self.learned_shortcut:
      norm_s = self.norm_s(x, seg)
      if self.opt.spade_attn:
        norm_s = norm_s[0]
      x_s = self.conv_s(norm_s)
    else:
      x_s = x
    return x_s

  def actvn(self, x):
    return F.leaky_relu(x, 2e-1)



# MySimpleSapleResBlock
class SimpleINResnetBlock(nn.Module):
  def __init__(self, fin, fout, opt, num_op):
    super().__init__()
    # Attributes
    self.learned_shortcut = (fin != fout)
    if opt.lang_dim != 0:
      if opt.use_op:
        self.semantic_lang_nc = opt.lang_dim + opt.semantic_nc
      elif opt.FiveK:
        self.semantic_lang_nc = opt.lang_dim
      elif opt.lang_encoder == 'bilstm':
        self.semantic_lang_nc = opt.lang_dim

    fmiddle = min(fin, fout)

    # create conv layers
    self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
    self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
    if self.learned_shortcut:
      self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

    # apply spectral norm if specified
    if 'spectral' in opt.norm_G:
      self.conv_0 = spectral_norm(self.conv_0)
      self.conv_1 = spectral_norm(self.conv_1)
      if self.learned_shortcut:
        self.conv_s = spectral_norm(self.conv_s)

    # define normalization layers
    self.norm_0 = MyIN(fin, opt, num_op)
    self.norm_1 = MyIN(fmiddle, opt, num_op)
    if self.learned_shortcut:
      self.norm_s = MyIN(fin, opt, num_op)

  # note the resnet block with SPADE also takes in gamma, beta
  def forward(self, x, lang_embed, input_semantics):
    """
    :param x:
    :param gamma: (b, op_num, h, w)
    :param beta:  (b, op_num, h, w)
    :return:
    """
    input_semantics = F.interpolate(input_semantics, size=x.size()[2:], mode='nearest')

    x_s = self.shortcut(x, lang_embed, input_semantics)

    dx = self.conv_0(self.actvn(self.norm_0(x, lang_embed, input_semantics)))
    dx = self.conv_1(self.actvn(self.norm_1(dx, lang_embed, input_semantics)))

    out = x_s + dx
    return out

  def shortcut(self, x, lang_embed, input_semantics):
    if self.learned_shortcut:
      x_s = self.conv_s(self.norm_s(x, lang_embed, input_semantics))
    else:
      x_s = x
    return x_s

  def actvn(self, x):
    return F.leaky_relu(x, 2e-1)



# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
  def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
    super().__init__()

    pw = (kernel_size - 1) // 2
    self.conv_block = nn.Sequential(
      nn.ReflectionPad2d(pw),
      norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
      activation,
      nn.ReflectionPad2d(pw),
      norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
    )

  def forward(self, x):
    y = self.conv_block(x)
    out = x + y
    return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
  def __init__(self, requires_grad=False):
    super().__init__()
    vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    for x in range(2):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(2, 7):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(7, 12):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])
    for x in range(12, 21):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])
    for x in range(21, 30):
      self.slice5.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, X):
    h_relu1 = self.slice1(X)
    h_relu2 = self.slice2(h_relu1)
    h_relu3 = self.slice3(h_relu2)
    h_relu4 = self.slice4(h_relu3)
    h_relu5 = self.slice5(h_relu4)
    out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
    return out
