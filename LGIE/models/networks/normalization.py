# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.mutan_fusion import MutanFusion, MutanHead

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
  # helper function to get # output channels of the previous layer
  def get_out_channel(layer):
    if hasattr(layer, 'out_channels'):
      return getattr(layer, 'out_channels')
    return layer.weight.size(0)

  # this function will be returned
  def add_norm_layer(layer, opt):
    nonlocal norm_type
    if norm_type.startswith('spectral'):
      if opt.netG != 'MYV2FusionGated':
        layer = spectral_norm(layer)
      subnorm_type = norm_type[len('spectral'):]

    if subnorm_type == 'none' or len(subnorm_type) == 0:
      return layer

    # remove bias in the previous layer, which is meaningless
    # since it has no effect after normalization
    if getattr(layer, 'bias', None) is not None:
      delattr(layer, 'bias')
      layer.register_parameter('bias', None)

    if subnorm_type == 'batch':
      norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
    elif subnorm_type == 'sync_batch':
      norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
    elif subnorm_type == 'instance':
      # todo: affine in Encoder is True
      # norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
      norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
    elif 'MyIN' in subnorm_type:
      norm_layer = MyIN(get_out_channel(layer))
    else:
      raise ValueError('normalization layer %s is not recognized' % subnorm_type)

    return nn.Sequential(layer, norm_layer)

  return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADEWithAttn(nn.Module):
  def __init__(self, config_text, norm_nc, input_nc, opt):
    super().__init__()
    self.opt = opt

    assert config_text.startswith('spade')
    parsed = re.search('spade(\D+)(\d)x\d', config_text)
    param_free_norm_type = str(parsed.group(1))
    ks = int(parsed.group(2))

    if param_free_norm_type == 'instance':
      self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
    elif param_free_norm_type == 'syncbatch':
      self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
    elif param_free_norm_type == 'batch':
      self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
    else:
      raise ValueError('%s is not a recognized param-free norm type in SPADE'
                       % param_free_norm_type)

    if opt.spade_fusion:
      # todo: add if for Fusion V1 and V2
      # V1: all norm_nc, V2: use hidden
      self.mutan_head = MutanHead(vis_ori_size=norm_nc, vis_reduced_size=norm_nc, lang_size=input_nc, hid_size=norm_nc)

    # The dimension of the intermediate embedding space. Yes, hardcoded.
    nhidden = 128
    pw = ks // 2

    # todo: add if for Fusion V1 and V2
    # V1: all norm_nc
    self.mlp_shared = nn.Sequential(
      nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw),
      nn.ReLU()
    )
    self.mlp_gamma = nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw)
    self.mlp_beta = nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw)

    # V2: use hidden
    # self.mlp_shared = nn.Sequential(
    #   nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
    #   nn.ReLU()
    # )
    # self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
    # self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

  def forward(self, x, segmap=None):

    # Part 1. generate parameter-free normalized activations
    normalized = self.param_free_norm(x)

    # Part 2. produce scaling and bias conditioned on semantic map
    # todo: 这里有大问题，直接把segmap用interpolate变小了，而且之前还是nearest差值
    segmap = F.interpolate(segmap, size=x.size()[2:])

    segmap = self.mutan_fusion(x, segmap)

    actv = self.mlp_shared(segmap)
    gamma = self.mlp_gamma(actv)
    beta = self.mlp_beta(actv)

    # apply scale and bias
    out = normalized * (1 + gamma) + beta

    return out


class SPADE(nn.Module):
  def __init__(self, config_text, norm_nc, input_nc, opt):
    super().__init__()
    self.opt = opt

    assert config_text.startswith('spade')
    parsed = re.search('spade(\D+)(\d)x\d', config_text)
    param_free_norm_type = str(parsed.group(1))
    ks = int(parsed.group(2))

    if param_free_norm_type == 'instance':
      self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
    elif param_free_norm_type == 'syncbatch':
      self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
    elif param_free_norm_type == 'batch':
      self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
    else:
      raise ValueError('%s is not a recognized param-free norm type in SPADE'
                       % param_free_norm_type)

    if opt.spade_fusion:
      # todo: add if for Fusion V1 and V2
      # V1: all norm_nc, V2: use hidden
      self.mutan_fusion = MutanFusion(vis_ori_size=norm_nc, vis_reduced_size=norm_nc, lang_size=input_nc, hid_size=norm_nc)
      # self.mutan_fusion = MutanFusion(vis_ori_size=norm_nc, vis_reduced_size=opt.lang_dim, lang_size=input_nc, hid_size=opt.lang_dim)
    elif opt.spade_attn:
      # 都用tanh转换到[-1, 1]区间，然后做点乘 --> [-1, 1]
      # self.mutan_head = MutanHead(vis_ori_size=norm_nc, vis_reduced_size=norm_nc, lang_size=input_nc, hid_size=norm_nc, reduce_sum=True, activation=nn.Sigmoid())
      self.mutan_head = MutanHead(vis_ori_size=norm_nc, vis_reduced_size=norm_nc, lang_size=input_nc, hid_size=norm_nc, reduce_sum=True)
      self.lang_tran = nn.Conv2d(opt.lang_dim, norm_nc, 1)    # todo: 不管lang_dim和当前的norm_nc是否一样，都做一个转化

    # The dimension of the intermediate embedding space. Yes, hardcoded.
    nhidden = 128
    pw = ks // 2

    # todo: add if for Fusion V1 and V2
    # V1: all norm_nc
    self.mlp_shared = nn.Sequential(
      nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw),
      nn.ReLU()
    )
    self.mlp_gamma = nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw)
    self.mlp_beta = nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw)

    # V2: use hidden
    # self.mlp_shared = nn.Sequential(
    #   nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
    #   nn.ReLU()
    # )
    # self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
    # self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

  def forward(self, x, segmap=None):

    # Part 1. generate parameter-free normalized activations
    normalized = self.param_free_norm(x)

    # Part 2. produce scaling and bias conditioned on semantic map
    # todo: 这里有大问题，直接把segmap用interpolate变小了，而且之前还是nearest差值
    segmap = F.interpolate(segmap, size=x.size()[2:])
    if self.opt.spade_fusion:
      segmap = self.mutan_fusion(x, segmap)
    elif self.opt.spade_attn:
      attn = self.mutan_head(x, segmap)
      ori_attn = attn
      attn = torch.sigmoid(attn)
      segmap = self.lang_tran(segmap)
      segmap = segmap * attn

    actv = self.mlp_shared(segmap)
    gamma = self.mlp_gamma(actv)
    beta = self.mlp_beta(actv)

    # apply scale and bias
    out = normalized * (1 + gamma) + beta

    if self.opt.spade_attn:
      return out, ori_attn
    else:
      return out


class MyIN(nn.Module):
  def __init__(self, norm_nc, opt, num_op):
    super().__init__()

    # assert config_text.startswith('spade')
    # parsed = re.search('spade(\D+)(\d)x\d', config_text)
    # param_free_norm_type = str(parsed.group(1))
    # ks = int(parsed.group(2))

    # if param_free_norm_type == 'instance':
    self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
    # elif param_free_norm_type == 'syncbatch':
    #   self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
    # elif param_free_norm_type == 'batch':
    #   self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
    # else:
    #   raise ValueError('%s is not a recognized param-free norm type in SPADE'
    #                    % param_free_norm_type)

    # todo: 把embed用mlp得到参数，再用prior调整
    nhidden = 128
    self.num_op = num_op
    if self.num_op > 1:
      self.reduce_dim = nn.Linear(self.num_op, 1)
    self.mlp_shared = nn.Linear(opt.lang_dim, nhidden)
    self.mlp_gamma = nn.Linear(nhidden, num_op)
    self.mlp_beta = nn.Linear(nhidden, num_op)
    # self.mlp_gamma = nn.Linear(norm_nc, num_op)
    # self.mlp_beta = nn.Linear(norm_nc, num_op)

  def forward(self, x, lang_embed, input_semantics):
    """
    :param input_semantics: (b, op_num, h, w)
    :param x: （b, c, h, w)
    :param lang_embed: (b, lang_embed)
    :return: (b, c, h, w)
    """
    # Part 1. generate parameter-free normalized activations
    normalized = self.param_free_norm(x)
    # if lang_embed.sum() == 0:  # None means there is no such phrase
    #   return normalized

    # Part 2. produce scaling and bias conditioned on semantic map
    actv = self.mlp_shared(lang_embed)
    gamma = self.mlp_gamma(actv)    # (b, num_op)
    beta = self.mlp_beta(actv)

    if self.num_op == 1: # inpaint
      gamma = gamma.unsqueeze(-1).unsqueeze(-1) * input_semantics[:, 1, :, :].unsqueeze(1)
      beta = beta.unsqueeze(-1).unsqueeze(-1) * input_semantics[:, 1, :, :].unsqueeze(1)
      out = normalized * (1 + gamma) + beta     # 因为这里的gamma和beta都是0，所以直接就是normalized！！不用做特殊处理
    else: # retouch
      # (b, num_op, h, w) * (b, num_op, h, w)
      gamma = gamma.unsqueeze(1).unsqueeze(2) * input_semantics[:, 2:, :, :].permute(0, 2, 3, 1)  # (b, h, w, 7)
      beta = beta.unsqueeze(1).unsqueeze(2) * input_semantics[:, 2:, :, :].permute(0, 2, 3, 1)
      out = x.unsqueeze(-1) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # (b, c, h, w, 7)
      out = self.reduce_dim(out).squeeze()

    return out
