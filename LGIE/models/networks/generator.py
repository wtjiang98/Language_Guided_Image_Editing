# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
# from models.networks.architecture import ResnetBlock as ResnetBlock, SNGatedConv2dWithActivation, SNGatedDeConv2dWithActivation, Self_Attn, get_pad
# from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock, SimpleINResnetBlock, SNGatedConv2dSPADEWithActivation, GatedConv2dWithActivation, SPADEGatedResnetBlock
from models.networks.architecture import *
from models.networks.mutan_fusion import MutanFusion
# import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models
from util.util_jwt import op2ind
import numpy as np

def conv1x1(in_planes, out_planes):
  """1x1 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                   padding=0, bias=False)

class GlobalBUAttentionGeneral(nn.Module):
  def __init__(self, idf, cdf, buattn_norm):
    super(GlobalBUAttentionGeneral, self).__init__()
    self.conv_context = conv1x1(cdf, idf)
    self.sm = nn.Softmax(dim=-1)
    self.mask = None
    self.eps = 1e-8
    self.buattn_norm = buattn_norm

  def applyMask(self, mask):
    self.mask = mask  # batch x sourceL

  def forward(self, input, context1, context2):
    """
        input: batch x idf2 x ih x iw (queryL=ihxiw), label features  (slabels_feat: batch x cdf2 x max_num_roi x 1)
        context1: batch x idf2 x sourceL, glove_word_embs
        context2: batch x cdf x sourceL, word_embs
    """
    ih, iw = input.size(2), input.size(3)
    queryL = ih * iw
    batch_size, sourceL = context2.size(0), context2.size(2)

    # --> batch x queryL x idf
    target = input.view(batch_size, -1, queryL)
    targetT = torch.transpose(target, 1, 2).contiguous()
    # batch x cdf x sourceL --> batch x cdf x sourceL x 1
    sourceT = context2.unsqueeze(3)
    # --> batch x idf x sourceL
    sourceT = self.conv_context(sourceT).squeeze(3)

    # Get attention
    # (batch x queryL x idf)(batch x idf x sourceL)
    # -->batch x queryL x sourceL
    attn = torch.bmm(targetT, context1)
    if self.buattn_norm:
      norm_targetT = torch.norm(targetT, 2, dim=2, keepdim=True)
      norm_context1 = torch.norm(context1, 2, dim=1, keepdim=True)
      attn = attn / (norm_targetT * norm_context1).clamp(min=self.eps)

    # --> batch*queryL x sourceL
    attn = attn.view(batch_size*queryL, sourceL)
    if self.mask is not None:
      # batch_size x sourceL --> batch_size*queryL x sourceL
      mask = self.mask.repeat(queryL, 1)
      attn.data.masked_fill_(mask.data, -float('inf'))
    attn = self.sm(attn)  # Eq. (2)
    # --> batch x queryL x sourceL
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attn = torch.transpose(attn, 1, 2).contiguous()

    # (batch x idf x sourceL)(batch x sourceL x queryL)
    # --> batch x idf x queryL
    weightedContext = torch.bmm(sourceT, attn)
    weightedContext = weightedContext.view(batch_size, -1, ih, iw)
    attn = attn.view(batch_size, -1, ih, iw)

    return weightedContext, attn


class CA_NET(nn.Module):
  # some code is modified from vae examples
  # (https://github.com/pytorch/examples/blob/master/vae/main.py)
  def __init__(self, opt):
    super(CA_NET, self).__init__()
    # self.t_dim = cfg.TEXT.DIMENSION
    # self.c_dim = cfg.GAN.CONDITION_DIM
    self.t_dim = opt.lang_dim
    self.c_dim = opt.ca_condition_dim
    self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
    self.relu = nn.ReLU()

  def encode(self, text_embedding):
    x = self.relu(self.fc(text_embedding))
    mu = x[:, :self.c_dim]
    logvar = x[:, self.c_dim:]
    return mu, logvar

  def reparametrize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    # eps = Variable(eps)
    return eps.mul(std).add_(mu)

  def forward(self, text_embedding):
    mu, logvar = self.encode(text_embedding)
    c_code = self.reparametrize(mu, logvar)
    return c_code, mu, logvar


class AugPhraseAttention(nn.Module):
  def __init__(self, input_dim, phrase_num):
    super(AugPhraseAttention, self).__init__()
    # initialize pivot
    self.fc = nn.Linear(input_dim, phrase_num)
    self.phrase_num = phrase_num

  def forward(self, context, embedded, prior=None):
    """

    :param context:
    :param embedded:
    :param input_semantics: prior for atten
    :return:
    """
    cxt_scores = self.fc(context) # (batch, seq_len, phrase_num)
    attn = F.softmax(cxt_scores, dim=1)  # (batch, seq_len, phrase_num), attn.sum(1) = 1.

    # mask zeros in "seq_len" dimension
    is_not_zero = (context[:, :, 0] != 0).float().unsqueeze(-1) # (batch, seq_len, 1)
    attn = attn * is_not_zero # (batch, seq_len, phrase_num)
    # 用mask得到该有的word之后再归一化到1
    attn = attn / attn.sum(1).view(attn.size(0), 1, attn.size(2)).expand(attn.size(0), attn.size(1), attn.size(2))

    # if has_op is not None:
    #   # make zeros in "phrase_num" dimenstion
    #   is_not_zero = has_op.unsqueeze(1)  # (b, phrase_num) --> (b, 1, phrase_num)
    #   attn = attn * is_not_zero # (batch, seq_len, phrase_num)

    # compute weighted embedding
    attn3 = attn.permute(0, 2, 1)     # (batch, phrase_num, seq_len)
    weighted_emb = torch.bmm(attn3, embedded) #  (batch, phrase_num, seq_len) * (batch, seq_len, word_vec_size) = (batch, phrase_num, word_vec_size)

    return attn3, weighted_emb


class PhraseAttention(nn.Module):
  def __init__(self, input_dim):
    super(PhraseAttention, self).__init__()
    # initialize pivot
    self.fc = nn.Linear(input_dim, 1)

  def forward(self, context, embedded):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    cxt_scores = self.fc(context).squeeze(2) # (batch, seq_len)
    attn = F.softmax(cxt_scores)  # (batch, seq_len), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (context[:, :, 0] != 0).float() # (batch, seq_len)
    attn = attn * is_not_zero # (batch, seq_len)
    attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len) # 太妙了，用mask得到该有的word之后再归一化到1

    # compute weighted embedding
    attn3 = attn.unsqueeze(1)     # (batch, 1, seq_len)
    weighted_emb = torch.bmm(attn3, embedded) #  (batch, 1, seq_len) * (batch, seq_len, word_vec_size) = (batch, 1, word_vec_size)
    weighted_emb = weighted_emb.squeeze(1)    # (batch, word_vec_size)

    return attn, weighted_emb


class SimpleINGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.phrase_num = 2
      self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      # self.op_weight_fc = nn.Linear(self.opt.lang_dim, self.phrase_num)

    # ****************** generate ******************

    if self.opt.encoder_nospade:
      kw = 3    # kernel size
      pw = int(np.ceil((kw - 1.0) / 2))
      norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
      self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
      self.down_0 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
      self.down_1 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))
      self.actvn = nn.LeakyReLU(0.2, False)

    self.G_middle_0 = SimpleINResnetBlock(4 * nf, 4 * nf, opt, num_op=1)
    self.G_middle_1 = SimpleINResnetBlock(4 * nf, 4 * nf, opt, num_op=1)

    if self.opt.skiplayer:
      self.up_0 = SimpleINResnetBlock(8 * nf, 2 * nf, opt, num_op=7)
      self.up_1 = SimpleINResnetBlock(4 * nf, 1 * nf, opt, num_op=7)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SimpleINResnetBlock(4 * nf, 2 * nf, opt, num_op=7)
      self.up_1 = SimpleINResnetBlock(2 * nf, 1 * nf, opt, num_op=7)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

  def forward(self, **input_G):
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # todo: 通过input_semantics知道有几个operator，设置先验
    has_op = input_semantics.sum([2, 3]).bool().float()

    op_attns, op_embs = self.aug_pattn(context_embs, words_embs, has_op)          # (b, phrase_num, lang_dim)
    # op_weights = F.softmax(self.op_weight_fc(sent_emb), dim=1)            # (b, phrase_num)

    # todo: 从先验中得知，如果没有该操作，则设为零
    # (b, lang_dim)
    retouch_embed = op_embs[:, 0, :] * has_op[:, 2:].sum(1).bool().float().unsqueeze(-1)     # 只要有一个不是0就行
    inpaint_embed = op_embs[:, 1, :] * has_op[:, 1].unsqueeze(-1)

    x_conv_embed = self.conv_embed(input_image)
    x_down_0 = self.down_0(self.actvn(x_conv_embed))
    x_down_1 = self.down_1(self.actvn(x_down_0))
    x_down_2 = self.actvn(x_down_1)

    x_middle = self.G_middle_0(x_down_2, inpaint_embed, input_semantics)
    x_middle = self.G_middle_1(x_middle, inpaint_embed, input_semantics)
    # x_middle = self.G_middle_2(x_middle, inpaint_embed, input_semantics)         # (1, 4 * nf, 64, 64)

    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_2], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, retouch_embed, input_semantics)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_0], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, retouch_embed, input_semantics)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_conv_embed], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, retouch_embed, input_semantics)
      x = self.up(x)
      x = self.up_1(x, retouch_embed, input_semantics)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, op_attns


class InpaintV2Generator(BaseNetwork):
  """
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  只有Inpaint，看看图像模糊是什么造成的
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    # parser.add_argument('--num_upsampling_layers',
    #                     choices=('normal', 'more', 'most'), default='normal',
    #                     help="If 'more', adds upsampling layer between the two middle resnet blocks."
    #                          " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    # ***** Gated Conv *****
    n_in_channel = 5
    self.coarse_net = nn.Sequential(
      # input is 5*256*256, but it is full convolution network, so it can be larger than 256
      SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
      # downsample 128
      SNGatedConv2dWithActivation(nf, 2 * nf, 4, 2, padding=get_pad(256, 4, 2)),
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      # downsample to 64
      SNGatedConv2dWithActivation(2 * nf, 4 * nf, 4, 2, padding=get_pad(128, 4, 2)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      # atrous convlution
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      # Self_Attn(4*nf, 'relu'),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      # upsample
      SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      # Self_Attn(2*nf, 'relu'),
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),

      SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
      # Self_Attn(nf//2, 'relu'),
      SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
    )

    self.refine_conv_net = nn.Sequential(
      # input is 5*256*256
      SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
      # downsample
      SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2)),
      SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      # downsample
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2)),
      SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
      # Self_Attn(4*nf, 'relu'),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
    )
    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.refine_upsample_net = nn.Sequential(
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),

      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),

      SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
      # Self_Attn(nf, 'relu'),
      SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
    )

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)

    # Coarse
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    x = self.coarse_net(input_imgs)
    x = torch.clamp(x, -1., 1.)
    coarse_x = x
    # Refine
    masked_imgs = input_image * (1 - inpaint_mask) + coarse_x * inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)
    x = self.refine_conv_net(input_imgs)
    x = self.refine_attn(x)
    x = self.refine_upsample_net(x)
    x = torch.clamp(x, -1., 1.)
    return x, coarse_x


class InpaintGenerator(BaseNetwork):
  """
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  只有Inpaint，看看图像模糊是什么造成的
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    # if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      # self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ****************** generate ******************

    # ***** Gated Conv *****
    n_in_channel = 5
    self.conv_net_down = nn.Sequential(
      # input is 5*256*256
      SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
      # downsample
      SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2)),
      SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      # downsample
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2)),
      SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),   # 从此的map size: (b, 256, 64, 64)
    )

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.conv_net_up = nn.Sequential(
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),
      SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
      # Self_Attn(nf, 'relu'),
      SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
    )

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    x = self.conv_net_down(input_imgs)

    x = self.middle_0(x)
    x = self.middle_1(x)
    x = self.middle_2(x)
    x = self.middle_3(x)
    x = self.middle_4(x)
    x = self.middle_5(x)

    x = self.refine_attn(x)
    x = self.conv_net_up(x)
    x = torch.clamp(x, -1., 1.)

    return x, x


class MYV2FusionInpaintSkipGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 4

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivation(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask], dim=1)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)   # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)   # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)   # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)   # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)   # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)   # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)   # (b, 128, 128, 128)
    x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)   # (b, 64, 256, 256)
    x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)   # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)   # (b, 3, 256, 256)

    # x = torch.clamp(x, -1., 1.)
    x = F.tanh(x_up_6)

    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }


class MYV2FusionInpaintResGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 4

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    # downsample
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # downsample
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))  # 从此的map size: (b, 256, 64, 64)

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))

    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))

    self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
      # Self_Attn(nf, 'relu'),
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

    self.res_donw_0 = SNGatedConv2dWithActivation(nf, 2 * nf, 4, 2, padding=get_pad(256, 4, 2))
    self.res_down_1 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.res_up_0 = SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.res_up_1 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask], dim=1)

    x_down_0 = self.down_0(input_imgs)     # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)              # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1) + self.res_donw_0(x_down_0)          # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)             # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3) + self.res_down_1(x_down_2)          # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb) + x_down_4
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb) + x_middle_1
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb) + x_middle_3

    x_up_0 = self.up_0(x_middle_5)                                   # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0) + x_middle_5                         # (b, 256, 64, 64)
    x_up_2 = self.up_2(x_up_1)                                     # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2) + self.res_up_0(x_up_1)              # (b, 128, 128, 128)
    x_up_4 = self.up_4(x_up_3)                                    # (b, 64, 256, 256)
    x_up_5 = self.up_5(x_up_4) + self.res_up_1(x_up_3)            # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)                                      # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)
    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }



class MYV2FusionInpaintV0Skip1Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

    # self.conv_net_down = nn.Sequential(
    #   # input is 5*256*256
    #   SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
    #   # downsample
    #   SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2)),
    #   SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
    #   # downsample
    #   SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2)),
    #   SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),  # 从此的map size: (b, 256, 64, 64)
    # )
    # self.conv_net_up = nn.Sequential(
    #   SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
    #   SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
    #   SNGatedDeConv2dWithActivationV0(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
    #   SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
    #   SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),
    #   SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
    #   # Self_Attn(nf, 'relu'),
    #   SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
    # )

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    # x = self.conv_net_down(input_imgs)
    # # start spade
    # x = self.middle_0(x, retouch_emb)
    # x = self.middle_1(x, retouch_emb)
    # x = self.middle_2(x, retouch_emb)
    # x = self.middle_3(x, retouch_emb)
    # x = self.middle_4(x, retouch_emb)
    # x = self.middle_5(x, retouch_emb)
    #
    # x = self.refine_attn(x)
    # x = self.conv_net_up(x)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    return x, retouch_attn


class MYV2FusionInpaintV0Skip1V2Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  Skip1：跳2层，尽量偏向正常的版本（IN）
  结果：光斑严重！！
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf  # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1), norm_config=opt.norm_G)
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2), norm_config=opt.norm_G)
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1), norm_config=opt.norm_G)
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2), norm_config=opt.norm_G)
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), norm_config=opt.norm_G)

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), norm_config=opt.norm_G)
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), norm_config=opt.norm_G)
    self.up_2 = SNGatedDeConv2dWithActivation(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1), norm_config=opt.norm_G)
    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1), norm_config=opt.norm_G)
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1), norm_config=opt.norm_G)
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1), norm_config=opt.norm_G)
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None, norm_config=opt.norm_G)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    # masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([input_image, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    return x, retouch_attn


class MYV2FusionInpaintV0Skip0V3SimpleV1Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch （V1:把retouch放到其他地方）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0 = self.middle_0(x_down_4)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3, attn_list_m0 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict



class MYV2FusionInpaintV0Skip0V3SimpleFiveKGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    # self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = torch.zeros_like(input_image)[:, 0, :, :].unsqueeze(1)
    input_imgs = torch.cat([input_image, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)

    # x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    # x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    x_up_0 = self.up_0(x_middle_5)

    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict


class MYV2FusionInpaintV0Skip0V3SimpleV2Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch
  SimpleV2: 在skip layer concat feature时，把inpaint部分去掉，并乘上spade_attn得到的map
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    # x_up_0 = self.up_0(x_middle_5)

    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict



class MYV2FusionInpaintV0Skip0V3SimpleGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    # x_up_0 = self.up_0(x_middle_5)

    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)

    x_down_4_useful = x_down_4 * F.interpolate((1-inpaint_mask), size=x_down_4.size()[2:])
    x_up_1 = torch.cat([x_up_1, x_down_4_useful], dim=1)

    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict


class MYV2FusionInpaintV0Skip0V3SimpleWoIRDGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    # x_up_0 = self.up_0(x_middle_5)

    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)

    x_down_4_useful = x_down_4 * F.interpolate((1-inpaint_mask), size=x_down_4.size()[2:])
    x_up_1 = torch.cat([x_up_1, x_down_4_useful], dim=1)

    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict


class MYV2FusionInpaintV0Skip0V3SimpleALLGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  Simple: 只有一层用来retouch
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.middle_2 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
    self.middle_3 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
    self.middle_5 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    # masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([input_image, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0)
      x_middle_2 = self.middle_2(x_middle_1)
      x_middle_3 = self.middle_3(x_middle_2)
      x_middle_4 = self.middle_4(x_middle_3)
      x_middle_5 = self.middle_5(x_middle_4)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    # x_up_0 = self.up_0(x_middle_5)

    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)

    x_down_4_useful = x_down_4 * F.interpolate((1-inpaint_mask), size=x_down_4.size()[2:])
    x_up_1 = torch.cat([x_up_1, x_down_4_useful], dim=1)

    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0]

    return x, ret_dict



class MYV2FusionInpaintV0Skip0V3Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  V3: use gray mask for inpainting
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
    # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    if self.opt.spade_attn:
      x_middle_0, attn_list_m0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1, attn_list_m1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2, attn_list_m2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3, attn_list_m3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4, attn_list_m4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5, attn_list_m5 = self.middle_5(x_middle_4, retouch_emb)
    else:
      x_middle_0 = self.middle_0(x_down_4, retouch_emb)
      x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
      x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
      x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
      x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
      x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_refined = self.refine_attn(x_middle_5)    # todo: dont forget!!!
    x_up_0 = self.up_0(x_refined)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    ret_dict = {'retouch_attn': retouch_attn}
    if self.opt.spade_attn:
      ret_dict['attn_list'] = [attn_list_m0, attn_list_m1, attn_list_m2, attn_list_m3, attn_list_m4, attn_list_m5]

    return x, ret_dict


class MYV2FusionInpaintV0Skip0V2Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    # retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    # masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([input_image, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())  # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    return x, retouch_attn



class MYV2FusionInpaintV0Skip0Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivationV0(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    # x = self.conv_net_down(input_imgs)
    # # start spade
    # x = self.middle_0(x, retouch_emb)
    # x = self.middle_1(x, retouch_emb)
    # x = self.middle_2(x, retouch_emb)
    # x = self.middle_3(x, retouch_emb)
    # x = self.middle_4(x, retouch_emb)
    # x = self.middle_5(x, retouch_emb)
    #
    # x = self.refine_attn(x)
    # x = self.conv_net_up(x)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)  # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)  # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)  # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)  # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)  # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)  # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)  # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)  # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)  # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)  # (b, 3, 256, 256)

    x = torch.clamp(x_up_6, -1., 1.)

    return x, retouch_attn



class MYV2FusionInpaintV0Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5
    self.conv_net_down = nn.Sequential(
      # input is 5*256*256
      SNGatedConv2dWithActivationV0(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
      # downsample
      SNGatedConv2dWithActivationV0(nf, nf, 4, 2, padding=get_pad(256, 4, 2)),
      SNGatedConv2dWithActivationV0(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      # downsample
      SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2)),
      SNGatedConv2dWithActivationV0(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),   # 从此的map size: (b, 256, 64, 64)
    )

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    self.conv_net_up = nn.Sequential(
      SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedConv2dWithActivationV0(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
      SNGatedDeConv2dWithActivationV0(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedConv2dWithActivationV0(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
      SNGatedDeConv2dWithActivationV0(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),
      SNGatedConv2dWithActivationV0(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
      # Self_Attn(nf, 'relu'),
      SNGatedConv2dWithActivationV0(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
    )

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    x = self.conv_net_down(input_imgs)
    # start spade
    x = self.middle_0(x, retouch_emb)
    x = self.middle_1(x, retouch_emb)
    x = self.middle_2(x, retouch_emb)
    x = self.middle_3(x, retouch_emb)
    x = self.middle_4(x, retouch_emb)
    x = self.middle_5(x, retouch_emb)

    x = self.refine_attn(x)
    x = self.conv_net_up(x)
    x = torch.clamp(x, -1., 1.)

    return x, retouch_attn



class MYV2FusionInpaintGenerator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 5

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)
    if self.opt.skiplayer:
      self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
      self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
      self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
      self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
      self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

      self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
      self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
      self.up_2 = SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
      self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
      self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
      self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
      self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)

    else:
      self.conv_net_down = nn.Sequential(
        # input is 5*256*256
        SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1)),
        # downsample
        SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2)),
        SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
        # downsample
        SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2)),
        SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),  # 从此的map size: (b, 256, 64, 64)
      )
      self.conv_net_up = nn.Sequential(
        SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
        SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1)),
        SNGatedDeConv2dWithActivation(2, 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
        SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1)),
        SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1)),
        SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1)),
        # Self_Attn(nf, 'relu'),
        SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
      )


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    if self.opt.skiplayer:
      x = self.down_0(input_imgs)
      x = self.down_1(x)
      x = self.down_2(x)
      x = self.down_3(x)
      x = self.down_4(x)

      x = self.middle_0(x, retouch_emb)
      x = self.middle_1(x, retouch_emb)
      x = self.middle_2(x, retouch_emb)
      x = self.middle_3(x, retouch_emb)
      x = self.middle_4(x, retouch_emb)
      x = self.middle_5(x, retouch_emb)

      x = self.up_0(x)
      x = self.up_1(x)
      x = self.up_2(x)
      x = self.up_3(x)
      x = self.up_4(x)
      x = self.up_5(x)
      x = self.up_6(x)

    else:
      x = self.conv_net_down(input_imgs)
      # start spade
      x = self.middle_0(x, retouch_emb)
      x = self.middle_1(x, retouch_emb)
      x = self.middle_2(x, retouch_emb)
      x = self.middle_3(x, retouch_emb)
      x = self.middle_4(x, retouch_emb)
      x = self.middle_5(x, retouch_emb)

      x = self.refine_attn(x)
      x = self.conv_net_up(x)

    # x = torch.clamp(x, -1., 1.)
    x = F.tanh(x)

    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }



class MYV2FusionInpaintSkip0V2Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  Skip0：只跳一小层 V2: 把inpaint不要变灰色，retouch的地方不包括inpaint
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 4

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivation(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    # masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([input_image, inpaint_mask], dim=1)

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_mask = torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool()
    retouch_mask = retouch_mask & (retouch_mask ^ inpaint_mask.bool())      # 相当于retouch - inpaint！
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * retouch_mask.float()

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)   # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)   # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)   # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)   # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)   # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)   # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)   # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)   # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)   # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)   # (b, 3, 256, 256)

    # x = torch.clamp(x, -1., 1.)
    x = F.tanh(x_up_6)
    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }



class MYV2FusionInpaintSkip0Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  Skip0：只跳一小层
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 4

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivation(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask], dim=1)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)   # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)   # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)   # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)   # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)   # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)   # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)   # (b, 128, 128, 128)
    # x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)   # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)   # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)   # (b, 3, 256, 256)

    # x = torch.clamp(x, -1., 1.)
    x = F.tanh(x_up_6)

    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }


class MYV2FusionInpaintSkip1Generator(BaseNetwork):
  """
  MYV2: 优化embed计算速度；把不在的phrase不值为零，与上面做对比
  Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
  Inpaint: 在整个encoder, decoder层用上GatedConv，而不再使用gamma, beta来卷积（其中都使用SpectralNorm，似乎原来也在opt.norm_E中使用了）
  Skip0：只跳一小层
  """
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.phrase_num = 1
      # self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ***** Gated Conv *****
    n_in_channel = 4

    # 只在中间做retouch
    self.middle_0 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_1 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1), opt=opt)
    self.middle_2 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), opt=opt)
    self.middle_3 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), opt=opt)
      # Self_Attn(4*nf, 'relu'),
    self.middle_4 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), opt=opt)
    self.middle_5 = SNGatedConv2dSPADEWithActivation(4 * nf, 4 * nf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), opt=opt)

    self.refine_attn = Self_Attn(4 * nf, 'relu', with_attn=False)

    self.down_0 = SNGatedConv2dWithActivation(n_in_channel, nf, 5, 1, padding=get_pad(256, 5, 1))
    self.down_1 = SNGatedConv2dWithActivation(nf, nf, 4, 2, padding=get_pad(256, 4, 2))
    self.down_2 = SNGatedConv2dWithActivation(nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.down_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 4, 2, padding=get_pad(128, 4, 2))
    self.down_4 = SNGatedConv2dWithActivation(2 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))

    self.up_0 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_1 = SNGatedConv2dWithActivation(4 * nf, 4 * nf, 3, 1, padding=get_pad(64, 3, 1))
    self.up_2 = SNGatedDeConv2dWithActivation(2, 2 * 4 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_3 = SNGatedConv2dWithActivation(2 * nf, 2 * nf, 3, 1, padding=get_pad(128, 3, 1))
    self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_5 = SNGatedConv2dWithActivation(2 * nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    # self.up_4 = SNGatedDeConv2dWithActivation(2, 2 * nf, nf, 3, 1, padding=get_pad(256, 3, 1))
    self.up_5 = SNGatedConv2dWithActivation(nf, nf // 2, 3, 1, padding=get_pad(256, 3, 1))
    self.up_6 = SNGatedConv2dWithActivation(nf // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_emb = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask], dim=1)

    x_down_0 = self.down_0(input_imgs)  # (b, 64, 256, 256)
    x_down_1 = self.down_1(x_down_0)   # (b, 64, 128, 128)
    x_down_2 = self.down_2(x_down_1)   # (b, 128, 128, 128)
    x_down_3 = self.down_3(x_down_2)   # (b, 128, 64, 64)
    x_down_4 = self.down_4(x_down_3)   # (b, 256, 64, 64)
    # start spade
    x_middle_0 = self.middle_0(x_down_4, retouch_emb)
    x_middle_1 = self.middle_1(x_middle_0, retouch_emb)
    x_middle_2 = self.middle_2(x_middle_1, retouch_emb)
    x_middle_3 = self.middle_3(x_middle_2, retouch_emb)
    x_middle_4 = self.middle_4(x_middle_3, retouch_emb)
    x_middle_5 = self.middle_5(x_middle_4, retouch_emb)

    x_up_0 = self.up_0(x_middle_5)  # (b, 256, 64, 64)
    x_up_1 = self.up_1(x_up_0)   # (b, 256, 64, 64)
    x_up_1 = torch.cat([x_up_1, x_down_4], dim=1)
    x_up_2 = self.up_2(x_up_1)   # (b, 128, 128, 128)
    x_up_3 = self.up_3(x_up_2)   # (b, 128, 128, 128)
    x_up_3 = torch.cat([x_up_3, x_down_2], dim=1)
    x_up_4 = self.up_4(x_up_3)   # (b, 64, 256, 256)
    # x_up_4 = torch.cat([x_up_4, x_down_0], dim=1)
    x_up_5 = self.up_5(x_up_4)   # (b, 32, 256, 256)
    x_up_6 = self.up_6(x_up_5)   # (b, 3, 256, 256)

    # x = torch.clamp(x, -1., 1.)
    x = F.tanh(x_up_6)

    return x,  {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }

class MYV2FusionFiveKGenerator(BaseNetwork):
  """SInpaint: Simple Inpaint，不使用inpaint phrase"""
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if self.opt.use_pattn:
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ****************** generate ******************

    kw = 3    # kernel size
    pw = int(np.ceil((kw - 1.0) / 2))
    norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
    self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
    self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw), opt)
    self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw), opt)
    self.actvn = nn.LeakyReLU(0.2, False)

    self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_2 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

    # MUTAN Fusion
    # self.inpaint_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)
    # self.retouch_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_condition = retouch_emb.unsqueeze(-1).unsqueeze(-1)

    x_down_0 = self.conv_embed(input_image)    # 输入masked image！！
    x_down_1 = self.layer1(self.actvn(x_down_0))
    x_down_2 = self.layer2(self.actvn(x_down_1))
    x_down_3 = self.actvn(x_down_2)      # (b, 4 * nf=256, 64, 64)

    # start inpainting
    x_middle, attn_list_m0 = self.G_middle_0(x_down_3, retouch_condition)
    x_middle, attn_list_m1 = self.G_middle_1(x_middle, retouch_condition)
    x_middle, attn_list_m2 = self.G_middle_2(x_middle, retouch_condition)         # (1, 4 * nf, 64, 64)

    # start retouching
    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, retouch_condition)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, retouch_condition)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x, attn_list_up0 = self.up_0(x, retouch_condition)
      x = self.up(x)
      x, attn_list_up1 = self.up_1(x, retouch_condition)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, {
      'retouch_attn': retouch_attn,
      'attn_list': attn_list_m0 + attn_list_m1 + attn_list_m2 + attn_list_up0 + attn_list_up1
    }


class MYV2FusionSInpaintGenerator(BaseNetwork):
  """SInpaint: Simple Inpaint，不使用inpaint phrase"""
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ****************** generate ******************

    kw = 3    # kernel size
    pw = int(np.ceil((kw - 1.0) / 2))
    norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
    self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
    self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw), opt)
    self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw), opt)
    self.actvn = nn.LeakyReLU(0.2, False)

    self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt, spade_condi_nc=4)   # (rgb + inpaint_mask)
    self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt, spade_condi_nc=4)
    self.G_middle_2 = SPADEResnetBlock(4 * nf, 4 * nf, opt, spade_condi_nc=4)

    if self.opt.skiplayer:
      self.up_0 = SPADEResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

    # MUTAN Fusion
    # self.inpaint_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)
    # self.retouch_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_condition = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()
    # get condition
    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)

    # make the inpaint obj grey!
    masked_image = input_image * (1 - inpaint_mask) + inpaint_mask
    inpaint_condition = torch.cat([masked_image, (1 - inpaint_mask)], dim=1)

    x_down_0 = self.conv_embed(masked_image)    # 输入masked image！！
    x_down_1 = self.layer1(self.actvn(x_down_0))
    x_down_2 = self.layer2(self.actvn(x_down_1))
    x_down_3 = self.actvn(x_down_2)      # (b, 4 * nf=256, 64, 64)

    # start inpainting
    x_middle = self.G_middle_0(x_down_3, inpaint_condition)
    x_middle = self.G_middle_1(x_middle, inpaint_condition)
    x_middle = self.G_middle_2(x_middle, inpaint_condition)         # (1, 4 * nf, 64, 64)

    # start retouching
    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, retouch_condition)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, retouch_condition)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, retouch_condition)
      x = self.up(x)
      x = self.up_1(x, retouch_condition)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, retouch_attn


class MYV2FusionGatedV2Generator(BaseNetwork):
  """V2: 在encoder也加上shortcut"""
  @staticmethod
  def modify_commandline_options(parser, is_train):
    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ****************** generate ******************

    kw = 3    # kernel size
    pw = int(np.ceil((kw - 1.0) / 2))
    norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
    # self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
    # self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
    # self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))

    self.conv_embed = SNGatedConv2dWithActivation(4, nf, kw, padding=pw, norm_type='instance')
    # self.layer1 = SNGatedConv2dWithActivation(nf * 1, nf * 2, kw, stride=2, padding=pw, norm_type='instance')
    # self.layer2 = SNGatedConv2dWithActivation(nf * 2, nf * 4, kw, stride=2, padding=pw, norm_type='instance')
    # self.actvn = nn.LeakyReLU(0.2, False)

    self.down_0 = SPADEGatedResnetBlock(1 * nf, 2 * nf, opt)
    self.down_1 = SPADEGatedResnetBlock(2 * nf, 4 * nf, opt)

    self.G_middle_0 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_2 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEGatedResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEGatedResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEGatedResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEGatedResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = SNGatedConv2dWithActivation(nf, 3, 3, padding=1, norm_type='instance')

    self.up = nn.Upsample(scale_factor=2)
    self.down = nn.Upsample(scale_factor=0.5)

    # MUTAN Fusion
    # self.inpaint_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)
    # self.retouch_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_condition = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    # make the inpaint obj grey!
    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask], dim=1)

    x_down_0 = self.conv_embed(input_imgs)    # 输入masked image！！
    # x_down_1 = self.layer1(self.actvn(x_down_0))
    # x_down_2 = self.layer2(self.actvn(x_down_1))
    # x_down_3 = self.actvn(x_down_2)      # (b, 4 * nf=256, 64, 64)
    x_down_1 = self.down_0(x_down_0)
    x_down_2 = self.down(x_down_1)
    x_down_3 = self.down_1(x_down_2)
    x_down_4 = self.down(x_down_3)

    # start inpainting
    x_middle = self.G_middle_0(x_down_4, retouch_condition)
    x_middle = self.G_middle_1(x_middle, retouch_condition)
    x_middle = self.G_middle_2(x_middle, retouch_condition)         # (1, 4 * nf, 64, 64)

    # start retouching
    if self.opt.skiplayer:
      pass
      # x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      # x = self.up_0(x, retouch_condition)                                 # (1, 2 * nf, 128, 128)
      # x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      # x = self.up_1(x, retouch_condition)                                  # (1, nf, 256, 256)
      # x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, retouch_condition)
      x = self.up(x)
      x = self.up_1(x, retouch_condition)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }


class MYV2FusionGatedGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.retouch_attn = PhraseAttention(input_dim=self.opt.lang_dim)

    # ****************** generate ******************

    kw = 3    # kernel size
    pw = int(np.ceil((kw - 1.0) / 2))
    norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
    # self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
    # self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
    # self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))

    self.conv_embed = SNGatedConv2dWithActivation(5, nf, kw, padding=pw, norm_type='instance')
    self.layer1 = SNGatedConv2dWithActivation(nf * 1, nf * 2, kw, stride=2, padding=pw, norm_type='instance')
    self.layer2 = SNGatedConv2dWithActivation(nf * 2, nf * 4, kw, stride=2, padding=pw, norm_type='instance')
    self.actvn = nn.LeakyReLU(0.2, False)

    self.G_middle_0 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_2 = SPADEGatedResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEGatedResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEGatedResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEGatedResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEGatedResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = SNGatedConv2dWithActivation(nf, 3, 3, padding=1, norm_type='instance')

    self.up = nn.Upsample(scale_factor=2)

    # MUTAN Fusion
    # self.inpaint_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)
    # self.retouch_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    retouch_attn, retouch_emb = self.retouch_attn(context_embs, words_embs)  # (b, lang_dim)
    retouch_condition = retouch_emb.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()

    # make the inpaint obj grey!
    inpaint_mask = input_semantics[:, 1, :, :].unsqueeze(1)
    masked_imgs = input_image * (1 - inpaint_mask) + inpaint_mask
    input_imgs = torch.cat([masked_imgs, inpaint_mask, torch.full_like(inpaint_mask, 1.)], dim=1)

    x_down_0 = self.conv_embed(input_imgs)    # 输入masked image！！
    x_down_1 = self.layer1(self.actvn(x_down_0))
    x_down_2 = self.layer2(self.actvn(x_down_1))
    x_down_3 = self.actvn(x_down_2)      # (b, 4 * nf=256, 64, 64)

    # start inpainting
    x_middle = self.G_middle_0(x_down_3, retouch_condition)
    x_middle = self.G_middle_1(x_middle, retouch_condition)
    x_middle = self.G_middle_2(x_middle, retouch_condition)         # (1, 4 * nf, 64, 64)

    # start retouching
    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, retouch_condition)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, retouch_condition)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, retouch_condition)
      x = self.up(x)
      x = self.up_1(x, retouch_condition)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, {
      'retouch_attn': retouch_attn,
      'masks': inpaint_mask
    }


class MYV2FusionGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.phrase_num = 2
      self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)

    # ****************** generate ******************

    kw = 3    # kernel size
    pw = int(np.ceil((kw - 1.0) / 2))
    norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
    # self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
    # self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
    # self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))

    self.conv_embed = GatedConv2dWithActivation(3, nf, kw, padding=pw, activation=None)
    self.layer1 = norm_layer(GatedConv2dWithActivation(nf * 1, nf * 2, kw, stride=2, padding=pw, activation=None))
    self.layer2 = norm_layer(GatedConv2dWithActivation(nf * 2, nf * 4, kw, stride=2, padding=pw, activation=None))
    self.actvn = nn.LeakyReLU(0.2, False)

    self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

    # MUTAN Fusion
    # self.inpaint_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)
    # self.retouch_mutan_fusion = MutanFusion(self.opt, vis_reduced_size=4 * nf, hid_size=4 * nf)


  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    op_attns, op_embs = self.aug_pattn(context_embs, words_embs)         # (b, phrase_num, lang_dim)
    # op_weights = F.softmax(self.op_weight_fc(sent_emb), dim=1)  # (b, phrase_num)

    # (b, lang_dim)
    # has_op = input_semantics.sum([2, 3]).bool().float()
    # retouch_embed = op_embs[:, 0, :] * has_op[:, 2:].sum(1).bool().float().unsqueeze(-1)  # 只要有一个不是0就行
    # inpaint_embed = op_embs[:, 1, :] * has_op[:, 1].unsqueeze(-1)
    retouch_embed = op_embs[:, 0, :]
    inpaint_embed = op_embs[:, 1, :]

    # seg_inpaint = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    # seg_retouch = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    # for b in range(input_image.size(0)):
    #   for op_idx in range(1, self.opt.label_nc + self.opt.contain_dontcare_label):  # 因为第一维是废的
    #     # # input semantics是从1开始的
    #     row_idx, col_idx = torch.nonzero(input_semantics[b, op_idx, :, :], as_tuple=True)
    #     if op_idx != 1:  # for retouch opertor
    #       seg_retouch[b, :, row_idx, col_idx] = retouch_embed.unsqueeze(-1)
    #     else:  # for inpaint operator
    #       seg_inpaint[b, :, row_idx, col_idx] = inpaint_embed.unsqueeze(-1)

    x_down_0 = self.conv_embed(input_image)
    x_down_1 = self.layer1(self.actvn(x_down_0))
    x_down_2 = self.layer2(self.actvn(x_down_1))
    x_down_3 = self.actvn(x_down_2)      # (b, 4 * nf=256, 64, 64)

    # MUTAN Fusion
    # retouch_embed = self.retouch_mutan_fusion(x_down_3, retouch_embed)
    # inpaint_embed = self.inpaint_mutan_fusion(x_down_3, inpaint_embed)  # (b, 4 * nf, h, w)

    # get condition
    seg_retouch = retouch_embed.unsqueeze(-1).unsqueeze(-1) * torch.sum(input_semantics[:, 2:, :, :], dim=1, keepdim=True).bool().float()
    seg_inpaint = inpaint_embed.unsqueeze(-1).unsqueeze(-1) * input_semantics[:, 1, :, :].unsqueeze(1)

    # start inpainting
    x_middle = self.G_middle_0(x_down_3, seg_inpaint)
    x_middle = self.G_middle_1(x_middle, seg_inpaint)
    # x_middle = self.G_middle_2(x_middle, seg_inpaint)         # (1, 4 * nf, 64, 64)

    # start retouching
    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, seg_retouch)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, seg_retouch)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, seg_retouch)
      x = self.up(x)
      x = self.up_1(x, seg_retouch)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, op_attns


class MYV2Generator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64
    # if self.opt.FiveK:
    #   self.embed_to_op = nn.Linear(self.opt.lang_dim, self.opt.label_nc + opt.contain_dontcare_label)

    if (not self.opt.FiveK) and self.opt.use_pattn:
      self.phrase_num = 2
      self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)

    # ****************** generate ******************

    if self.opt.encoder_nospade:
      kw = 3    # kernel size
      pw = int(np.ceil((kw - 1.0) / 2))
      norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
      self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
      self.layer1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw), opt)
      self.layer2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw), opt)
      self.actvn = nn.LeakyReLU(0.2, False)
    else:
      pass
      # self.conv_embed = nn.Conv2d(3, nf, 3, padding=1)
      # self.down_0 = SPADEResnetBlock(1 * nf, 2 * nf, opt)
      # self.down_1 = SPADEResnetBlock(2 * nf, 4 * nf, opt)
      # self.down = nn.Upsample(scale_factor=0.5)

    self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    has_op = input_semantics.sum([2, 3]).bool().float()
    # start phrase attn
    # inpaint_attn, inpaint_phrase_emb = self.inpaint_att(context_embs, words_embs) # (b, lang_dim)
    # retouch_attn, retouch_phrase_emb = self.retouch_att(context_embs, words_embs)
    op_attns, op_embs = self.aug_pattn(context_embs, words_embs)         # (b, phrase_num, lang_dim)
    # op_weights = F.softmax(self.op_weight_fc(sent_emb), dim=1)  # (b, phrase_num)

    # (b, lang_dim)
    # retouch_embed = op_embs[:, 0, :] * has_op[:, 2:].sum(1).bool().float().unsqueeze(-1)  # 只要有一个不是0就行
    # inpaint_embed = op_embs[:, 1, :] * has_op[:, 1].unsqueeze(-1)
    retouch_embed = op_embs[:, 0, :]
    inpaint_embed = op_embs[:, 1, :]

    seg_retouch = retouch_embed.unsqueeze(-1).unsqueeze(-1) * input_semantics[:, 2:, :, :].sum(1).unsqueeze(1).bool().float()
    seg_inpaint = inpaint_embed.unsqueeze(-1).unsqueeze(-1) * input_semantics[:, 1, :, :].unsqueeze(1)

    # seg_inpaint = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    # seg_retouch = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    # for b in range(input_image.size(0)):
    #   for op_idx in range(1, self.opt.label_nc + self.opt.contain_dontcare_label):  # 因为第一维是废的
    #     # # input semantics是从1开始的
    #     row_idx, col_idx = torch.nonzero(input_semantics[b, op_idx, :, :], as_tuple=True)
    #     if op_idx != 1:  # for retouch opertor
    #       seg_retouch[b, :, row_idx, col_idx] = retouch_embed.unsqueeze(-1)
    #     else:  # for inpaint operator
    #       seg_inpaint[b, :, row_idx, col_idx] = inpaint_embed.unsqueeze(-1)


    x_down_0 = self.conv_embed(input_image)
    x_down_1 = self.layer1(self.actvn(x_down_0))
    x_down_2 = self.layer2(self.actvn(x_down_1))
    x_down_3 = self.actvn(x_down_2)

    x_middle = self.G_middle_0(x_down_3, seg_inpaint)
    x_middle = self.G_middle_1(x_middle, seg_inpaint)
    # x_middle = self.G_middle_2(x_middle, seg_inpaint)         # (1, 4 * nf, 64, 64)

    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, seg_retouch)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, seg_retouch)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, seg_retouch)
      x = self.up(x)
      x = self.up_1(x, seg_retouch)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, op_attns


class MYGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks."
                             " If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf            # 64
    # if self.opt.FiveK:
    #   self.embed_to_op = nn.Linear(self.opt.lang_dim, self.opt.label_nc + opt.contain_dontcare_label)

    if (not self.opt.FiveK) and self.opt.use_pattn:
      # self.inpaint_att = PhraseAttention(self.opt.lang_dim)
      # self.retouch_att = PhraseAttention(self.opt.lang_dim)
      self.phrase_num = 8
      self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.op_weight_fc = nn.Linear(self.opt.lang_dim, self.phrase_num)

    if self.opt.use_buattn:
      self.buattn = GlobalBUAttentionGeneral(opt.lang_dim, opt.lang_dim, opt.buattn_norm)

    # ****************** predict filter parameter ******************
    if self.opt.predict_param:
      self.resnet34 = models.resnet34(pretrained=False, num_classes=opt.label_nc + opt.contain_dontcare_label)

    # ****************** generate ******************

    if self.opt.encoder_nospade:
      kw = 3    # kernel size
      pw = int(np.ceil((kw - 1.0) / 2))
      norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
      self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
      # self.layer1 = norm_layer(nn.Conv2d(3, nf, kw, stride=2, padding=pw))
      self.layer2 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw), opt)
      self.layer3 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw), opt)

      self.actvn = nn.LeakyReLU(0.2, False)
    else:
      pass
      # self.conv_embed = nn.Conv2d(3, nf, 3, padding=1)
      # self.down_0 = SPADEResnetBlock(1 * nf, 2 * nf, opt)
      # self.down_1 = SPADEResnetBlock(2 * nf, 4 * nf, opt)
      # self.down = nn.Upsample(scale_factor=0.5)

    self.G_middle_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
    self.G_middle_2 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    if self.opt.skiplayer:
      self.up_0 = SPADEResnetBlock(8 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(4 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)
    else:
      self.up_0 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
      self.up_1 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
      self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

  # def forward(self, input_semantics, input_image, sent_embed, glove_words_embed, labels_embed):
  def forward(self, **input_G):
    """
    :param input_semantics: (1, label_nc+lang_dim, 256, 256)
    :param input_image: (1, 3, h, w)
    :param sent_embed: (bs, )
    :param glove_words_embed: probably be None (1, caption_len, lang_dim)
    :param labels_embed: probably be None (1, label_nc, lang_dim)
    :return:
    """
    input_image = input_G['input_image']
    # seg代表的是输入G网络的condition

    input_semantics = input_G['input_semantics']
    words_embs = input_G['words_embs']
    sent_emb = input_G['sent_emb']
    context_embs = input_G['context_embs']

    # start phrase attn
    # inpaint_attn, inpaint_phrase_emb = self.inpaint_att(context_embs, words_embs) # (b, lang_dim)
    # retouch_attn, retouch_phrase_emb = self.retouch_att(context_embs, words_embs)
    op_attns, op_embs = self.aug_pattn(context_embs, words_embs)         # (b, phrase_num, lang_dim)
    op_weights = F.softmax(self.op_weight_fc(sent_emb), dim=1)  # (b, phrase_num)

    seg_inpaint = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    seg_retouch = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
    for b in range(input_image.size(0)):
      for op_idx in range(1, self.opt.label_nc + self.opt.contain_dontcare_label):  # 因为第一维是废的
        # # input semantics是从1开始的
        row_idx, col_idx = torch.nonzero(input_semantics[b, op_idx, :, :], as_tuple=True)
        if op_idx != 1:  # for retouch opertor
          seg_retouch[b, :, row_idx, col_idx] = (op_embs[b, 0] * op_weights[b, 0]).unsqueeze(-1)
        else:  # for inpaint operator
          seg_inpaint[b, :, row_idx, col_idx] = (op_embs[b, 1] * op_weights[b, 1]).unsqueeze(-1)

    # predict paramter
    if self.opt.encoder_nospade:
      x_down_0 = self.conv_embed(input_image)
      x_down_1 = self.layer2(self.actvn(x_down_0))
      x_down_2 = self.layer3(self.actvn(x_down_1))
      x_down_3 = self.actvn(x_down_2)
    else:
      pass
      # x_conv_embed = self.conv_embed(F.leaky_relu(input_image, 2e-1))    # (1, nf, 256, 256)
      # # start generate
      # x_down_0 = self.down(x_conv_embed)                # (1, nf, 128, 128)
      # x_down_1 = self.down_0(x_down_0, seg_inpaint)             # (1, 2 * nf, 128, 128)
      # x_down_2 = self.down(x_down_1)                    # (1, 2 * nf, 64, 64)
      # x_down_3 = self.down_1(x_down_2, seg_inpaint)             # (1, 4 * nf, 64, 64) = (1, 256, 64, 64)

    x_middle = self.G_middle_0(x_down_3, seg_inpaint)
    x_middle = self.G_middle_1(x_middle, seg_inpaint)
    x_middle = self.G_middle_2(x_middle, seg_inpaint)         # (1, 4 * nf, 64, 64)

    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, seg_retouch)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, seg_retouch)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_down_0], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, seg_retouch)
      x = self.up(x)
      x = self.up_1(x, seg_retouch)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    return x, op_attns
    # return x, [seg_inpaint, seg_retouch], [op_attns, op_weights]


class SPADEGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")

    return parser

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    nf = opt.ngf

    self.sw, self.sh = self.compute_latent_vector_size(opt)

    if opt.use_vae:
      # In case of VAE, we will sample from random z vector
      self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
    else:
      # Otherwise, we make the network deterministic by starting with
      # downsampled segmentation map instead of random z
      self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

    self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
    self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
    self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

    self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
    self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
    self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
    self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

    final_nc = nf

    if opt.num_upsampling_layers == 'most':
      self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
      final_nc = nf // 2

    self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

    self.up = nn.Upsample(scale_factor=2)

  def compute_latent_vector_size(self, opt):
    if opt.num_upsampling_layers == 'normal':
      num_up_layers = 5
    elif opt.num_upsampling_layers == 'more':
      num_up_layers = 6
    elif opt.num_upsampling_layers == 'most':
      num_up_layers = 7
    else:
      raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                       opt.num_upsampling_layers)

    sw = opt.crop_size // (2**num_up_layers)
    sh = round(sw / opt.aspect_ratio)

    return sw, sh

  def forward(self, input, z=None):
    seg = input # (bs, nc, h, w) segmentation map

    if self.opt.use_vae:
      # we sample z from unit normal and reshape the tensor
      if z is None:
        z = torch.randn(input.size(0), self.opt.z_dim,
                        dtype=torch.float32, device=input.get_device())
      x = self.fc(z)
      x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
    else:
      # we downsample segmap and run convolution
      x = F.interpolate(seg, size=(self.sh, self.sw))     # 根据上采样的层数确定down sample大小 (1, 184, 8, 8)
      x = self.fc(x)          # (1, 1024, 8, 8)

    x = self.head_0(x, seg)

    x = self.up(x)
    x = self.G_middle_0(x, seg)

    if self.opt.num_upsampling_layers == 'more' or \
        self.opt.num_upsampling_layers == 'most':
      x = self.up(x)

    x = self.G_middle_1(x, seg)

    x = self.up(x)              # up负责上采样
    x = self.up_0(x, seg)       # up_负责让channel减半
    x = self.up(x)
    x = self.up_1(x, seg)
    x = self.up(x)
    x = self.up_2(x, seg)
    x = self.up(x)
    x = self.up_3(x, seg)       # (bs, 64, h, w)

    if self.opt.num_upsampling_layers == 'most':
      x = self.up(x)
      x = self.up_4(x, seg)

    x = self.conv_img(F.leaky_relu(x, 2e-1))
    x = F.tanh(x)

    return x


# class MyPix2PixHDGenerator(BaseNetwork):
#   @staticmethod
#   def modify_commandline_options(parser, is_train):
#     parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
#     parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
#     parser.add_argument('--resnet_kernel_size', type=int, default=3,
#                         help='kernel size of the resnet block')
#     parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
#                         help='kernel size of the first convolution')
#     parser.set_defaults(norm_G='instance')
#     return parser
#
#   def __init__(self, opt):
#     super().__init__()
#     self.actvn = nn.LeakyReLU(0.2, False)
#     norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
#     nf = opt.ngf
#     kw = 3  # kernel size
#     pw = int(np.ceil((kw - 1.0) / 2))
#
#     self.conv_embed = nn.Conv2d(3, nf, kw, padding=pw)
#     self.down_1 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
#     self.down_2 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))
#
#     model = []
#
#     # # initial conv
#     # model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
#     #           norm_layer(nn.Conv2d(input_nc, opt.ngf,
#     #                                kernel_size=opt.resnet_initial_kernel_size,
#     #                                padding=0)),
#     #           activation]
#
#     # downsample
#     # mult = 1
#     # for i in range(opt.resnet_n_downsample):
#     #   model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
#     #                                  kernel_size=3, stride=2, padding=1)),
#     #             activation]
#     #   mult *= 2
#
#     # resnet blocks
#     for i in range(opt.resnet_n_blocks):
#       model += [ResnetBlock(opt.ngf * mult,
#                             norm_layer=norm_layer,
#                             activation=activation,
#                             kernel_size=opt.resnet_kernel_size)]
#
#     # upsample
#     for i in range(opt.resnet_n_downsample):
#       nc_in = int(opt.ngf * mult)
#       nc_out = int((opt.ngf * mult) / 2)
#       model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
#                                               kernel_size=3, stride=2,
#                                               padding=1, output_padding=1)),
#                 activation]
#       mult = mult // 2
#
#     # final output conv
#     model += [nn.ReflectionPad2d(3),
#               nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
#               nn.Tanh()]
#
#     self.model = nn.Sequential(*model)
#
#   def forward(self, input, z=None):
#     return self.model(input)
#
#
# class Pix2PixHDGenerator(BaseNetwork):
#   @staticmethod
#   def modify_commandline_options(parser, is_train):
#     parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
#     parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
#     parser.add_argument('--resnet_kernel_size', type=int, default=3,
#                         help='kernel size of the resnet block')
#     parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
#                         help='kernel size of the first convolution')
#     parser.set_defaults(norm_G='instance')
#     return parser
#
#   def __init__(self, opt):
#     super().__init__()
#     input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
#
#     norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
#     model = []
#
#     # initial conv
#     model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
#               norm_layer(nn.Conv2d(input_nc, opt.ngf, kernel_size=opt.resnet_initial_kernel_size, padding=0)),
#               activation]
#
#     # downsample
#     mult = 1
#     for i in range(opt.resnet_n_downsample):
#       model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
#                                      kernel_size=3, stride=2, padding=1)),
#                 activation]
#       mult *= 2
#
#     # resnet blocks
#     for i in range(opt.resnet_n_blocks):
#       model += [ResnetBlock(opt.ngf * mult,
#                             norm_layer=norm_layer,
#                             activation=activation,
#                             kernel_size=opt.resnet_kernel_size)]
#
#     # upsample
#     for i in range(opt.resnet_n_downsample):
#       nc_in = int(opt.ngf * mult)
#       nc_out = int((opt.ngf * mult) / 2)
#       model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
#                                               kernel_size=3, stride=2,
#                                               padding=1, output_padding=1)),
#                 activation]
#       mult = mult // 2
#
#     # final output conv
#     model += [nn.ReflectionPad2d(3),
#               nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
#               nn.Tanh()]
#
#     self.model = nn.Sequential(*model)
#
#   def forward(self, input, z=None):
#     return self.model(input)
