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
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
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
    cxt_scores = self.fc(context) # (batch, seq_len, phrase_num)
    attn = F.softmax(cxt_scores, dim=1)  # (batch, seq_len, phrase_num), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (context[:, :, 0] != 0).float().unsqueeze(-1) # (batch, seq_len, 1)
    attn = attn * is_not_zero # (batch, seq_len, phrase_num)
    # (batch, seq_len, phrase_num) # 太妙了，用mask得到该有的word之后再归一化到1
    attn = attn / attn.sum(1).view(attn.size(0), 1, attn.size(2)).expand(attn.size(0), attn.size(1), attn.size(2))

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
      self.phrase_num = 2
      self.aug_pattn = AugPhraseAttention(input_dim=self.opt.lang_dim, phrase_num=self.phrase_num)
      self.op_weight_fc = nn.Linear(self.opt.lang_dim, self.phrase_num)

    if self.opt.use_buattn:
      self.buattn = GlobalBUAttentionGeneral(opt.lang_dim, opt.lang_dim, opt.buattn_norm)

    # ****************** predict filter parameter ******************
    if self.opt.predict_param:
      self.resnet34 = models.resnet34(pretrained=False, num_classes=opt.label_nc + opt.contain_dontcare_label)

    # ****************** generate ******************

    if self.opt.encoder_nospade:
      kw = 3
      pw = int(np.ceil((kw - 1.0) / 2))
      norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
      self.layer1 = norm_layer(nn.Conv2d(3, nf, kw, stride=2, padding=pw))
      self.layer2 = norm_layer(nn.Conv2d(nf * 1, nf * 2, kw, stride=2, padding=pw))
      self.layer3 = norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw))
      # self.layer4 = norm_layer(nn.Conv2d(nf * 4, nf * 8, kw, stride=2, padding=pw))
      # self.layer5 = norm_layer(nn.Conv2d(nf * 8, nf * 8, kw, stride=2, padding=pw))
      # if opt.crop_size >= 256:
      #   self.layer6 = norm_layer(nn.Conv2d(nf * 4, nf * 4, kw, stride=2, padding=pw))

      # self.so = s0 = 4
      # self.fc_mu = nn.Linear(nf * 8 * s0 * s0, 256)
      # self.fc_var = nn.Linear(nf * 8 * s0 * s0, 256)

      self.actvn = nn.LeakyReLU(0.2, False)
    else:
      self.conv_embed = nn.Conv2d(3, nf, 3, padding=1)
      self.down_0 = SPADEResnetBlock(1 * nf, 2 * nf, opt)
      self.down_1 = SPADEResnetBlock(2 * nf, 4 * nf, opt)
      self.down = nn.Upsample(scale_factor=0.5)

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
    if self.opt.use_pattn:
      input_semantics = input_G['input_semantics']
      words_embs = input_G['words_embs']
      sent_emb = input_G['sent_emb']
      context_embs = input_G['context_embs']

      # start phrase attn
      # inpaint_attn, inpaint_phrase_emb = self.inpaint_att(context_embs, words_embs) # (b, lang_dim)
      # retouch_attn, retouch_phrase_emb = self.retouch_att(context_embs, words_embs)
      op_attns, op_embs = self.aug_pattn(context_embs, words_embs)         # (b, phrase_num, lang_dim)
      op_weights = F.softmax(self.op_weight_fc(sent_emb), dim=1)  # (b, phrase_num)

      seg = torch.zeros((input_image.size(0), self.opt.lang_dim, input_image.size(2), input_image.size(3))).cuda()
      for b in range(input_image.size(0)):
        for op_idx in range(1, self.opt.label_nc + self.opt.contain_dontcare_label):  # 因为第一维是废的
          # # input semantics是从1开始的
          row_idx, col_idx = torch.nonzero(input_semantics[b, op_idx, :, :], as_tuple=True)
          # # embs是从0开始的
          # seg[b, :, row_idx, col_idx] = (op_embs[b, op_idx-1] * op_weights[b, op_idx-1]).unsqueeze(-1)
          if op_idx != 1:  # for retouch opertor
            seg[b, :, row_idx, col_idx] = seg[b, :, row_idx, col_idx] + (op_embs[b, 0] * op_weights[b, 0]).unsqueeze(-1)
          else:  # for inpaint operator
            seg[b, :, row_idx, col_idx] = seg[b, :, row_idx, col_idx] + (op_embs[b, 1] * op_weights[b, 1]).unsqueeze(-1)

    else:
      seg = input_G['sent_emb']
      seg = seg.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input_image.size(2), input_image.size(3))    # (b, nc, h, w)

    # if self.opt.FiveK:
    #   # seg = self.embed_to_op(seg)
    #   # 把seg复制到全图大小
    #   seg = seg.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input_image.size(2), input_image.size(3))    # (b, nc, h, w)

    # predict paramter
    if self.opt.predict_param:
      parameter = self.resnet34(input_image)  # (1, label_nc)
      if not self.opt.FiveK:
        parameter = torch.cat([parameter, torch.ones(parameter.size(0), self.opt.lang_dim).cuda()], dim=1)  # (1, label_nc + lang_dim)
      parameter = parameter.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, seg.size(2), seg.size(3))  # (1, label_nc, 256, 256)
      seg = seg * parameter

    # if self.opt.use_buattn:
    #     # todo: 让word跟label的word embed做attention，需要使用并行计算加速！
    #     glove_words_embed = glove_words_embed.permute(0, 2, 1)  # (1, lang_dim, caption_len)
    #     labels_embed = labels_embed.permute(0, 2, 1).unsqueeze(-1)
    #     weightedContext, attn = self.buattn(labels_embed, glove_words_embed, glove_words_embed) # (batch, idf, queryL) queryL = label_nc
    #
    #     for batch in range(seg.size(0)):
    #         for h in range(seg.size(2)):
    #             for w in range(seg.size(3)):
    #                 sum_embed = input_semantics[batch, 1:self.opt.label_nc + self.opt.contain_dontcare_label, h, w] *\
    #                             weightedContext.squeeze(-1)[0]
    #                 seg[batch, self.opt.label_nc + self.opt.contain_dontcare_label:, h, w] = sum_embed.sum(dim=1)
    if self.opt.encoder_nospade:
      x_down_0 = self.layer1(input_image)
      x_down_1 = self.layer2(self.actvn(x_down_0))
      x_down_2 = self.layer3(self.actvn(x_down_1))
      x_down_3 = self.actvn(x_down_2)
    else:
      x_conv_embed = self.conv_embed(F.leaky_relu(input_image, 2e-1))    # (1, nf, 256, 256)
      # start generate
      x_down_0 = self.down(x_conv_embed)                # (1, nf, 128, 128)
      x_down_1 = self.down_0(x_down_0, seg)             # (1, 2 * nf, 128, 128)
      x_down_2 = self.down(x_down_1)                    # (1, 2 * nf, 64, 64)
      x_down_3 = self.down_1(x_down_2, seg)             # (1, 4 * nf, 64, 64)

    x_middle = self.G_middle_0(x_down_3, seg)
    x_middle = self.G_middle_1(x_middle, seg)
    x_middle = self.G_middle_2(x_middle, seg)         # (1, 4 * nf, 64, 64)

    if self.opt.skiplayer:
      x = self.up(torch.cat([x_middle, x_down_3], dim=1))    # (1, 4 * nf, 128, 128)
      x = self.up_0(x, seg)                                 # (1, 2 * nf, 128, 128)
      x = self.up(torch.cat([x, x_down_1], dim=1))          # (1, 2 * nf, 256, 256)
      x = self.up_1(x, seg)                                  # (1, nf, 256, 256)
      x = self.conv_img(F.leaky_relu(torch.cat([x, x_conv_embed], dim=1), 2e-1))        # (1, 3, 256, 256)
    else:
      x = self.up(x_middle)
      x = self.up_0(x, seg)
      x = self.up(x)
      x = self.up_1(x, seg)
      x = self.conv_img(F.leaky_relu(x, 2e-1))

    x = F.tanh(x)

    # return x, seg, [inpaint_attn, retouch_attn]
    return x, seg, [op_attns, op_weights]


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


class Pix2PixHDGenerator(BaseNetwork):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
    parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
    parser.add_argument('--resnet_kernel_size', type=int, default=3,
                        help='kernel size of the resnet block')
    parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                        help='kernel size of the first convolution')
    parser.set_defaults(norm_G='instance')
    return parser

  def __init__(self, opt):
    super().__init__()
    input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

    norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
    activation = nn.ReLU(False)

    model = []

    # initial conv
    model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
              norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                   kernel_size=opt.resnet_initial_kernel_size,
                                   padding=0)),
              activation]

    # downsample
    mult = 1
    for i in range(opt.resnet_n_downsample):
      model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                     kernel_size=3, stride=2, padding=1)),
                activation]
      mult *= 2

    # resnet blocks
    for i in range(opt.resnet_n_blocks):
      model += [ResnetBlock(opt.ngf * mult,
                            norm_layer=norm_layer,
                            activation=activation,
                            kernel_size=opt.resnet_kernel_size)]

    # upsample
    for i in range(opt.resnet_n_downsample):
      nc_in = int(opt.ngf * mult)
      nc_out = int((opt.ngf * mult) / 2)
      model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1)),
                activation]
      mult = mult // 2

    # final output conv
    model += [nn.ReflectionPad2d(3),
              nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
              nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, input, z=None):
    return self.model(input)
