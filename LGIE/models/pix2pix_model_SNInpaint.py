# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from models.networks.generator import CA_NET
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable



# ############## Text2Image Encoder-Decoder #######

class RNN_ENCODER(nn.Module):
  def __init__(self, ntoken, ninput=300, drop_prob=0.5,
               nhidden=128, nlayers=1, bidirectional=True):
    super(RNN_ENCODER, self).__init__()
    self.n_steps = 20                   # todo: 这里需要改吗
    self.ntoken = ntoken                # size of the dictionary
    self.ninput = ninput                # size of each embedding vector
    self.drop_prob = drop_prob          # probability of an element to be zeroed
    self.nlayers = nlayers              # Number of recurrent layers
    self.bidirectional = bidirectional
    self.rnn_type = 'LSTM'
    if bidirectional:
      self.num_directions = 2
    else:
      self.num_directions = 1
    # number of features in the hidden state
    self.nhidden = nhidden // self.num_directions

    self.define_module()
    self.init_weights()

  def define_module(self):
    self.encoder = nn.Embedding(self.ntoken, self.ninput)
    self.drop = nn.Dropout(self.drop_prob)
    if self.rnn_type == 'LSTM':
      # dropout: If non-zero, introduces a dropout layer on
      # the outputs of each RNN layer except the last layer
      self.rnn = nn.LSTM(self.ninput, self.nhidden,
                         self.nlayers, batch_first=True,
                         dropout=self.drop_prob,
                         bidirectional=self.bidirectional)
    elif self.rnn_type == 'GRU':
      self.rnn = nn.GRU(self.ninput, self.nhidden,
                        self.nlayers, batch_first=True,
                        dropout=self.drop_prob,
                        bidirectional=self.bidirectional)
    else:
      raise NotImplementedError

  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)

  def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
      return (Variable(weight.new(self.nlayers * self.num_directions,
                                  bsz, self.nhidden).zero_()),
              Variable(weight.new(self.nlayers * self.num_directions,
                                  bsz, self.nhidden).zero_()))
    else:
      return Variable(weight.new(self.nlayers * self.num_directions,
                                 bsz, self.nhidden).zero_())

  def post_process_words(self, words_emb, max_len):
    batch_size, cur_len = words_emb.size(0), words_emb.size(2)
    new_words_emb = Variable(torch.zeros(batch_size, max_len, self.nhidden * self.num_directions))
    new_words_emb = new_words_emb.cuda()
    new_words_emb[:, :, :cur_len] = words_emb

    return new_words_emb

  def forward(self, captions, cap_lens, max_len, mask=None):
    # for multi-gpu
    self.rnn.flatten_parameters()

    batch_size = captions.size(0)
    hidden = self.init_hidden(batch_size)

    # input: torch.LongTensor of size batch x n_steps
    # --> emb: batch x n_steps x ninput
    emb = self.drop(self.encoder(captions))
    #
    # Returns: a PackedSequence object
    cap_lens = cap_lens.data.tolist()
    emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
    # #hidden and memory (num_layers * num_directions, batch, hidden_size):
    # tensor containing the initial hidden state for each element in batch.
    # #output (batch, seq_len, hidden_size * num_directions)
    # #or a PackedSequence object:
    # tensor containing output features (h_t) from the last layer of RNN
    output, hidden = self.rnn(emb, hidden)
    # PackedSequence object
    # --> (batch, seq_len, hidden_size * num_directions)
    output = pad_packed_sequence(output, batch_first=True)[0]
    # output = self.drop(output)

    # --> batch x hidden_size*num_directions x seq_len
    words_emb = output    #.transpose(1, 2)      # 这里暂时不要转置，保持(batch, seq_len, lang_dim)
    # --> batch x num_directions*hidden_size

    if self.rnn_type == 'LSTM':
      sent_emb = hidden[0].transpose(0, 1).contiguous()
    else:
      sent_emb = hidden.transpose(0, 1).contiguous()
    sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

    # todo: 为什么这里要处理！变长和定长到底有什么不同！
    # words_emb = self.post_process_words(words_emb, max_len)

    # 把emb也输出出去
    emb = pad_packed_sequence(emb, batch_first=True)[0]
    return words_emb, sent_emb, emb


class Pix2PixModel(torch.nn.Module):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    networks.modify_commandline_options(parser, is_train)
    return parser

  def __init__(self, opt, dataset):
    super().__init__()
    self.opt = opt
    self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
    self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

    self.netG, self.netD, self.netE = self.initialize_networks(opt)

    # add by jwt
    self.for_D = {}
    self.noise = torch.FloatTensor(opt.batchSize, opt.ca_condition_dim).cuda()
    if self.opt.lang_encoder == 'bilstm':
      self.n_words = dataset.n_words
      self.ixtoword = dataset.ixtoword
      self.text_encoder = RNN_ENCODER(self.n_words, nhidden=opt.lang_dim, ninput=opt.lang_dim).cuda()
      if not opt.isTrain or opt.continue_train:
        self.text_encoder = util.load_network(self.text_encoder, 'text_encoder', opt.which_epoch, opt)

      # state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
      # self.text_encoder.load_state_dict(state_dict)
      # todo: 我们要train它
      # for p in self.text_encoder.parameters():
      #   p.requires_grad = False
      # print('Load text encoder from:', cfg.TRAIN.NET_E)


    # set loss functions
    if opt.isTrain:
      if opt.netD == 'InpaintSA':
        self.criterionGAN = networks.SNLoss(weight=1, opt=opt)
      else:
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
      self.criterionFeat = torch.nn.L1Loss()
      self.criterionL1 = torch.nn.L1Loss()
      if not opt.no_vgg_loss:
        self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
      if opt.use_vae:
        self.KLDLoss = networks.KLDLoss()
      if opt.ca_condition_dim:    # if 0, then don't use ca
        self.KLDLoss = networks.KLDLoss()
        self.ca_net = CA_NET(opt)

      # follow the setting of SA
      # self.netG.train()
      # self.netD.train()


  def forward(self, data, mode):
    # 如果为None则代表不要
    if not self.opt.FiveK:
      if self.opt.use_pattn:
        input_image, output_image, captions, cap_lens, input_semantics = self.preprocess_input_for_me(data)
      else:
        input_image, output_image, captions, cap_lens = self.preprocess_input_for_me(data)
      max_len = int(torch.max(cap_lens))

      if mode == 'generator':
        context_embs, sent_emb, words_embs = self.text_encoder(captions, cap_lens, max_len)
        g_loss, generated = self.compute_generator_loss_my(
          input_image=input_image, output_image=output_image, context_embs=context_embs, sent_emb=sent_emb,
          words_embs=words_embs, input_semantics=input_semantics)

        return g_loss, generated

      elif mode == 'discriminator':
        context_embs, sent_emb, words_embs = self.text_encoder(captions, cap_lens, max_len)
        d_loss = self.compute_discriminator_loss_my(input_image=input_image, output_image=output_image, context_embs=context_embs, sent_emb=sent_emb,
          words_embs=words_embs, input_semantics=input_semantics)
        return d_loss

      elif mode == 'inference':
        with torch.no_grad():
          context_embs, sent_emb, words_embs = self.text_encoder(captions, cap_lens, max_len)
          if self.opt.use_pattn:
            fake_image, op_attns = self.generate_fake_my(input_semantics=input_semantics, input_image=input_image,
                                                          sent_emb=sent_emb, words_embs=words_embs, context_embs=context_embs)
            return fake_image.detach(), op_attns.detach()


    else:   # FiveK
      input_image = data['input_img'].cuda()
      output_image = data['output_img'].cuda()
      captions = data['caption'].cuda()
      cap_lens = data['caption_len'].cuda()
      max_len = int(torch.max(cap_lens))
      words_embs, sent_emb = self.text_encoder(captions, cap_lens, max_len)

      if mode == 'generator':
        g_loss, generated = self.compute_generator_loss_my(
          input_image=input_image, output_image=output_image, sent_emb=sent_emb)
        return g_loss, generated
      elif mode == 'discriminator':
        d_loss = self.compute_discriminator_loss_my(
          input_image=input_image, output_image=output_image, sent_emb=sent_emb)
        return d_loss
      elif mode == 'inference':
        fake_image = self.netG(input_image=input_image, sent_emb=sent_emb)
        return fake_image

  def create_optimizers(self, opt):
    G_params = list(self.netG.parameters())
    if opt.use_vae:
      G_params += list(self.netE.parameters())
    if opt.isTrain and (opt.lambda_gan or opt.lambda_gan_uncond):
      D_params = list(self.netD.parameters())

    beta1, beta2 = opt.beta1, opt.beta2
    if opt.no_TTUR:
      G_lr, D_lr = opt.lr, opt.lr
    else:
      G_lr, D_lr = opt.lr / 2, opt.lr * 2

    optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2)) if opt.isTrain and (opt.lambda_gan or opt.lambda_gan_uncond) else None

    return optimizer_G, optimizer_D

  def save(self, epoch, opt):
    if self.opt.lang_encoder == 'bilstm':
      util.save_network(self.text_encoder, 'text_encoder', epoch, self.opt)
    util.save_network(self.netG, 'G', epoch, self.opt)
    if opt.isTrain and (opt.lambda_gan or opt.lambda_gan_uncond):
      util.save_network(self.netD, 'D', epoch, self.opt)
    if self.opt.use_vae:
      util.save_network(self.netE, 'E', epoch, self.opt)

  ############################################################################
  # Private helper methods
  ############################################################################

  def initialize_networks(self, opt):
    netG = networks.define_G(opt)
    netD = networks.define_D(opt) if opt.isTrain and (opt.lambda_gan or opt.lambda_gan_uncond) else None
    netE = networks.define_E(opt) if opt.use_vae else None

    if not opt.isTrain or opt.continue_train:
      netG = util.load_network(netG, 'G', opt.which_epoch, opt)
      netG = netG.cuda()
      if opt.isTrain and (opt.lambda_gan or opt.lambda_gan_uncond):
        netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        netD = netD.cuda()
      if opt.use_vae:
        netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        netE = netE.cuda()

    return netG, netD, netE

  def preprocess_input_for_me(self, data):
    # move to GPU and change data types
    if not self.opt.FiveK:
      data['label_list'] = [label.long().cuda() for label in data['label_list']]

      # create multi-hot label map
      label_list = data['label_list']                                   # label_map: (bs, 1, h, w)
      bs, _, h, w = label_list[0].size()
      nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
      input_semantics = self.FloatTensor(bs, nc, h, w).zero_()
      # print(f'label_list len: {len(label_list)}')
      for label in label_list:
        # 注意！因为设置了dontcarelabel，所以multihot vector第一维(0)有时会被赋为1
        input_semantics.scatter_(1, label, 1.0)   # input_semantics: (bs, nc, h, w) 变成0, 1
      # 取消第0维被赋上的1（mask里的0会被放到input_semantics的第一维）
      input_semantics[:, 0, :, :] = 0

      # # 在这里给semantic加上caption_embed，当lang_dim=0时表示不使用lang_dim
      # if self.opt.lang_dim and (not self.opt.ca_condition_dim):
      #   repeat_cap = data['caption_embed'].unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
      #   input_semantics = torch.cat([input_semantics, repeat_cap], dim=1)   # (bs, nc+lang_dim, h, w)

      # concatenate instance map if it exists
      if not self.opt.no_instance:
        inst_map = data['instance']
        instance_edge_map = self.get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

      return data['input_img'].cuda(), data['output_img'].cuda(), data['caption'].cuda(),\
             data['caption_len'].cuda(), input_semantics.cuda(),

    else:
      # elif self.opt.lang_encoder == 'bilstm':
      #   # sort data by the length in a decreasing order
      #   # sorted_cap_lens, sorted_cap_indices = torch.sort(data['caption_len'], 0, True)
      #   # captions = data['caption'][sorted_cap_indices].squeeze()
      return data['input_img'], data['output_img'], data['caption'], data['caption_len']


  def compute_generator_loss_my(self, **input_G):
    G_losses = {}
    fake_image, ret_dict = self.netG(input_semantics=input_G['input_semantics'], input_image=input_G['input_image'],
                                         sent_emb=input_G['sent_emb'], words_embs=input_G['words_embs'], context_embs=input_G['context_embs'])

    if self.opt.lambda_gan > 0:
      neg_imgs = torch.cat([fake_image, ret_dict['masks']], dim=1)
      pred_neg = self.netD(neg_imgs)
      G_losses['GAN-SN-Patch'] = self.criterionGAN(neg=pred_neg) * self.opt.lambda_gan

    if self.opt.lambda_gan_uncond > 0:
      pred_fake, pred_real = self.discriminate(None, fake_image=fake_image, real_image=input_G['output_image'])
      G_losses['GAN_uncond'] = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan_uncond

    # 找到multihot vector为全0的pixel
    # if self.opt.lambda_unchange:
    #   unchange_mask = (torch.max(input_semantics, dim=1, keepdim=True)[0] == 0).float()
    #   G_losses['Unchange'] = self.opt.lambda_unchange * self.criterionL1(fake_image * unchange_mask, output_image * unchange_mask)

    if self.opt.lambda_feat and (not self.opt.no_ganFeat_loss):
      num_D = len(pred_fake)
      GAN_Feat_loss = self.FloatTensor(1).fill_(0)
      for i in range(num_D):  # for each discriminator
        # last output is the final prediction, so we exclude it
        num_intermediate_outputs = len(pred_fake[i]) - 1
        for j in range(num_intermediate_outputs):  # for each layer output
          unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
          GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
      G_losses['GAN_Feat'] = GAN_Feat_loss
    if self.opt.lambda_vgg and (not self.opt.no_vgg_loss):
      G_losses['VGG'] = self.criterionVGG(fake_image, input_G['output_image']) * self.opt.lambda_vgg
    if self.opt.lambda_L1:
      G_losses['L1'] = self.opt.lambda_L1 * self.criterionL1(fake_image, input_G['output_image'])

    return G_losses, fake_image

  # def compute_discriminator_loss(self, input_semantics, real_image):
  #   D_losses = {}
  #   with torch.no_grad():
  #     fake_image, _ = self.generate_fake(input_semantics, real_image)
  #     fake_image = fake_image.detach()
  #     fake_image.requires_grad_()
  #
  #   pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
  #
  #   D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
  #   D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
  #
  #   return D_losses
  # def compute_discriminator_loss_my(self, input_semantics, input_image, output_image):

  def compute_discriminator_loss_my(self, **input_D):
    D_losses = {}
    # todo: 是否可以只生成一次fake_image，然后更新D和G （现在是D和G都需要生成，这里没搞清楚）可能不太行，因为D和G需要分开更新？
    with torch.no_grad():
      fake_image, ret_dict = self.netG(input_semantics=input_D['input_semantics'], input_image=input_D['input_image'],
                                       sent_emb=input_D['sent_emb'], words_embs=input_D['words_embs'], context_embs=input_D['context_embs'])

      fake_image = fake_image.detach()
      fake_image.requires_grad_()

    if self.opt.lambda_gan > 0:
      pos_imgs = torch.cat([input_D['output_image'], ret_dict['masks']], dim=1)
      neg_imgs = torch.cat([fake_image, ret_dict['masks']], dim=1)
      pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

      pred_pos_neg = self.netD(pos_neg_imgs)
      pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
      D_losses['D_SN-Patch'] = self.criterionGAN(pos=pred_pos, neg=pred_neg) * self.opt.lambda_gan

    if self.opt.lambda_gan_uncond > 0:
      pred_fake, pred_real = self.discriminate(None, fake_image, input_D['output_image'])
      D_losses['D_Fake_uncond'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.opt.lambda_gan_uncond
      D_losses['D_real_uncond'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt.lambda_gan_uncond

    return D_losses


  def generate_fake_my(self, **input_G):
    fake_image = self.netG(**input_G)
    return fake_image

  # Given fake and real image, return the prediction of discriminator
  # for each fake and real image.

  def discriminate(self, condition, fake_image, real_image):
    if self.opt.FiveK:
      condition = condition.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fake_image.size(2), fake_image.size(3))
    elif self.opt.lang_encoder == 'bilstm':
      if not self.opt.use_pattn:
        condition = condition.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fake_image.size(2), fake_image.size(3))
    # todo: D需要接受原图
    if condition is not None:
      if self.opt.three_input_D:
        fake_example = torch.cat([condition, real_image, fake_image], dim=1)   # (5, 3+3+65, h, w)
        real_example = torch.cat([condition, real_image, real_image], dim=1)
      else:
        fake_example = torch.cat([condition, fake_image], dim=1)  # (5, 3+3+65, h, w)
        real_example = torch.cat([condition, real_image], dim=1)
    else:
      fake_example = fake_image
      real_example = real_image
    # In Batch Normalization, the fake and real images are
    # recommended to be in the same batch to avoid disparate
    # statistics in fake and real images.
    # So both fake and real images are fed to D all at once.
    fake_and_real = torch.cat([fake_example, real_example], dim=0)

    discriminator_out = self.netD(fake_and_real)

    pred_fake, pred_real = self.divide_pred(discriminator_out)

    return pred_fake, pred_real

  # Take the prediction of fake and real images from the combined batch
  def divide_pred(self, pred):
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
      fake = []
      real = []
      for p in pred:
        fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
        real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
      fake = pred[:pred.size(0) // 2]
      real = pred[pred.size(0) // 2:]

    return fake, real

  def get_edges(self, t):
    edge = self.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

  def use_gpu(self):
    return len(self.opt.gpu_ids) > 0
