# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from cocoapi.PythonAPI.pycocotools import mask as cocomask
import os.path
import torch
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import get_params, get_transform
from PIL import Image
# 解决image file is truncated (41 bytes not processed)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import util.util as util
import json
import pickle
from util.util_jwt import op2ind
import numpy as np
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


# from fastNLP.embeddings import StaticEmbedding
# from fastNLP import Vocabulary


class FiveKDataset(Pix2pixDataset):
  @staticmethod
  def modify_commandline_options(parser, is_train):
    parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
    parser.add_argument('--coco_no_portraits', action='store_true')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='path to the directory that contains label images')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='path to the directory that contains photo images')
    parser.add_argument('--lang_dim', type=int, required=True,
                        help='dim of language embedding')
    parser.set_defaults(preprocess_mode='resize_and_crop')
    if is_train:
      parser.set_defaults(load_size=286)
    else:
      parser.set_defaults(load_size=256)
    parser.set_defaults(crop_size=256)
    parser.set_defaults(display_winsize=256)
    parser.set_defaults(label_nc=182)
    parser.set_defaults(contain_dontcare_label=True)
    parser.set_defaults(cache_filelist_read=True)
    parser.set_defaults(cache_filelist_write=True)
    return parser

  def initialize(self, opt, mode='train'):
    self.opt = opt
    self.opt.FiveK = True
    self.single_test = True if self.opt.input_path != 'None' else False

    if not self.single_test:
      with open(opt.anno_path, 'r') as f:
        self.anno_list = json.load(f)  # 读入新json
      if not self.opt.FiveK:
        self.masks_dir = self.opt.label_dir
      self.images_dir = self.opt.image_dir
      self.dataset_size = len(self.anno_list)
      # max_len = 0     # max_len = 43
    else:
      self.dataset_size = 1

    # json_name = opt.anno_path.split('/')[-1].split('.')[0]
    # filepath = f'{json_name}_captions.pickle'
    filepath = opt.anno_path.replace('splits', 'captions').replace('.json', '.pkl')   # ???
    # print(" *" *10)
    # print(self.single_test)
    # print(filepath)
    print('filepath:  ', filepath)
    if os.path.exists(filepath):
      with open(filepath, 'rb') as f:
        x = pickle.load(f)
        if not self.single_test:
          self.captions = x[0]

        self.ixtoword, self.wordtoix = x[1], x[2]
        del x
        self.n_words = len(self.ixtoword)
        print('Load from: ', filepath)
    else:
      all_captions = self.load_captions(self.anno_list)
      self.captions, self.ixtoword, self.wordtoix, self.n_words = self.build_dictionary(all_captions)
      with open(filepath, 'wb') as f:
        pickle.dump([self.captions, self.ixtoword, self.wordtoix], f, protocol=2)
        print('Save to: ', filepath)

    # 这里要注意，test和train的wordtoix得一样，要不然就完全不对了
    # test_num = int(0.2 * self.dataset_size)
    # if mode == 'train':
    #   self.anno_list = self.anno_list[test_num:]
    #   self.dataset_size = self.dataset_size - test_num
    #   self.captions = self.captions[test_num:]
    # else:
    #   self.anno_list = self.anno_list[:test_num]
    #   self.dataset_size = test_num
    #   self.captions = self.captions[:test_num]

  def load_captions(self, anno_list):
    all_captions = []
    for anno in anno_list:
      # cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
      # with open(cap_path, "r") as f:
      cap = anno['request']

      cap = cap.replace("\ufffd\ufffd", " ")
      # picks out sequences of alphanumeric characters as tokens
      # and drops everything else
      tokenizer = RegexpTokenizer(r'\w+')
      tokens = tokenizer.tokenize(cap.lower())
      # print('tokens', tokens)
      if len(tokens) == 0:
        print('cap', cap)
        continue

      tokens_new = []
      for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
          tokens_new.append(t)
      all_captions.append(tokens_new)
    return all_captions

  def build_dictionary(self, captions):
    word_counts = defaultdict(float)
    for sent in captions:
      for word in sent:
        word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    captions_new = []
    for t in captions:
      rev = []
      for w in t:
        if w in wordtoix:
          rev.append(wordtoix[w])
      # rev.append(0)  # do not need '<end>' token
      captions_new.append(rev)

    return [captions_new, ixtoword, wordtoix, len(ixtoword)]

  def get_paths(self, opt):
    root = opt.dataroot
    phase = 'val' if opt.phase == 'test' else opt.phase

    label_dir = os.path.join(root, '%s_label' % phase)
    label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

    if not opt.coco_no_portraits and opt.isTrain:
      label_portrait_dir = os.path.join(root, '%s_label_portrait' % phase)
      if os.path.isdir(label_portrait_dir):
        label_portrait_paths = make_dataset(label_portrait_dir, recursive=False, read_cache=True)
        label_paths += label_portrait_paths

    image_dir = os.path.join(root, '%s_img' % phase)
    image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

    if not opt.coco_no_portraits and opt.isTrain:
      image_portrait_dir = os.path.join(root, '%s_img_portrait' % phase)
      if os.path.isdir(image_portrait_dir):
        image_portrait_paths = make_dataset(image_portrait_dir, recursive=False, read_cache=True)
        image_paths += image_portrait_paths

    if not opt.no_instance:
      instance_dir = os.path.join(root, '%s_inst' % phase)
      instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

      if not opt.coco_no_portraits and opt.isTrain:
        instance_portrait_dir = os.path.join(root, '%s_inst_portrait' % phase)
        if os.path.isdir(instance_portrait_dir):
          instance_portrait_paths = make_dataset(instance_portrait_dir, recursive=False, read_cache=True)
          instance_paths += instance_portrait_paths

    else:
      instance_paths = []

    return label_paths, image_paths, instance_paths

  def __getitem__(self, index):
    # get images
    if not self.single_test:
      anno = self.anno_list[index]
      input_imname = os.path.join(self.images_dir, anno['input'].replace('/', '_'))
      output_imname = os.path.join(self.images_dir, anno['output'].replace('/', '_'))
    else:
      # 00001225d3b3.jpg, farm1_255_19645548243_8513acc505_b.jpg
      # print("*" * 5, "The image is fix now !!!", "*" * 5)
      input_imname = self.opt.input_path
      output_imname = input_imname

    input_img = Image.open(input_imname).convert('RGB')
    output_img = Image.open(output_imname).convert('RGB')

    # darken part of the input image
    # input_img = np.asarray(input_img)
    # edited_input_img = input_img.copy()
    # for i in range(edited_input_img.shape[0]//2):
    #   for j in range(edited_input_img.shape[1]//2):
    #     edited_input_img[i, j] = edited_input_img[i, j] * 0.85
    # edited_input_img = edited_input_img // 2
    # input_img = Image.fromarray(np.uint8(edited_input_img))

    # todo: check一下数据的预处理，包括crop和resize
    params = get_params(self.opt, input_img.size)  # input_img.size 生图大小
    transform_image = get_transform(self.opt, params)
    input_img_tensor = transform_image(input_img)
    output_img_tensor = transform_image(output_img)
    input_dict = {
      'input_img': input_img_tensor,
      'output_img': output_img_tensor,
    }

    # caption = anno['request']
    # words = torch.LongTensor([self.vocab.to_index(word) for word in caption.split()])  # 将文本转为index
    # caption_embed = torch.mean(self.embed(words), dim=0)  # (50)
    # input_dict['caption_embed'] = caption_embed

    if not self.single_test:
      # **************** for usual training ****************
      input_dict['caption'] = torch.LongTensor(self.captions[index])
      input_dict['caption_len'] = (input_dict['caption'] > 0).sum()
      input_dict['uid'] = torch.tensor(index)
    else:
      # **************** for one caption ****************
      # print("*" * 5, "The caption is fix now !!!", "*" * 5)
      # fix_cap = ['brighten the image a little bit', 'brighten the image a lot', 'add blue tone to the image',
      #            'change the orange color to blue in the whole image',
      #            'brighten the image to make the people visble']
      # fix_cap = ['brighten the image', 'brighten the image a little bit', 'brighten the image a lot', 'brighten the image to make the people visble',
      #            'darken the background then brighten the people to make the face visible']
      # fix_cap = fix_cap[index % len(fix_cap)]

      fix_cap = self.opt.request
      caption = [self.wordtoix[word] if word in self.wordtoix else self.wordtoix['the'] for word in fix_cap.split()]
      input_dict['caption'] = torch.LongTensor(caption)
      input_dict['caption_len'] = len(input_dict['caption'])

    # Give subclasses a chance to modify the final output
    self.postprocess(input_dict)
    return input_dict

  def postprocess(self, input_dict):
    # for key in input_dict:
    #     if isinstance(input_dict[key], list):
    #         for i in range(len(input_dict[key])):
    #             input_dict[key][i] = input_dict[key][i].cuda()
    #     else:
    #         input_dict[key] = input_dict[key].cuda()
    return input_dict
