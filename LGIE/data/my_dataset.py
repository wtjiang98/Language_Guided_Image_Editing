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
import random


# from fastNLP.embeddings import StaticEmbedding
# from fastNLP import Vocabulary


class MyDataset(Pix2pixDataset):

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
    self.single_test = True if self.opt.input_path != 'None' else False
    self.opt.FiveK = False

    if not self.single_test:
      with open(opt.anno_path, 'r') as f:
        self.anno_list = json.load(f)  # 读入新json
      if not self.opt.FiveK:
        self.masks_dir = self.opt.label_dir
      self.images_dir = self.opt.image_dir
      self.dataset_size = len(self.anno_list)
    else:
      self.dataset_size = 1

    # max_len = 0     # max_len = 43
    if self.opt.lang_dim:
      if opt.lang_encoder == 'bilstm':
        json_name = opt.anno_path.split('/')[-1].split('.')[0]
        filepath = f'{json_name}_captions.pickle'
        if opt.all_request:
          filepath = 'all_request_' + filepath
        # todo: use all request here
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
          all_captions = self.load_captions(self.anno_list, opt.all_request)
          self.captions, self.ixtoword, self.wordtoix, self.n_words = self.build_dictionary(all_captions)
          with open(filepath, 'wb') as f:
            pickle.dump([self.captions, self.ixtoword, self.wordtoix], f, protocol=2)
            print('Save to: ', filepath)

    # 这里要注意，test和train的wordtoix得一样，要不然就完全不对了
    if not self.single_test:
      test_num = int(0.2 * self.dataset_size)
      if mode == 'train':
        self.anno_list = self.anno_list[test_num:]
        self.dataset_size = self.dataset_size - test_num
        self.captions = self.captions[test_num:]
      else:
        # self.anno_list = self.anno_list[:test_num]
        # self.dataset_size = test_num
        # self.captions = self.captions[:test_num]
        self.anno_list = self.anno_list[test_num:]
        self.dataset_size = self.dataset_size - test_num
        self.captions = self.captions[test_num:]

  def load_captions(self, anno_list, all_request):

    def get_tokens(cap):
      cap = cap.replace("\ufffd\ufffd", " ")
      # picks out sequences of alphanumeric characters as tokens
      # and drops everything else
      tokenizer = RegexpTokenizer(r'\w+')
      tokens = tokenizer.tokenize(cap.lower())
      assert len(tokens) > 0, "cap should not be empty"
      # if len(tokens) == 0:
      #   print('cap', cap)
      #   continue
      tokens_new = []
      for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
          tokens_new.append(t)
      return tokens_new

    all_captions = []
    for anno in anno_list:
      # cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
      # with open(cap_path, "r") as f:
      if not self.opt.FiveK:
        if not all_request:
          cap = anno['expert_summary'][0] if anno['expert_summary'] else anno['amateur_summary'][0]
        else:
          cap = anno['expert_summary'] + anno['amateur_summary']
      else:
        cap = anno['request']

      if isinstance(cap, list):
        tokens_list = [get_tokens(cur_cap) for cur_cap in cap]
        all_captions.append(tokens_list)
      else:
        all_captions.append(get_tokens(cap))

    return all_captions

  def build_dictionary(self, captions):
    word_counts = defaultdict(float)
    for sent in captions:
      if isinstance(sent, list):
        for cur_sent in sent:
          for word in cur_sent:
            word_counts[word] += 1
      else:
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
      if isinstance(t, list):
        rev = []
        for cur_t in t:
          cur_rev = []
          for w in cur_t:
            if w in wordtoix:
              cur_rev.append(wordtoix[w])
          rev.append(cur_rev)
        # rev.append(0)  # do not need '<end>' token
        captions_new.append(rev)
      else:
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
      # for testing single image:
      # 00001225d3b3.jpg, farm1_255_19645548243_8513acc505_b.jpg
      input_imname = self.opt.input_path
      output_imname = input_imname

    input_img = Image.open(input_imname)
    output_img = Image.open(output_imname)
    input_img = input_img.convert('RGB')
    output_img = output_img.convert('RGB')

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

    if not self.single_test:
      # produce label: 生成最终输入的label (h, w)，里面的每一个元素是op的编号
      im_name = input_imname.split('_')[-1].split('.')[0]
      with open(os.path.join(self.masks_dir, f'{im_name}_{im_name}_mask.json'), 'r') as f:
        mask_list = json.load(f)
      label_tensor_list = []
      for id, op_name in enumerate(anno['operator']):
        op_ind = op2ind[op_name]
        assert op_ind != 0, "需要把op的id从1起，因为后面要乘以op_ind"
        if anno['operator'][op_name]['local'] is True:
          if anno['operator'][op_name]['ids']:  # 这里的ids可能为空，为空则表示全图
            for obj_id in anno['operator'][op_name]['ids']:  # op可能有多个操作对象
              rleObj = mask_list[obj_id]
              mask = cocomask.decode(rleObj) * op_ind  # 把mask乘上ind
              label_tensor_list.append(mask)
          else:
            label_tensor_list.append(np.ones((input_img.height, input_img.width)) * op_ind)  # 把mask乘上ind
        else:
          label_tensor_list.append(np.ones((input_img.height, input_img.width)) * op_ind)  # 把mask乘上ind

      assert label_tensor_list, 'label_tensor_list should not be empty!'

      transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
      # 这里犯了一个重大错误：如果把label当做PIL图片处理（方便Resize，Crop），则Totensor操作会将其scale到[0, 1]
      # 但是在全图为某个数的情况下，又不会scale，故先转PIL为ndarray再转Tensor
      # label_tensor_list = [transform_label(Image.fromarray(label)) for label in label_tensor_list] # (1, 256, 256)
      # 注意！这里面的tensor要换成byte类型才能在多卡时正常跑（为何？）
      label_tensor_list = [torch.from_numpy(np.array(transform_label(Image.fromarray(label)))).unsqueeze(0).byte()
                           for label in label_tensor_list]
      # print(f'cur label_list len: {len(label_tensor_list)}')
      input_dict['label_list'] = label_tensor_list

    else:   # single_test
      transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
      label_tensor_list = [np.ones((input_img.height, input_img.width)) * 2]    # 2代表一个全图的retouching
      label_tensor_list = [torch.from_numpy(np.array(transform_label(Image.fromarray(label)))).unsqueeze(0).byte()
                           for label in label_tensor_list]
      input_dict['label_list'] = label_tensor_list

    # end use_op

    if not self.single_test:
      # ***** for usual training *****
      if self.opt.all_request:
        cap_idx = random.randint(0, len(self.captions[index]) - 1)
        input_dict['caption'] = torch.LongTensor(self.captions[index][cap_idx])
      else:
        input_dict['caption'] = torch.LongTensor(self.captions[index])
      input_dict['caption_len'] = len(input_dict['caption'])

    else:
      fix_cap = self.opt.request
      caption = [self.wordtoix[word] if word in self.wordtoix else self.wordtoix['the'] for word in fix_cap.split()]
      input_dict['caption'] = torch.LongTensor(caption)
      input_dict['caption_len'] = len(input_dict['caption'])
      # fix_cap = 'darken the photo'
      # fix_cap = 'brighten the image a lot'
      # fix_cap = 'add blue tone to the image'
      # fix_cap = 'change the orange color to blue in the whole image'
      # caption = [self.wordtoix[word] for word in fix_cap.split()]
      # input_dict['caption'] = torch.LongTensor(caption)
      # input_dict['caption_len'] = len(input_dict['caption'])

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
