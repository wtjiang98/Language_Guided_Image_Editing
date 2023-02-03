import os
import sys
import glob
import re
import json
import pdb
import base64
import h5py

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.mask import decode as mask_decode

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, '../../data/IER2'))
sys.path.append(os.path.join(cur_dir, '..'))
from data.IER_wjy import IER

'''
merget request, operator, mask label together
'''

# requires two files: request data, operator json
img_file = '../datasets/data/IER2/images'
mask_file = '../datasets/data/IER2/masks'
showind2maskind_file = '../datasets/data/IER2/show2mask'
feature_file = '../datasets/data/IER2/features'
operator_file = '../datasets/data/IER2/IER2.json'
vocab_dir = '../datasets/data/language'


class IERDataset(Dataset):
  def __init__(self, img_file, mask_file, show2mask_file, operator_file, feature_file, vocab_dir, phase, opt):
    self.opt = opt
    self.img_file = img_file
    self.session = opt.session
    self.IER = IER(mask_file, show2mask_file, operator_file, feature_file, vocab_dir, self.session)
    self.set_boxes()
    self.global_op_req_ids = self.filter_global()
    self.local_op_req_ids = self.filter_local()
    self.phase = phase
    self.train_index_local = [i for i in self.local_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.train_req_ids]
    self.train_index_global = [i for i in self.global_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.train_req_ids]
    self.val_index_local = [i for i in self.local_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.val_req_ids]
    self.val_index_global = [i for i in self.global_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.val_req_ids]
    self.test_index_local = [i for i in self.local_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.test_req_ids]
    self.test_index_global = [i for i in self.global_op_req_ids if self.IER.OpReqId2ReqId[i] in self.IER.test_req_ids]
    self.filter_req(opt.filter_req)
    self.max_length = 6
    self.n_words = len(self.IER.id2vocab)
    self.ixtoword = self.IER.id2vocab

  @property
  def vocab_size(self):
    return len(self.IER.vocab2id)

  @property
  def op_vocab_size(self):
    return len(self.IER.op_vocab2id)

  @property
  def pan_dim(self):
    return 512

  def filter_req(self, filter_req_setting):
    """filter out the requests with no valid operation"""
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/data/filter_req',
                             'filter_req_' + filter_req_setting + '.json')
    if os.path.exists(save_path):
      with open(save_path, 'r') as f:
        filter_req = json.load(f)
        self.filtered_train_req_ids = filter_req['train']
        self.filtered_val_req_ids = filter_req['val']
        self.filtered_test_req_ids = filter_req['test']
      return
    filter_req = {}
    self.filtered_train_req_ids = []
    for req_id in self.IER.train_req_ids:
      req_op_ids = self.IER.ReqId2OpReqId[req_id]
      cnt_op = 0
      for req_op_id in req_op_ids:
        if req_op_id in self.train_index_local:
          cnt_op += 1
        if req_op_id in self.train_index_global:
          cnt_op += 1
      if cnt_op > 0:
        self.filtered_train_req_ids.append(req_id)
    self.filtered_val_req_ids = []
    for req_id in self.IER.val_req_ids:
      req_op_ids = self.IER.ReqId2OpReqId[req_id]
      cnt_op = 0
      for req_op_id in req_op_ids:
        if req_op_id in self.val_index_local:
          cnt_op += 1
        if req_op_id in self.val_index_global:
          cnt_op += 1
      if cnt_op > 0:
        self.filtered_val_req_ids.append(req_id)
    self.filtered_test_req_ids = []
    for req_id in self.IER.test_req_ids:
      req_op_ids = self.IER.ReqId2OpReqId[req_id]
      cnt_op = 0
      for req_op_id in req_op_ids:
        if req_op_id in self.test_index_local:
          cnt_op += 1
        if req_op_id in self.test_index_global:
          cnt_op += 1
      if cnt_op > 0:
        self.filtered_test_req_ids.append(req_id)
    filter_req['train'] = self.filtered_train_req_ids
    filter_req['val'] = self.filtered_val_req_ids
    filter_req['test'] = self.filtered_test_req_ids
    with open(save_path, 'w') as f:
      json.dump(filter_req, f)

  def set_boxes(self):
    # each op_req_id has a box
    # assign ann id
    """
    ann: object or mask id
    idea: op_id -> pair_id -> candidate_mask_id
    ann_id: mask id
    anns: list of box(mask)
    ops: {op_id: {'candidate_anns': [], 'gt_anns': [], 'is_local': bool, 'mask_mode': inclusive/exclusive}}
    where 'gt_anns' are all inclusively recorded.
    """
    if os.path.exists(os.path.join(cur_dir, 'IER_ann_ops_sess_{}.json'.format(self.session))):
      ann_ops = json.load(open(os.path.join(cur_dir, 'IER_ann_ops_sess_{}.json'.format(self.session)), 'r'))
      self.anns = ann_ops['anns']
      self.ops = {int(k): v for (k, v) in ann_ops['ops'].items()}
      return

    ops = {}
    anns = []
    ann_id = 0
    # get the absolute value
    for op_id in self.IER.getOp:
      pair_id = self.IER.OpId2PairId[op_id]
      # load candidate mask
      masks, sizes, clss = self.IER.get_candidate_masks_with_clss(pair_id)
      candi_ann_ids = []
      for mask, size, cls in zip(masks, sizes, clss):
        candi_ann_ids.append(ann_id)
        ann_id += 1
        box_dict = {'box': self.get_box(mask, cls), 'op_id': op_id, 'size': size, 'class': int(cls)}
        anns.append(box_dict)

      ops[op_id] = {}
      ops[op_id]['candidate_anns'] = candi_ann_ids
      # load gt mask
      op_info = self.get_op_info(op_id)
      ops[op_id]['is_local'] = op_info['is_local']
      if op_info['is_local']:
        if op_info['mask_mode'] == 'inclusive':
          rel_mask_ids = op_info['mask_ids']
        else:
          rel_mask_ids = [i for i in range(len(candi_ann_ids)) if i not in op_info['mask_ids']]
        gt_mask_ids = [candi_ann_ids[i] for i in rel_mask_ids]
        ops[op_id]['gt_anns'] = gt_mask_ids
        ops[op_id]['mask_mode'] = op_info['mask_mode']

      print('op_id: {}/{}'.format(op_id, len(self.IER.getOp)))

    self.anns = anns
    self.ops = ops
    ann_ops = {'anns': anns, 'ops': ops}
    with open(os.path.join(cur_dir, 'IER_ann_ops_sess_{}.json'.format(self.session)), 'w') as f:
      json.dump(ann_ops, f)

  def filter_local(self):
    """filter out local operators"""
    local_op_req_ids = []
    for op_req_id in self.IER.OpReqId2ReqId.keys():
      op_id = self.IER.OpReqId2OpId[op_req_id]
      if self.ops[op_id]['is_local'] and len(self.ops[op_id]['gt_anns']) > 0:
        if (np.array(self.IER.getReqIdx[self.IER.OpReqId2ReqId[op_req_id]]) != 0).sum() == 0:
          continue
        local_op_req_ids.append(op_req_id)
    return local_op_req_ids

  def filter_global(self):
    """filter out global operator"""
    global_op_req_ids = []
    for op_req_id in self.IER.OpReqId2ReqId.keys():
      op_id = self.IER.OpReqId2OpId[op_req_id]
      if not self.ops[op_id]['is_local']:
        if (np.array(self.IER.getReqIdx[self.IER.OpReqId2ReqId[op_req_id]]) != 0).sum() == 0:
          continue
        global_op_req_ids.append(op_req_id)
    return global_op_req_ids

  # adapt for gan
  def __getitem__(self, item):
    max_length = self.max_length
    if self.phase == 'train':
      req_id = self.filtered_train_req_ids[item]
      req_op_ids = self.IER.ReqId2OpReqId[req_id]
      data = []
      cnt = 0
      for req_op_id in req_op_ids:
        if req_op_id in self.train_index_global:
          data += self.getTrainItem(req_op_id)
          cnt += 1
      for i in range(max_length - cnt):
        data += self.getBlankTrainItem(req_id)
      return tuple(data)
    elif self.phase == 'val' or self.phase == 'test':
      return self.getTestItem(item)
    else:
      raise Exception("Invalid Phase {}!".format(self.phase))

  def __len__(self):
    if self.phase == 'train':
      return len(self.filtered_train_req_ids)
    elif self.phase == 'val':
      return len(self.filtered_val_req_ids)
    elif self.phase == 'test':
      return len(self.filtered_test_req_ids)
    else:
      raise Exception("Invalid Phase {}!".format(self.phase))

  def load_train_img(self, img_name):
    img_path = os.path.join(self.img_file, img_name)
    img = cv2.imread(img_path)
    # resize to same
    img = cv2.resize(img, (128, 128))
    # BGR -> RGB
    img = img[:, :, ::-1].astype(np.float32)
    # normalize to (0, 1)
    img = img / 255
    # permutate
    img = img.transpose(2, 0, 1)
    return img

  def getTrainItem(self, item):
    """
    :param: item: op_req_id
    :return: Feats
             - pfeat (512,) panotpic feat
             - lfeat (5,)
             - dif_lfeats (25,)
             - ctx_pfeats (5, 512,)
             - ctx_lfeats (5, 5)
    :return: pos_req (max_len,)
    :return: pos_op int
    :return: neg_Feats (same as Feats)
    :return: neg_reqs (max_len,)
    """
    # once get
    # op_req_id = self.index[item]

    # adapt fot gan
    op_req_id = item

    req_id = self.IER.OpReqId2ReqId[op_req_id]
    op_id = self.IER.OpReqId2OpId[op_req_id]
    pair_id = self.IER.ReqId2PairId[req_id]
    input_img_name = self.IER.regularize_name(self.IER.op_data[pair_id]['input'])
    img = self.load_train_img(input_img_name)

    # positive sample
    pos_op = self.IER.OpId2OpIdx(op_id) if self.opt.use_op_prior else 0  # int
    pos_req = self.IER.getReqIdx[req_id]  # (max_len,)

    # adapt fot gan
    flag = 1
    return [img, pos_req, pos_op, req_id, flag]

  # adapt for gan
  def getBlankTrainItem(self, item):
    img = np.zeros((3, 128, 128)).astype('float32')
    pos_req = np.zeros(15).astype('int64')
    pos_op = 0
    req_id = item
    flag = 0
    return [img, pos_req, pos_op, req_id, flag]

  def getTestItem(self, item):
    """
    :param: item: op_req_id
    :return: Feats
             - pfeat (n, 512) panotpic feat
             - lfeat (n, 5)
             - dif_lfeats (n, 25)
             - ctx_pfeats (n, 5, 512)
             - ctx_lfeats (n, 5, 5)
    :return: req (max_len,)
    :return: op int

    :param item:
    :return:
    """
    # once get
    op_req_id = self.index[item]
    req_id = self.IER.OpReqId2ReqId[op_req_id]
    op_id = self.IER.OpReqId2OpId[op_req_id]
    pair_id = self.IER.OpId2PairId[op_id]

    # sample
    op = self.IER.OpId2OpIdx(op_id) if self.opt.use_op_prior else 0  # int
    req = self.IER.getReqIdx[req_id]  # (max_len,)
    gt_ann_ids = self.ops[op_id]['gt_anns']  # (m,)
    ann_ids = self.ops[op_id]['candidate_anns']  # (n,)
    pfeats = [self.fetch_feats(ann_id) for ann_id in ann_ids]  # (n, 512)
    lfeats = [self.compute_lfeats(ann_id) for ann_id in ann_ids]  # (5,)
    dif_lfeats = [self.compute_dif_lfeats(ann_id) for ann_id in ann_ids]  # (25,)
    cxt_pfeats, cxt_lfeats, cxt_ann_ids = [], [], []
    for ann_id in ann_ids:
      cxt_pfeat, cxt_lfeat, cxt_ann_id = self.fetch_cxt_feats(ann_id, self.opt)
      cxt_pfeats.append(cxt_pfeat)
      cxt_lfeats.append(cxt_lfeat)
      cxt_ann_ids.append(cxt_ann_id)
      # (n, 5, 512), (n, 5, 5), (n, 5)

    # return data
    data = {}
    data['pair_id'] = pair_id
    data['gt_ann_ids'] = gt_ann_ids
    data['cxt_ann_ids'] = cxt_ann_ids  # absolute
    data['req_id'] = req_id
    data['ann_ids'] = ann_ids
    data['Feats'] = {'pfeats': pfeats, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                     'cxt_pfeats': cxt_pfeats, 'cxt_lfeats': cxt_lfeats}
    data['req'] = req
    data['op'] = op

    return data


if __name__ == '__main__':
  opt = {'num_cxt': 5, 'sample_ratio': 0.3, 'session': 3, 'use_op_prior': 1}
  train_dataset = IERDataset(img_file, mask_file, showind2maskind_file, operator_file, feature_file, vocab_dir, 'train',
                             opt)
  # test_dataset = IERDataset(mask_file, showind2maskind_file, operator_file, feature_file, vocab_dir, 'test', opt)
  for i in range(len(train_dataset)):
    train_dataset[i]
    print('iter {}'.format(i))

  # dataloader = DataLoader(train_dataset, batch_size=4,
  #                         shuffle=True, num_workers=4)
  # for i, data in enumerate(dataloader):
  #     print('iter {}'.format(i))
