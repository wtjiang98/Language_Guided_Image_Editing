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
from util.utils import CATEGORIES

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
    self.local_op_req_ids = self.filter_local()
    self.global_op_req_ids = self.filter_global()
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

  def get_op_info(self, op_id):
    """get operation info dict"""
    pair_id = self.IER.OpId2PairId[op_id]
    op_name = self.IER.getOp[op_id]
    img_name = self.IER.regularize_name(self.IER.op_data[pair_id]['input'])
    dataset = self.IER.op_data[pair_id]['dataset']
    is_local, mask_mode, mask_ids, = self.IER.get_mask(pair_id, op_name)
    op_info = {'mask_ids': [], 'mask_mode': mask_mode, 'is_local': is_local}
    if is_local:
      show2mask_name = img_name.split('.')[0] + '_show2mask.json'
      show2mask_file = os.path.join(self.IER.show2mask_file.format(dataset), show2mask_name)
      with open(show2mask_file) as f:
        showind2maskind = json.load(f)
      for mask_id in mask_ids:
        op_info['mask_ids'].append(showind2maskind[mask_id])
    return op_info

  def get_box(self, mask, cls):
    """ Due to the bad mask from UPSNet, need to manually filter out the bad masks
    filter criterion: area < perimeter; isthing; less then 1/10 of the largest area
    :param mask: (h, w) \in {0, 1}
    :param cls: int
    :return: box (x1, y1, x2, y2)
    """
    mask = (mask * 255).astype(np.uint8)
    contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    c_areas = [cv2.contourArea(c) for c in contour]
    x1s, y1s, x2s, y2s = [], [], [], []
    for i, c in enumerate(contour):
      # eliminate the wrong masks patches
      if c_areas[i] < len(c) and c_areas[i] < max(c_areas) * 0.1 and CATEGORIES[cls]['isthing']:
        # cv2.drawContours(mask, [c], -1, 0, -1)
        continue
      x1s.append(int(c[:, 0, 0].min()))
      y1s.append(int(c[:, 0, 1].min()))
      x2s.append(int(c[:, 0, 0].max()))
      y2s.append(int(c[:, 0, 1].max()))

    box = [min(x1s), min(y1s), max(x2s), max(y2s)]
    return box

  def fetch_neighbour_ids(self, ref_ann_id):
    """
    For a given ref_ann_id, we return
    - st_ann_ids: same-type neighbouring ann_ids (not including itself)
    - dt_ann_ids: different-type neighbouring ann_ids
    Ordered by distance to the input ann_id
    """
    ref_ann = self.anns[ref_ann_id]
    x1, y1, x2, y2 = ref_ann['box']
    rx, ry = (x1 + x2) / 2, (y1 + y2) / 2

    def key(ann_id):
      x1, y1, x2, y2 = self.anns[ann_id]['box']
      ax0, ay0 = (x1 + x2) / 2, (y1 + y2) / 2
      r = (rx - ax0) ** 2 + (ry - ay0) ** 2
      return r

    op = self.ops[ref_ann['op_id']]

    ann_ids = list(op['candidate_anns'])  # copy in case the raw list is changed
    ann_ids = sorted(ann_ids, key=key)

    st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = [], [], [], []
    for ann_id in ann_ids:
      if ann_id != ref_ann_id:
        if self.anns[ann_id]['class'] == ref_ann['class']:
          st_ann_ids += [ann_id]
        else:
          dt_ann_ids += [ann_id]

    return st_ann_ids, dt_ann_ids

  def compute_lfeats(self, ann_id):
    """
    :param ann_ids: int
    :return lfeats (5,)
    """
    ann = self.anns[ann_id]
    x1, y1, x2, y2 = ann['box']
    ih, iw = ann['size']
    lfeat = np.array([x1 / iw, y1 / ih, x2 / iw, y2 / ih, (x2 - x1 + 1) * (y2 - y1 + 1) / (iw * ih)], np.float32)
    return lfeat

  def compute_dif_lfeats(self, ref_ann_id, topK=5):
    """
    :param ref_ann_id: ind
    :param topK:
    :return dif_lfeats: (5*topK)
    """
    dif_lfeats = np.zeros(5 * topK, dtype=np.float32)
    # reference box
    rbox = self.anns[ref_ann_id]['box']
    rx1, ry1, rx2, ry2 = rbox
    rcx, rcy, rw, rh = (rx1 + rx2) / 2, (ry1 + ry2) / 2, rx2 - rx1 + 1, ry2 - ry1 + 1
    # candidate boxes
    st_ann_ids, _ = self.fetch_neighbour_ids(ref_ann_id)
    for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
      cbox = self.anns[cand_ann_id]['box']
      cx1, cy1, cx2, cy2 = cbox[0], cbox[1], cbox[2], cbox[3]
      cw, ch = cx2 - cx1, cy2 - cy1
      dif_lfeats[j * 5:(j + 1) * 5] = \
        np.array([(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx2 - rcx) / rw, (cy2 - rcy) / rh, cw * ch / (rw * rh)])
    return dif_lfeats

  def fetch_cxt_feats(self, ref_ann_id, opt):
    """
    Return
    - cxt_feats  : ndarray (topK, fc7_dim)
    - cxt_lfeats : ndarray (topK, 5)
    - cxt_ann_ids: [ann_id] of size (topK,), padded with -1
    Note we only use neighbouring "different"(+"same") objects for computing context objects, zeros padded.
    """
    topK = opt.num_cxt
    cxt_feats = np.zeros((topK, self.pan_dim), dtype=np.float32)
    cxt_lfeats = np.zeros((topK, 5), dtype=np.float32)
    cxt_ann_ids = [-1 for _ in range(topK)]  # (topK,)
    # reference box
    rbox = self.anns[ref_ann_id]['box']
    rcx, rcy, rw, rh = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2, rbox[2] - rbox[0] + 1, rbox[3] - rbox[1] + 1
    # candidate boxes
    st_ann_ids, dt_ann_ids = self.fetch_neighbour_ids(ref_ann_id)
    cand_ann_ids = dt_ann_ids + st_ann_ids
    # if opt['with_st'] > 0:
    #     cand_ann_ids = dt_ann_ids + st_ann_ids
    # else:
    #     cand_ann_ids = dt_ann_ids
    cand_ann_ids = cand_ann_ids[:topK]
    for j, cand_ann_id in enumerate(cand_ann_ids):
      cand_ann = self.anns[cand_ann_id]
      cbox = cand_ann['box']
      cx1, cy1, cx2, cy2 = cbox[0], cbox[1], cbox[2], cbox[3]
      cw, ch = cx2 - cx1 + 1, cy2 - cy1 + 1
      cxt_lfeats[j, :] = np.array(
        [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx2 - rcx) / rw, (cy2 - rcy) / rh, cw * ch / (rw * rh)])
      cxt_feats[j, :] = self.fetch_feats(cand_ann_id)
      cxt_ann_ids[j] = cand_ann_id

    return cxt_feats, cxt_lfeats, cxt_ann_ids

  def fetch_feats(self, ann_id):
    """
    :param ann_ids: int
    :return: pan_feat (512,)
    """
    op_id = self.anns[ann_id]['op_id']
    pair_id = self.IER.OpId2PairId[op_id]
    pan_feats, _, _, _, _ = self.IER.load_mask_feature(pair_id)
    anns = self.ops[op_id]['candidate_anns']
    rel_ann_id = anns.index(ann_id)
    try:
      pan_feat = pan_feats[rel_ann_id]
    except:
      pdb.set_trace()
    return pan_feat

  def sample_pos_ann(self, op_id):
    """random choose one positive ann_id from multiple pos ann_ids belonging to a op_id"""
    ann_ids = self.ops[op_id]['gt_anns']
    ann_id = np.random.choice(ann_ids)
    return ann_id

  def sample_neg_id(self, ann_id, sample_ratio):
    """Return
    - neg_ann_id : ann_id that are negative to target ann_id
    - neg_op_req_id: op_req_id that are negative to target ann_id
    """
    st_ann_ids, dt_ann_ids = self.fetch_neighbour_ids(ann_id)

    # neg ann
    # neg_ann_id for negative visual representation: mainly from same-type objects
    if len(st_ann_ids) > 0 and np.random.uniform(0, 1, 1) < sample_ratio:
      neg_ann_id = np.random.choice(st_ann_ids)
    elif len(dt_ann_ids) > 0:
      neg_ann_id = np.random.choice(dt_ann_ids)
    else:
      neg_ann_id = np.random.choice(range(len(self.anns)))
    # neg_ref_id for negative language representations: mainly from same-type "referred" objects
    neg_op_req_id = np.random.choice(list(self.IER.OpReqId2ReqId.keys()))

    return neg_ann_id, neg_op_req_id

  # adapt for gan
  def __getitem__(self, item):
    max_length = self.max_length
    if self.phase == 'train':
      req_id = self.filtered_train_req_ids[item]
      req_op_ids = self.IER.ReqId2OpReqId[req_id]
      data = []
      cnt = 0
      for req_op_id in req_op_ids:
        if req_op_id in self.train_index_local:
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

    seg_per_ref = 1  # number of sentence per req_operator
    sample_ratio = self.opt.sample_ratio

    # positive sample
    pos_op = self.IER.OpId2OpIdx(op_id) if self.opt.use_op_prior else 0  # int
    pos_req = self.IER.getReqIdx[req_id]  # (max_len,)

    pos_idx = np.where(pos_req > 0)[0]
    if len(pos_idx) == 0:
      pdb.set_trace()
      pos_req = self.IER.getReqIdx[0]

    pos_ann_id = self.sample_pos_ann(op_id)
    pos_pfeat = self.fetch_feats(pos_ann_id)  # (512,)
    pos_lfeat = self.compute_lfeats(pos_ann_id)  # (5,)
    pos_dif_lfeat = self.compute_dif_lfeats(pos_ann_id)  # (25,)
    pos_cxt_pfeat, pos_cxt_lfeats, pos_cxt_ann_ids = self.fetch_cxt_feats(pos_ann_id,
                                                                          self.opt)  # (5, 512), (5, 5), (5,)

    # negative sample
    neg_ann_id, neg_op_req_id = self.sample_neg_id(pos_ann_id, sample_ratio)
    neg_req = self.IER.getReqIdx[self.IER.OpReqId2ReqId[neg_op_req_id]]

    neg_idx = np.where(neg_req > 0)[0]
    if len(neg_idx) == 0:
      neg_req = self.IER.getReqIdx[0]

    neg_pfeat = self.fetch_feats(neg_ann_id)
    neg_lfeat = self.compute_lfeats(neg_ann_id)
    neg_dif_lfeat = self.compute_dif_lfeats(neg_ann_id)
    neg_cxt_pfeat, neg_cxt_lfeats, neg_cxt_ann_ids = self.fetch_cxt_feats(neg_ann_id, self.opt)

    # return img, pos_pfeat, pos_lfeat, pos_dif_lfeat, pos_cxt_pfeat, pos_cxt_lfeats, pos_req, pos_op, \
    #        neg_pfeat, neg_lfeat, neg_dif_lfeat, neg_cxt_pfeat, neg_cxt_lfeats, neg_req

    # adapt for gan
    flag = 1
    return [img, pos_pfeat, pos_lfeat, pos_dif_lfeat, pos_cxt_pfeat, pos_cxt_lfeats, pos_req, pos_op, \
            neg_pfeat, neg_lfeat, neg_dif_lfeat, neg_cxt_pfeat, neg_cxt_lfeats, neg_req, req_id, flag]

  # adapt for gan
  def getBlankTrainItem(self, item):

    img = np.zeros((3, 128, 128)).astype('float32')
    pos_pfeat = np.zeros(512).astype('float32')
    pos_lfeat = np.zeros(5).astype('float32')
    pos_dif_lfeat = np.zeros(25).astype('float32')
    pos_cxt_pfeat = np.zeros((5, 512)).astype('float32')
    pos_cxt_lfeats = np.zeros((5, 5)).astype('float32')
    pos_req = np.zeros(15).astype('int64')
    pos_op = 0
    neg_pfeat = np.zeros(512).astype('float32')
    neg_lfeat = np.zeros(5).astype('float32')
    neg_dif_lfeat = np.zeros(25).astype('float32')
    neg_cxt_pfeat = np.zeros((5, 512)).astype('float32')
    neg_cxt_lfeats = np.zeros((5, 5)).astype('float32')
    neg_req = np.zeros(15).astype('int64')
    req_id = item
    flag = 0

    return [img, pos_pfeat, pos_lfeat, pos_dif_lfeat, pos_cxt_pfeat, pos_cxt_lfeats, pos_req, pos_op,
            neg_pfeat, neg_lfeat, neg_dif_lfeat, neg_cxt_pfeat, neg_cxt_lfeats, neg_req, req_id, flag]

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

    # adapt fot gan
    op_req_id = item

    req_id = self.IER.OpReqId2ReqId[op_req_id]
    op_id = self.IER.OpReqId2OpId[op_req_id]
    pair_id = self.IER.OpId2PairId[op_id]
    input_img_name = self.IER.regularize_name(self.IER.op_data[pair_id]['input'])
    img = self.load_train_img(input_img_name)

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
    data['img'] = [img]  # (1, 3, h, w)
    data['Feats'] = {'pfeats': pfeats, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                     'cxt_pfeats': cxt_pfeats, 'cxt_lfeats': cxt_lfeats}
    data['req'] = req
    data['op'] = op

    return data


def get_single_data_from_batch(data, idx):
  result = []
  for item in data:
    result.append(item[idx])
  return result


if __name__ == '__main__':
  torch.multiprocessing.set_sharing_strategy('file_system')
  opt = {'num_cxt': 5, 'sample_ratio': 0.3, 'session': 3, 'use_op_prior': 1, 'filter_req': 'v0'}
  train_dataset = IERDataset(img_file, mask_file, showind2maskind_file, operator_file, feature_file, vocab_dir, 'train',
                             opt)
  cnt_l_max = 0
  cnt_g_max = 0
  cnt_all_max = 0
  for i, req_id in enumerate(train_dataset.filtered_train_req_ids):
    cnt_l = 0
    cnt_g = 0
    cnt_all = 0
    req_op_ids = train_dataset.IER.ReqId2OpReqId[req_id]
    for req_op_id in req_op_ids:
      if req_op_id in train_dataset.train_index_local:
        cnt_l += 1
        cnt_all += 1
      if req_op_id in train_dataset.train_index_global:
        cnt_g += 1
        cnt_all += 1
    cnt_l_max = max(cnt_l, cnt_l_max)
    cnt_g_max = max(cnt_g, cnt_g_max)
    cnt_all_max = max(cnt_all, cnt_all_max)
    # if cnt_all_max == 7:
    #     pdb.set_trace()

  print(cnt_l_max, cnt_g_max, cnt_all_max)
  # print(cnt)

  pdb.set_trace()
  # for i in range(len(train_dataset)):
  #     data = train_dataset[i]
  #     print('iter {}'.format(i))
  #     req, op, neg_req = data[6], data[7], data[-1]
  #     pos_idx = np.where(req > 0)[0]
  #     neg_idx = np.where(neg_req > 0)[0]
  #     if len(pos_idx) == 0:
  #         pdb.set_trace()
  #     if len(neg_idx) == 0:
  #         pdb.set_trace()

  # test_dataset = IERDataset(img_file, mask_file, showind2maskind_file, operator_file, feature_file, vocab_dir, 'test', opt)
  # for i in range(len(test_dataset)):
  #     data = test_dataset[i]
  #     print('iter {}'.format(i))
  #     req = data['req']
  #     pos_idx = np.where(req > 0)[0]
  #     if len(pos_idx) == 0:
  #         pdb.set_trace()

  # dataloader = DataLoader(train_dataset, batch_size=4,
  #                         shuffle=True, num_workers=4)
  # for i, data in enumerate(dataloader):
  #     # pdb.set_trace()
  #     print('iter {}'.format(i))

  # aa = train_dataset.getTrainItem(0)
  # bb = train_dataset.getBlankTrainItem()

  # dataloader = DataLoader(train_dataset, batch_size=4,
  #                         shuffle=True, num_workers=4)
  # for data in dataloader:
  #     max_length = len(train_dataset.IER.op_vocab2id)
  #     datas = [data[i * int(len(data) / max_length) : (i + 1) * int(len(data) / max_length)] for i in range(max_length)]
  # batch_size = dataloader.batch_size
  # order_data = []
  # for m in range(max_length):
  #     for b in range(batch_size):
  #         cur_data = get_single_data_from_batch(datas[m], b)
  #         if cur_data[-1] == 1:
  #             pass
  pdb.set_trace()
  pass
