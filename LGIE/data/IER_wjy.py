import os
import glob
import re
import json
import pdb
import base64
import h5py
import string
from functools import reduce

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.mask import decode as mask_decode

'''
merget request, operator, mask label together
'''

# requires two files: request data, operator json
mask_file_temp = '../datasets/data/IER2/masks'
showind2maskind_file_temp = '../datasets/data/IER2/show2mask'
feature_file_temp = '../datasets/data/IER2/features'
operator_file = '../datasets/data/IER2/IER2.json'
vocab_dir = '../datasets/data/language'

"""
example of each cell in operator json
[{'input': input, 'output': output, 'segment': segment, 'palette': palette, 'request', request, 'detailed_request': detailed_request, 'dataset': dataset, 'operator': {'op1': {'mask_mode': 'inclusive', 'local': bool, 'ids': [ids]},}, 'expert_summary': ['sentence1', 'sentence2'], 'amateur_summary': ['sentence1', 'sentence2', 'sentence3']},]
"""

"""
The top operators
[brightness, contrast, saturation, hue, inpaint_obj, tint, sharpness, color_bg]
"""


def parse_sent(desc):
  table = str.maketrans('', '', string.punctuation)
  # tokenize
  desc = desc.split()
  # convert to lower case
  desc = [word.lower() for word in desc]
  # remove punctuation from each token
  desc = [w.translate(table) for w in desc]
  # remove hanging 's' and 'a'
  desc = [word for word in desc if len(word) > 1]
  # remove tokens with numbers in them
  tokens = [word for word in desc if word.isalpha()]
  return tokens


class IER(object):
  """
  op_req_id: operator distinguished by request
  op_id: operator distinguished by image pair
  The following API functions are defined:
  IER          - IER api class
  getImgId     - get image id based on image name
  getReq       - get reqest based on request id
  getOp        - get operator based on op_id
  getOpReq     - get operator name based on op_req_id
  ImgId2PairId - get image pair id based on either input image id or output image id
  ReqId2PairId - get image id based on ReqId
  OpId2PairId  - get op id based on pair id
  OpReqId2ReqId- get ReqId based on OpReqId
  OpReqID2OpId - get OpId based on OpReqId

  getMask      - get mask based on pair id and operator: provide each mask index from ids, and also the total mask it will use.

  getAnnIds  - get ann ids that satisfy given filter conditions.
  getImgIds  - get image ids that satisfy given filter conditions.
  getCatIds  - get category ids that satisfy given filter conditions.
  loadRefs   - load refs with the specified ref ids.
  loadAnns   - load anns with the specified ann ids.
  loadImgs   - load images with the specified image ids.
  loadCats   - load category names with the specified category ids.
  getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
  showRef    - show image, segmentation or box of the referred object with the ref
  getMask    - get mask and area of the referred object given ref
  showMask   - show mask of the referred object given ref
  """

  def __init__(self, mask_file, show2mask_file, operator_file, feature_file, vocab_dir, session, opt):
    self.opt = opt
    self.op_max_len = 10
    self.req_max_len = 15
    self.session = session
    self.mask_file = mask_file
    self.show2mask_file = show2mask_file
    self.feature_file = feature_file
    self.op_data = self.load_ops(operator_file)
    self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = self.load_vocab(vocab_dir)
    self.create_index(self.op_data)
    self.define_split(len(self.op_data))

  def load_ops(self, operator_file):
    with open(operator_file) as f:
      op_data = json.load(f)

    temp = []
    # filter data without any request
    for data in op_data:
      if data['expert_summary'] + data['amateur_summary']:
        temp.append(data)
    op_data = temp
    return op_data

  def req2idx(self, sent):
    """change sentence to index"""

    def token2idx(token):
      idx = self.vocab2id[token] if token in self.vocab2id else 3
      return idx

    tokens = parse_sent(sent)
    valid_sent_idx = np.array([token2idx(token) for token in tokens])
    valid_len = len(valid_sent_idx)
    sent_idx = np.zeros(self.req_max_len, dtype=int)
    sent_idx[:min(valid_len, self.req_max_len)] = valid_sent_idx[:self.req_max_len]
    return sent_idx

  def regularize_name(self, name):
    """ make the '5nxqz/35nxqz.jpg' to '5nxqz_35nxqz.jpg'
    """
    return name.replace('/', '_')

  def define_split(self, id_len):
    pair_ids = np.arange(id_len)
    np.random.seed(0)  # very important
    np.random.shuffle(pair_ids)
    self.train_pair_ids = pair_ids[:int(id_len * 0.8)]
    self.val_pair_ids = pair_ids[int(id_len * 0.8):int(id_len * 0.9)]
    self.test_pair_ids = pair_ids[int(id_len * 0.9):]

    self.train_req_ids = reduce(lambda x, y: x + y, [self.PairId2ReqId[i] for i in self.train_pair_ids])
    self.val_req_ids = reduce(lambda x, y: x + y, [self.PairId2ReqId[i] for i in self.val_pair_ids])
    self.test_req_ids = reduce(lambda x, y: x + y, [self.PairId2ReqId[i] for i in self.test_pair_ids])

  def filter_operator(self, op_list):
    filtered_op_list = list(filter(lambda x: x in self.op_vocab2id, list(op_list.keys())))
    return filtered_op_list

  def create_index(self, op_data):
    pair_ids = [i for i in range(len(op_data))]
    # go through all images
    imgs = []
    for data in op_data:
      imgs.append(self.regularize_name(data['input']))
      imgs.append(self.regularize_name(data['output']))
    imgs = np.unique(imgs)
    img_ids = [i for i in range(len(imgs))]
    getImgId = {name: id for name, id in zip(imgs, img_ids)}

    # go through all requests
    ReqId2PairId = {}
    ImgId2PairId = {}
    OpReqId2ReqId = {}
    OpReqId2OpId = {}
    OpId2PairId = {}
    getOpReq = {}
    getOp = {}
    getReq = {}
    getReqIdx = {}  # ReqIdx: the idx in vocabulary

    # adapt for gan
    ReqId2OpReqId = {}

    req_id = 0
    op_req_id = 0
    op_id = 0
    for pair_i, data in enumerate(op_data):
      op_id_start = op_id
      # get data
      for op in self.filter_operator(data['operator']):
        OpId2PairId[op_id] = pair_i
        getOp[op_id] = op
        op_id += 1
      if data['expert_summary'] == [] and data['amateur_summary'] == []:
        pdb.set_trace()
      if self.opt.use_expert:
        if len(data['expert_summary']) > 0:
          summary_list = data['expert_summary']
        else:
          summary_list = data['amateur_summary'][0]
      else:
        summary_list = data['expert_summary'] + data['amateur_summary']
      for req in summary_list:
        getReq[req_id] = req
        getReqIdx[req_id] = self.req2idx(req)
        ReqId2PairId[req_id] = pair_i
        ImgId2PairId[getImgId[self.regularize_name(data['input'])]] = pair_i
        ImgId2PairId[getImgId[self.regularize_name(data['output'])]] = pair_i
        ReqId2OpReqId[req_id] = []
        for op_i, op in enumerate(self.filter_operator(data['operator'])):
          OpReqId2ReqId[op_req_id] = req_id
          OpReqId2OpId[op_req_id] = op_id_start + op_i
          getOpReq[op_req_id] = op

          # adapt for gan
          ReqId2OpReqId[req_id].append(op_req_id)

          op_req_id += 1
        req_id += 1

    PairId2ReqId = {}
    for req_id in ReqId2PairId:
      pair_id = ReqId2PairId[req_id]
      if pair_id in PairId2ReqId:
        PairId2ReqId[pair_id].append(req_id)
      else:
        PairId2ReqId[pair_id] = [req_id]

    self.getImgId = getImgId
    self.getReq = getReq
    self.getReqIdx = getReqIdx
    self.getOpReq = getOpReq
    self.getOp = getOp
    self.ImgId2PairId = ImgId2PairId
    self.ReqId2PairId = ReqId2PairId
    self.PairId2ReqId = PairId2ReqId
    self.OpReqId2ReqId = OpReqId2ReqId
    self.OpReqId2OpId = OpReqId2OpId
    self.OpId2PairId = OpId2PairId

    # add by wjy
    self.ReqId2OpReqId = ReqId2OpReqId

  def OpId2OpIdx(self, op_id):
    return self.op_vocab2id[self.getOp[op_id]]

  def get_mask(self, pair_id, operator):
    mask_dict = self.op_data[pair_id]['operator'][operator]
    mask_id = mask_dict['ids']
    mask_mode = mask_dict['mask_mode']
    is_local = mask_dict['local']
    if np.isnan(is_local):
      pdb.set_trace()
    return is_local, mask_mode, mask_id

  def show_req_mask(self, img_name, out_name, req, masks, mask_mode):
    img = cv2.imread(os.path.join('images', img_name))[:, :, ::-1]
    out = cv2.imread(os.path.join('images', out_name))[:, :, ::-1]
    mask = np.zeros_like(img[:, :, 0])
    for mask_ in masks:
      h_mask, w_mask = mask_.shape
      h_img, w_img = mask.shape
      if h_mask != h_img or w_mask != w_img:
        print('unequal mask shape {} and image shape {}'.format((h_mask, w_mask), (h_img, w_img)))
      h = min(h_mask, h_img)
      w = min(w_mask, w_img)
      mask[:h, :w] += mask_[:h, :w]
    mask = np.clip(mask, 0, 1).astype(np.uint8)
    mask = 1 - mask if mask_mode == 'exclusive' else mask
    plt.axis("tight")
    fig = plt.figure(figsize=(25, 8), dpi=80)
    fig.suptitle(req, fontsize=12)
    ax = fig.add_subplot(131)
    ax.set_title('input image')
    ax.imshow(img)
    ax.axis("off")

    img[:, :, 0] = img[:, :, 0] * (1 - mask) + 255 * mask
    ax = fig.add_subplot(132)
    ax.set_title('masked part')
    ax.imshow(img)
    ax.axis("off")

    ax = fig.add_subplot(133)
    ax.set_title('edited image')
    ax.imshow(out)
    ax.axis("off")

    os.makedirs('vis', exist_ok=True)
    img_id, img_ext = img_name.split('.')
    img_suffix = base64.b64encode('{}'.format(req).encode()).decode()[:4]
    plt.savefig('./vis/{}'.format(img_id + img_suffix + '.' + img_ext))

  def load_mask_feature(self, pair_id):
    img_name = self.regularize_name(self.op_data[pair_id]['input'])
    feature_name = img_name.split('.')[0] + '.h5'
    dataset = self.op_data[pair_id]['dataset']
    feature_file = os.path.join(self.feature_file.format(dataset), feature_name)
    try:
      f = h5py.File(feature_file, 'r')
    except:
      pdb.set_trace()
    pan_feats = f['pan_feat'][:]
    rcnn_feats = f['rcnn_feat'][:]
    pan_clss = f['cls_inds'][:]
    inst_inds = f['inst_inds'][:]
    inst_ids = f['inst_ids'][:]

    return pan_feats, rcnn_feats, pan_clss, inst_inds, inst_ids

  def load_mask(self, pair_id):
    """
    load all the candidate masks
    :param pair_id:
    :return:
    """
    img_name = self.regularize_name(self.op_data[pair_id]['input'])
    dataset = self.op_data[pair_id]['dataset']
    # get the mask from the mask file
    mask_name = img_name.split('.')[0] + '_mask.json'
    mask_file = os.path.join(self.mask_file.format(dataset), mask_name)
    with open(mask_file) as f:
      mask_data = json.load(f)
    masks = [mask_decode(mask_rle) for mask_rle in mask_data]
    return masks

  def load_vocab(self, vocab_dir):
    """load vocabulary from files under vocab_dir"""
    with open(os.path.join(vocab_dir, 'IER_vocabs_sess_{}.json'.format(self.session))) as f:
      vocab = json.load(f)
    with open(os.path.join(vocab_dir, 'IER_operator_vocabs_sess_{}.json'.format(self.session))) as f:
      op_vocab = json.load(f)
    vocab2id = {token: i for i, token in enumerate(vocab)}
    id2vocab = {i: token for i, token in enumerate(vocab)}
    op_vocab2id = {token: i for i, token in enumerate(op_vocab)}
    id2op_vocab = {i: token for i, token in enumerate(op_vocab)}
    return vocab2id, id2vocab, op_vocab2id, id2op_vocab

  def get_single_mask(self, pair_id, op):
    input = self.regularize_name(self.op_data[pair_id]['input'])
    dataset = self.op_data[pair_id]['dataset']
    is_local, mask_mode, mask_ids, = self.get_mask(pair_id, op)
    if is_local:
      # get the mask from the mask file
      mask_name = input.split('.')[0] + '_mask.json'
      mask_file = os.path.join(self.mask_file.format(dataset), mask_name)
      with open(mask_file) as f:
        mask_data = json.load(f)
      show2mask_name = input.split('.')[0] + '_show2mask.json'
      show2mask_file = os.path.join(self.show2mask_file.format(dataset), show2mask_name)
      with open(show2mask_file) as f:
        showind2maskind = json.load(f)
      masks = []
      for mask_id in mask_ids:
        mask_rle = mask_data[showind2maskind[mask_id]]
        mask = mask_decode(mask_rle)
        masks.append(mask)
      return masks
    else:
      return None

  def get_candidate_masks_with_clss(self, pair_id):
    """
    :return: masks: list of (h, w)
    :return: pan_clss: (n,)
    """
    input = self.regularize_name(self.op_data[pair_id]['input'])
    dataset = self.op_data[pair_id]['dataset']
    # get the mask from the mask file
    mask_name = input.split('.')[0] + '_mask.json'
    mask_file = os.path.join(self.mask_file.format(dataset), mask_name)
    with open(mask_file) as f:
      mask_data = json.load(f)
    masks = []
    sizes = []
    for mask_rle in mask_data:
      sizes.append(mask_rle['size'])
      mask = mask_decode(mask_rle)
      masks.append(mask)

    _, _, pan_clss, _, _ = self.load_mask_feature(pair_id)
    return masks, sizes, pan_clss

  def test(self):
    for req_id in range(len(self)):
      pair_id = self.ReqId2PairId[req_id]
      input = self.regularize_name(self.op_data[pair_id]['input'])
      output = self.regularize_name(self.op_data[pair_id]['output'])
      ops = self.op_data[pair_id]['operator']
      req = self.getReq[req_id]
      dataset = self.op_data[pair_id]['dataset']
      pan_feats, rcnn_feats, pan_clss, inst_inds, inst_ids = self.load_mask_feature(pair_id)
      print('req_id', req_id)
      for op in ops:
        is_local, mask_mode, mask_ids, = self.get_mask(pair_id, op)
        if is_local:
          # get the mask from the mask file
          mask_name = input.split('.')[0] + '_mask.json'
          mask_file = os.path.join(self.mask_file.format(dataset), mask_name)
          with open(mask_file) as f:
            mask_data = json.load(f)
          show2mask_name = input.split('.')[0] + '_show2mask.json'
          show2mask_file = os.path.join(self.show2mask_file.format(dataset), show2mask_name)
          with open(show2mask_file) as f:
            showind2maskind = json.load(f)
          masks = []
          for mask_id in mask_ids:
            mask_rle = mask_data[showind2maskind[mask_id]]
            mask = mask_decode(mask_rle)
            masks.append(mask)
            # pan_feat = pan_feats[showind2maskind[mask_id]]
            # pan_cls = pan_clss[showind2maskind[mask_id]]
          # self.show_req_mask(input, output, req, masks, mask_mode)

  def get_op_info(self, pair_id):
    """
    operator_idx: (max_op_len)
    is_local: (bs, masx_op_len) (1 or 0)
    mask_id: {'operator_id': [1, 2, 3]} (the direct id that can index the feature and mask)
    :param pair_id:
    :return:
    """
    img_name = self.regularize_name(self.op_data[pair_id]['input'])
    op_dict = self.op_data[pair_id]['operator']
    is_local_list = []
    operator_idx = []
    mask_dict = {}
    masks, sizes, clss = self.get_candidate_masks_with_clss(pair_id)
    for op in op_dict:
      if op in self.op_vocab2id:
        operator_idx.append(self.op_vocab2id[op])
        is_local, mask_mode, mask_ids = self.get_mask(pair_id, op)
        is_local_list.append(int(is_local))
        if is_local:
          show2mask_name = img_name.split('.')[0] + '_show2mask.json'
          show2mask_file = os.path.join(self.show2mask_file, show2mask_name)
          with open(show2mask_file) as f:
            showind2maskind = json.load(f)
          mask_ids = [showind2maskind[mask_id] for mask_id in mask_ids]
          if mask_mode == 'exclusive':
            mask_ids = [i for i in range(len(masks)) if i not in mask_ids]
          mask_dict[int(self.op_vocab2id[op])] = mask_ids
    operator_idx += [0] * (self.op_max_len - len(operator_idx))
    is_local_list += [0] * (self.op_max_len - len(is_local_list))
    return operator_idx, is_local_list, mask_dict

  def get_item_(self, req_id):
    """ Get item for saving annotation
    :param req_id:
    :return: dict
    - request_idx: (max_req,)
    - operator_idx: (max_op,)
    - is_local: (max_op,) (1 or 0)
    - input: str
    - output: str
    - mask_id: {'operator_id': [1, 2, 3]} (the direct id that can index the feature and mask)
    """
    req_idx = self.getReqIdx[req_id].tolist()  # (max_len,)
    if sum(req_idx) == 0:
      pdb.set_trace()
    req = self.getReq[req_id]
    pair_id = self.ReqId2PairId[req_id]
    input = self.regularize_name(self.op_data[pair_id]['input'])
    output = self.regularize_name(self.op_data[pair_id]['output'])
    op_idx, is_local, mask_id = self.get_op_info(pair_id)
    return {'input': input, 'output': output, 'is_local': is_local, 'request_idx': req_idx, 'request': req,
            'operator_idx': op_idx, 'mask_id': mask_id}

  def save_annos(self):
    anno_dir = 'annotations'
    os.makedirs(anno_dir, exist_ok=True)
    annos = []
    for i in range(len(self)):
      anno = self.get_item_(i)
      annos.append(anno)
      print('parse index {}/{}'.format(i + 1, len(self)))
    anno_save_path = os.path.join(anno_dir, 'IER_sess_{}.json'.format(self.session))
    with open(anno_save_path, 'w') as f:
      json.dump(annos, f)

  def get_item(self, req_id):
    """ get item from saved annotation
    :param req_id:
    :return:
    - request_idx: (max,)
    - operator_idx: (n_op,)
    - is_local: (n_op,) (1 or 0)
    - input: str
    - output: str
    - mask_id: {'operator_id': [1, 2, 3]} (the direct id that can index the feature and mask)
    """
    anno_path = 'annotations/IER_rl_{}.json'.format(self.session)
    with open(anno_path, 'r') as f:
      annos = json.load(f)

  def mode_type_stat(self):
    op_dict = {}

    for op in self.op_data:
      op_names = op['operator'].keys()
      for op_name in op_names:
        if op['operator'][op_name]['local']:
          mask_mode = op['operator'][op_name]['mask_mode']
          if op_name not in op_dict.keys():
            op_dict[op_name] = {'inclusive': 0, 'exclusive': 0}
          else:
            if mask_mode == 'inclusive':
              op_dict[op_name]['inclusive'] += 1
            else:
              op_dict[op_name]['exclusive'] += 1

    for op in op_dict:
      inclusive_cnt = op_dict[op]['inclusive']
      exclusive_cnt = op_dict[op]['exclusive']
      cnt = inclusive_cnt + exclusive_cnt
      print('{}: inclusive: {} exclusive: {} in_ratio: {:.2f}%'.format(op, inclusive_cnt, exclusive_cnt,
                                                                       inclusive_cnt / cnt * 100))

  def __len__(self):
    return len(self.getReq)


if __name__ == '__main__':
  session = 3
  ier = IER(mask_file_temp, showind2maskind_file_temp, operator_file, feature_file_temp, vocab_dir, session)
  # ier.test()
  ier.save_annos()
