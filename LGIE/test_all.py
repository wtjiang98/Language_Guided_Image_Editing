# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_all_options import TestALLOptions
from models.all_model import ALLModel
from data.IERdataset_inference_wjy import IERDataset as IERDataset_inference
from trainers.all_trainer import AllTrainer
from torch.utils.data import DataLoader
from util.utils import set_seed, concat_single_data_to_batch, to_tensor, collate_fn_inference
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm
# from . import util
import numpy as np
import cv2
from util.util_jwt import use_op
import torch.nn.functional as F
from util.vis_atten.vis_simple_region_nonlocal import vis_map
import torch
import json
import pdb


def tensor_to_numpy(input_img):
  # [-1, 1] --> [0, 1]
  input_img.add_(1).div_(2).mul_(255)
  input_img = input_img.data.cpu().numpy()
  # b x c x h x w --> b x h x w x c
  input_img = np.transpose(input_img, (0, 2, 3, 1))
  return input_img


opt = TestALLOptions().parse()
opt.num_worker = opt.nThreads
opt.gpuid = opt.gpu_ids

# data path
img_file = './datasets/data/IER2/images'
mask_file = './datasets/data/IER2/masks'
showind2maskind_file = './datasets/data/IER2/show2mask'
feature_file = './datasets/data/IER2/features'
operator_file = './datasets/data/IER2/IER2.json'
vocab_dir = './datasets/data/language'

# set random seed
set_seed(opt.seed)

# load the dataset
opt.batch_size = opt.batchSize
dataset, dataloader = data.create_dataloader(opt)
ier_dataset_inference = IERDataset_inference(img_file, mask_file, showind2maskind_file, operator_file, feature_file, vocab_dir, opt.phase, opt)
test_loader_inference = DataLoader(ier_dataset_inference, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_worker, drop_last=False, collate_fn=collate_fn_inference)
print("test dataset: [%s] requests were created" % len(ier_dataset_inference))

opt.vocab_size = ier_dataset_inference.vocab_size
opt.op_vocab_size = ier_dataset_inference.op_vocab_size
opt.pan_dim = ier_dataset_inference.pan_dim
  
model = ALLModel(opt, ier_dataset_inference)
if len(opt.gpu_ids) > 0:
  model.cuda()
model.eval()

visualizer = Visualizer(opt)

# load checkpoint
if opt.load_checkpoint is not None:
  model.JointMatching.load_checkpoint(opt.load_checkpoint)


# create a webpage that summarizes the all results
if opt.ground_gt:
  phase = 'gt_' + opt.phase
else:
  phase = opt.phase
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, phase, opt.which_epoch))
abs_web_dir = os.path.abspath(web_dir)
json_dir = os.path.join(web_dir, 'jsons')
if not os.path.exists(json_dir):
  os.mkdir(json_dir)

# test
cnt = 0
l1_diff = 0
uid_map = {}
use_html = True

for i, data_inf in enumerate(tqdm(test_loader_inference)):
  if i * opt.batchSize >= opt.how_many: # opt.how_many
      break

  caption, caption_len, input_img_tensor, output_img_tensor, img, pfeats, lfeats, dif_lfeats, cxt_pfeats, cxt_lfeats, req, req_unique, op, op_unique, n, pair_id, req_id, op_req_id, masks_list, item = data_inf
  caption, caption_len, input_img_tensor, output_img_tensor, img, pfeats, lfeats, dif_lfeats, cxt_pfeats, cxt_lfeats, req, req_unique, op, op_unique, n, pair_id, req_id, op_req_id, item = \
    caption.cuda(), caption_len.cuda(), input_img_tensor.cuda(), output_img_tensor.cuda(), img.cuda(), pfeats.cuda(), lfeats.cuda(), dif_lfeats.cuda(), cxt_pfeats.cuda(), cxt_lfeats.cuda(), \
    req.cuda(), req_unique.cuda(), op.cuda(), op_unique.cuda(), n.cuda(), pair_id.cuda(), req_id.cuda(), op_req_id.cuda(), item.cuda()

  ns = n.cpu().tolist()
  n_accums = []
  n_accum = 0
  for n_i in ns:
    n_accum += n_i
    n_accums.append(n_accum)
  req_ids = req_id.cpu().tolist()
  req_ids_unique = []
  req_ids_start = []
  for req_idx, req_id in enumerate(req_ids):
    if req_idx == 0 or req_ids[req_idx] != req_ids[req_idx - 1]:
      req_ids_unique.append(req_id)
      req_ids_start.append(req_idx)
  op_req_ids = op_req_id.cpu().tolist()
  pair_ids = pair_id.cpu().tolist()
  caption_len = None
  for caption_i in caption:
    caption_len_i = (caption_i > 0).sum().unsqueeze(0)
    caption_len = caption_len_i if caption_len is None else torch.cat((caption_len, caption_len_i), 0)

  if not opt.ground_gt:

    max_lens = []
    for req_unique_i in req_unique:
      req_unique_i = req_unique_i.cpu().numpy()
      max_lens.append((req_unique_i != 0).sum())
    max_len = max(max_lens)
    req = req[:, : max_len]
    req_unique = req_unique[:, : max_len]

    with torch.no_grad():
      # classification
      probs, op_attn_unique = model.classify_forward(img, req_unique, op_unique)

      op_attn = None
      for n_idx, n_i in enumerate(ns):
        for n_i_idx in range(n_i):
          op_attn_single = op_attn_unique[n_idx].unsqueeze(0)
          op_attn = op_attn_single if op_attn is None else torch.cat((op_attn, op_attn_single), 0)

      # grounding
      scores, sub_attn, loc_attn, rel_attn, max_rel_ixs, weights = model.ground_forward(pfeats, lfeats, dif_lfeats, cxt_pfeats, cxt_lfeats, req, op_attn)

    pred_masks = []
    for probs_idx, probs_i in enumerate(probs):
      if float(probs_i) < 0.5:
        pred_mask = torch.ones(img.size()[2], img.size()[3]).cuda()
        pred_mask = (F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0), size=(opt.crop_size, opt.crop_size))).squeeze()
      else:
        thresh = np.array(0.25)
        if probs_idx == 0:
          scores_i = scores[: n_accums[probs_idx]]
          masks = masks_list[: n_accums[probs_idx]]
        else:
          scores_i = scores[n_accums[probs_idx - 1]: n_accums[probs_idx]]
          masks = masks_list[n_accums[probs_idx - 1]: n_accums[probs_idx]]
        masks = masks.cuda().float()
        scores_i = scores_i.cpu().numpy()
        preds = (scores_i > thresh).astype(int)
        pred_mask = masks[np.where(preds > 0)[0]].sum(0).clamp(0, 1)
      pred_masks.append(pred_mask)

  op_req_idx = 0
  data_ori = []
  for req_idx, req_id in enumerate(req_ids_unique):
    data_r = {}
    op_req_ids = test_loader_inference.dataset.IER.ReqId2OpReqId[req_id]
    label_tensor_inpaint_list = []
    label_tensor_retouch_list = []
    contain_coloerbg = False
    op_list = (torch.ones(len(test_loader_inference.dataset.IER.op_vocab2id)) * (-1)).cuda()
    op_cnt = 0
    for op_req_id in op_req_ids:
      if op_req_id in test_loader_inference.dataset.test_index_local or op_req_id in test_loader_inference.dataset.test_index_global:
        op_id = test_loader_inference.dataset.IER.OpReqId2OpId[op_req_id]
        op = test_loader_inference.dataset.IER.getOp[op_id]
        if opt.exclude_colorbg:
          assert test_loader_inference.dataset.IER.op_vocab2id[op] != 10
        # if train_loader_inference.dataset.IER.op_vocab2id[op] == 10:
        #   op_req_idx += 1
        #   contain_coloerbg = True
        #   continue
        if int(op_unique[op_req_idx]) == 7:
          label_tensor_list = label_tensor_inpaint_list
        else:
          label_tensor_list = label_tensor_retouch_list
        if opt.ground_gt:
          op_id = test_loader_inference.dataset.IER.OpReqId2OpId[op_req_id]
          pair_id = test_loader_inference.dataset.IER.OpId2PairId[op_id]
          local = test_loader_inference.dataset.ops[op_id]['is_local']
          gt_ann_ids = test_loader_inference.dataset.ops[op_id]['gt_anns'] if 'gt_anns' in test_loader_inference.dataset.ops[op_id] else []
          ann_ids = test_loader_inference.dataset.ops[op_id]['candidate_anns']
          assert not (local and len(gt_ann_ids) == 0)
          if not local:
            gt_mask = torch.ones(img.size()[2], img.size()[3]).cuda()
            gt_mask = (F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=(opt.crop_size, opt.crop_size))).squeeze()
          else:
            if op_req_idx == 0:
              masks = masks_list[: n_accums[op_req_idx]]
            else:
              masks = masks_list[n_accums[op_req_idx - 1]: n_accums[op_req_idx]]
            masks = masks.cuda().float()
            gts = np.array([int(ann_id in gt_ann_ids) for ann_id in ann_ids])
            gt_mask = masks[np.where(gts > 0)[0]].sum(0).clamp(0, 1)
          label_tensor_list.append(gt_mask)
        else:
          label_tensor_list.append(pred_masks[op_req_idx])
        op_list[op_cnt] = op_unique[op_req_idx]
        op_cnt += 1
        op_req_idx += 1

    if not contain_coloerbg:
      label_tensor_retouch = torch.zeros(opt.crop_size, opt.crop_size).cuda()
      label_tensor_inpaint = torch.zeros(opt.crop_size, opt.crop_size).cuda()
      for label_tensor in label_tensor_retouch_list:
        label_tensor_retouch += label_tensor
      label_tensor_retouch = label_tensor_retouch.clamp(0, 1) * 2
      for label_tensor in label_tensor_inpaint_list:
        label_tensor_inpaint += label_tensor
      data_r['label_list'] = [label_tensor_inpaint.unsqueeze(0), label_tensor_retouch.unsqueeze(0)]
      data_r['caption'] = caption[req_ids_start[req_idx]]
      data_r['caption_len'] = caption_len[req_ids_start[req_idx]]
      data_r['input_img'] = input_img_tensor[req_ids_start[req_idx]]
      data_r['output_img'] = output_img_tensor[req_ids_start[req_idx]]
      data_r['op_list'] = op_list
      data_r['req_id'] = torch.tensor(req_id).cuda()
      data_r['uid'] = item[req_ids_start[req_idx]]
      data_ori.append(data_r)

  data_i = data.collate_fn_caplen(data_ori)
  
  
  # attn list: [inpaint_attn, retouch_attn]
  generated, ret_dict = model.edit_forward(data_i, mode='inference')

  # op_attns, op_weights = op_attns_and_weights[0], op_attns_and_weights[1]   # (b, phrase_num, seq_len), (b, phrase_num)
  # JWT_VIS
  input_img, output_img, captions, req_ids = data_i['input_img'], data_i['output_img'], data_i['caption'], data_i['req_id']
  l1_diff += F.l1_loss(output_img.cuda(), generated, reduction='sum') / generated.shape[1] / generated.shape[2] / generated.shape[3]

  # (b x h x w x c)
  if use_html:
    if opt.spade_attn:
      # atten_list = [attn_map.sum(1).unsqueeze(1) for attn_map in ret_dict['attn_list']]
      atten_list = ret_dict['attn_list']

    for b in range(generated.shape[0]):
      uid_map[data_i['uid'][b].cpu().tolist()] = cnt
      with open(os.path.join(json_dir, str(cnt) + '.json'), 'w') as f:
        data_j = {}
        data_j['img0'] = os.path.join(abs_web_dir, 'images', 'input_image', str(cnt) + '.png')
        data_j['img1_syn'] = os.path.join(abs_web_dir, 'images', 'synthesized_image', str(cnt) + '.png')
        data_j['img1_gt'] = os.path.join(abs_web_dir, 'images', 'output_image', str(cnt) + '.png')
        data_j['sents'] = [test_loader_inference.dataset.IER.getReq[int(req_ids[b])]]
        data_j['uid'] = str(cnt)
        json.dump(data_j, f)

      visuals = OrderedDict([
        ('input_image', data_i['input_img'][b]),
        ('output_image', data_i['output_img'][b]),
        ('synthesized_image', generated[b]),
      ])

      if opt.spade_attn:
        atten_vis_list = []
        for atten_map_i in atten_list:
          atten_vis_list.append(vis_map(atten_map_i[b]))
        visuals['attn_map_0'] = atten_vis_list[0]
        # visuals['attn_map_1'] = atten_vis_list[1]
        # visuals['attn_map_2'] = atten_vis_list[2]
        # visuals['attn_map_3'] = atten_vis_list[3]
        # visuals['attn_map_4'] = atten_vis_list[4]
        # visuals['attn_map_5'] = atten_vis_list[5]

        # test weight for vis
        ori_val = atten_map_i[b]
        # visuals['atten_map_orival'] = vis_map(ori_val)
        for w in [1, 4, 8, 15, 20, 30, 50, 100, 200]:
          visuals[f'atten_map_w={w}'] = vis_map( torch.tan((torch.sigmoid(ori_val)-0.5) * w) )

      # cur_path = os.path.join(web_dir, str(cnt))
      cur_cap = captions[b].data.cpu().numpy()
      sentence = []
      for cap_idx in range(len(cur_cap)):
        if cur_cap[cap_idx] == 0:
          break
        word = ier_dataset_inference.ixtoword[cur_cap[cap_idx]].encode('ascii', 'ignore').decode('ascii')
        sentence.append(word)
      header = f"({' '.join(str(data_i['uid'][b].cpu().tolist()))}) ||  "
      header += f"Request: ({' '.join(sentence)})"
      if not opt.FiveK:
        operators = []
        # for label in data_i['label_list']:
        #   cur_op = label[b].max()
        #   if cur_op > 0:
        #     operators.append(use_op[cur_op - 1])
        for op_i in data_i['op_list'][b]:
          if int(op_i) != -1:
            operators.append(test_loader_inference.dataset.IER.id2op_vocab[int(op_i)])
        operators = list(set(operators))
        header += f"  ||  Operators: ({' , '.join(operators)})"

      visualizer.save_images(webpage, visuals, str(cnt), header)
      cnt += 1

  # else:
  #   input_img, output_img, fake_img = tensor_to_numpy(input_img), tensor_to_numpy(output_img), tensor_to_numpy(generated)
  #   row_len = 15
  #   gap = row_len * 2 + 10  # 15: each row; 10: begining
  #   # gap = 50
  #   text_bg = np.zeros((gap, 256 * 3, 3))
  #   res = np.zeros((1, 256 * 3, 3))
  #   for b in range(generated.shape[0]):
  #     cur_input_img, cur_output_img, cur_fake_img = input_img[b], output_img[b], fake_img[b]
  #     row = np.concatenate([cur_input_img, cur_output_img, cur_fake_img], 1)  # (h, w * 3, 3)
  #     row = np.concatenate([row, text_bg], 0)  # (h+gap, w * 3, 3)
  #
  #     cur_cap = captions[b].data.cpu().numpy()
  #     sentence = []
  #     for cap_idx in range(len(cur_cap)):
  #       if cur_cap[cap_idx] == 0:
  #         break
  #       word = dataset.ixtoword[cur_cap[cap_idx]].encode('ascii', 'ignore').decode('ascii')
  #       sentence.append(word)
  #
  #     text_color = (255, 0, 0)
  #
  #     cv2.putText(row, ' '.join(sentence), (5, 256 + 10), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, 1)
  #
  #     att_val_list = [str(attn_val.item() * 100)[:3] for attn_val in op_attns_and_weights[b][0]]
  #     cv2.putText(row, '  '.join(att_val_list), (5, 256 + 25), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, 1)
  #     att_val_list = [str(attn_val.item() * 100)[:3] for attn_val in op_attns_and_weights[b][1]]
  #     cv2.putText(row, '  '.join(att_val_list), (5, 256 + 40), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, 1)
  #     # for op_idx in range(opt.label_nc):
  #     #   op_attn_str = '  '.join([str(attn_val.item() * 100)[:3] for attn_val in op_attns[b, op_idx]])
  #     #   op_weight_str = str(op_weights[b, op_idx].item() * 100)[:3]
  #     #   cv2.putText(row, f'{use_op[op_idx]} ({op_weight_str}):  {op_attn_str}', (5, 256 + 25 + op_idx * row_len), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, 1)
  #     res = np.concatenate([res, row], 0)
  #
  #   # finish and write image
  #   cv2.imwrite(os.path.join(web_dir, f'G_jwtvis_{cnt}.png'), res[..., ::-1])  # rgb2bgr
  #   cnt += 1
with open(os.path.join(web_dir, 'uid_map.json'), 'w') as f:
  json.dump(uid_map, f)
webpage.save()
print(f'\n l1 diff: {l1_diff.item() / cnt}')





