# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
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

def tensor_to_numpy(input_img):
  # [-1, 1] --> [0, 1]
  input_img.add_(1).div_(2).mul_(255)
  input_img = input_img.data.cpu().numpy()
  # b x c x h x w --> b x h x w x c
  input_img = np.transpose(input_img, (0, 2, 3, 1))
  return input_img


opt = TestOptions().parse()
dataset, dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt, dataset)
model.eval()
visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
cnt = 0
l1_diff = 0
use_html = True

def logit(p):
  return torch.log(p / (1 - p))

for i, data_i in enumerate(tqdm(dataloader)):
  if i * opt.batchSize >= opt.how_many: # opt.how_many
      break

  # attn list: [inpaint_attn, retouch_attn]
  generated, ret_dict = model(data_i, mode='inference')

  # op_attns, op_weights = op_attns_and_weights[0], op_attns_and_weights[1]   # (b, phrase_num, seq_len), (b, phrase_num)
  # JWT_VIS
  input_img, output_img, captions = data_i['input_img'], data_i['output_img'], data_i['caption']
  l1_diff += F.l1_loss(output_img.cuda(), generated, reduction='sum') / generated.shape[1] / generated.shape[2] / generated.shape[3]
  # l1_diff += F.l1_loss(output_img.cuda(), generated)
  # (b x h x w x c)
  if use_html:
    if opt.spade_attn:
      # atten_list = [attn_map.sum(1).unsqueeze(1) for attn_map in ret_dict['attn_list']]
      atten_list = ret_dict['attn_list']

    for b in range(generated.shape[0]):
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
      # for w in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
      for w in [10, 6, 3, 2, 0.1, 0.01]:
        visuals[f'atten_map_w={w}'] = vis_map( torch.sigmoid(ori_val * w) )

        # cur_path = os.path.join(web_dir, str(cnt))
      cur_cap = captions[b].data.cpu().numpy()
      sentence = []
      for cap_idx in range(len(cur_cap)):
        if cur_cap[cap_idx] == 0:
          break
        word = dataset.ixtoword[cur_cap[cap_idx]].encode('ascii', 'ignore').decode('ascii')
        sentence.append(word)
      header = f"Request: ({' '.join(sentence)})"
      if not opt.FiveK:
        operators = []
        for label in data_i['label_list']:
          cur_op = label[b].max()
          if cur_op > 0:
            operators.append(use_op[cur_op - 1])
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

webpage.save()
print(f'\n l1 diff: {l1_diff.item() / cnt}')
print(cnt)





