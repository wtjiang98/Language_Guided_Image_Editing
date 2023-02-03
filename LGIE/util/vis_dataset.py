
import json
import pickle
import numpy as np
# import sys
# sys.path.append('..')
# sys.path.append('../external/cocoapi/PythonAPI')
# from pycocotools import mask as cocomask
# from util import data_reader
import cv2
import os
from tqdm import tqdm
from cocoapi.PythonAPI.pycocotools import mask as cocomask

def gaochen():
  instances = json.load(open('../external/refer/data/gier/instances.json', 'r'))
  data = {}
  data['refs'] = pickle.load(open('../external/refer/data/gier/refs(gc).p', 'rb'))
  IER2 = json.load(open('../data/IER2_for_zip/IER2.json', 'r'))

  with open('../data/IER2_for_zip/masks/25zgyw_25zgyw_mask.json', 'r') as f:
    mask = json.loads(f.read())

  for n, ref in enumerate(data['refs']):
    im = cv2.imread(os.path.join('../data/IER2_for_zip/images', ref['file_name']))
    gt_im = cv2.imread(os.path.join('../data/IER2_for_zip/images', ref['gt_im_name']))

    mask_rle = instances['annotations'][ref['ann_id']]['segmentation']
    mask_rle['counts'] = mask_rle['counts'].encode(encoding="utf-8")  # str to bytes
    bimask = cocomask.decode(mask_rle)
    vis_map = np.zeros((bimask.shape[0], bimask.shape[1], 3), dtype=np.uint8)
    pos = np.where(bimask == 1)
    vis_map[pos[0], pos[1], :] = [0, 0, 255]
    vis_im = cv2.addWeighted(im, 0.6, vis_map, 0.4, 0)

    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
    vis_im = cv2.resize(vis_im, (512, 512), interpolation=cv2.INTER_LINEAR)
    gt_im = cv2.resize(gt_im, (512, 512), interpolation=cv2.INTER_LINEAR)
    text_bg = np.zeros((60, 512*3, 3))
    res = np.concatenate([im, vis_im, gt_im], 1)
    res = np.concatenate([res, text_bg], 0)
    text = ref['sentences'][0]['raw']
    cv2.putText(res, text, (40, 542), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)

    cv2.imwrite(os.path.join('../data/gier_pair_vis', ref['file_name'].split('.')[0]+'.png'), res)

    print('done')
    # reader = data_reader.DataReader('../gier/train_batch', 'gier_train')
    # batch = reader.read_batch(is_log=False)
    # text = batch['text_batch']
    # im = batch['im_batch'].astype(np.float32)
    # mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)
    # im = (im[:, :, ::-1] * 255).astype(np.uint8)

def check_path(path):
  if not os.path.exists(path):
    os.makedirs(path)


def vis_MIT_Fivek():
  image_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images'
  op_v1_path = '/mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json'
  result_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/vis_dataset'
  check_path(result_dir)

  with open(op_v1_path, 'r') as f:
    anno_dict = json.load(f)
  for anno in tqdm(anno_dict):
    input_fn = anno['input'].split('.')[0]

    # load image and mask
    input_img = cv2.imread(os.path.join(image_dir, anno['input'].replace('/', '_')))
    input_img = cv2.resize(input_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    output_img = cv2.imread(os.path.join(image_dir, anno['output'].replace('/', '_')))
    output_img = cv2.resize(output_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    gap = 50
    text_bg = np.zeros((gap, 512 * 2, 3))

    res = np.concatenate([input_img, output_img], 1)  # (512, 512 * 3, 3)
    res = np.concatenate([res, text_bg], 0)  # (572, 512 * 3, 3)
    text = anno['request']
    cv2.putText(res, text, (40, 532), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)

    # for cap in anno['expert_summary']:
    #   cv2.putText(res, 'expert: ' + cap, (15, 532 + cnt * gap), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
    #   cnt += 1
    # for cap in anno['amateur_summary']:
    #   cv2.putText(res, 'amateur: ' + cap, (15, 532 + cnt * gap), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
    #   cnt += 1

    cv2.imwrite(os.path.join(result_dir, anno['output']), res)


def vis_dataset():
  """对数据库进行可视化，对每一个operator单独创建一张图"""
  # anno_path = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/IER2.json'
  # op_v1_path = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json'
  # mask_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks'
  # image_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images'
  # result_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/vis_dataset'

  op_v1_path = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json'
  mask_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks'
  image_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images'
  result_dir = '/mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/vis_dataset'

  check_path(result_dir)

  with open(op_v1_path, 'r') as f:
    anno_dict = json.load(f)
  for anno in tqdm(anno_dict):
    input_fn = anno['input'].split('/')[0]
    check_path(os.path.join(result_dir, input_fn))

    for op_name in anno['operator']:
      # load image and mask
      input_img = cv2.imread(os.path.join(image_dir, anno['input'].replace('/', '_')))
      input_img = cv2.resize(input_img, (512, 512), interpolation=cv2.INTER_LINEAR)
      with open(os.path.join(mask_dir, f'{input_fn}_{input_fn}_mask.json'), 'r') as f:
        mask_list = json.load(f)

      if anno['operator'][op_name]['local'] is True:
        if anno['operator'][op_name]['ids']:                    # 这里的ids可能为空，为空则表示全图
          mask = np.zeros((input_img.shape[0], input_img.shape[1]))
          for obj_id in anno['operator'][op_name]['ids']:     # op可能有多个操作对象
            rleObj = mask_list[obj_id]
            obj_mask = cv2.resize(cocomask.decode(rleObj), (512, 512), interpolation=cv2.INTER_NEAREST) # cv2.INTER_LINEAR
            mask += obj_mask                                # mask直接累加，在下面判断改为 >= 1
        else:
          mask = np.ones((input_img.shape[0], input_img.shape[1]))
      else:
        mask = np.ones((input_img.shape[0], input_img.shape[1]))

      vis_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
      pos = np.where(mask >= 1)                                    # >= 1
      assert pos, 'pos should not be none'

      vis_map[pos[0], pos[1], :] = [0, 0, 255]
      vis_map = cv2.resize(vis_map, (512, 512), interpolation=cv2.INTER_LINEAR)
      vis_im = cv2.addWeighted(input_img, 0.5, vis_map, 0.5, 0)
      output_img = cv2.imread(os.path.join(image_dir, anno['output'].replace('/', '_')))
      output_img = cv2.resize(output_img, (512, 512), interpolation=cv2.INTER_LINEAR)
      cnt = 0
      gap = 30
      cap_num = len(anno['expert_summary']) + len(anno['amateur_summary'])
      text_bg = np.zeros((gap * cap_num, 512 * 3, 3))

      res = np.concatenate([input_img, vis_im, output_img], 1)    # (512, 512 * 3, 3)
      res = np.concatenate([res, text_bg], 0)                     # (572, 512 * 3, 3)
      # text = anno['expert_summary'][0]
      # cv2.putText(res, text, (40, 532), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1.3)

      for cap in anno['expert_summary']:
        cv2.putText(res, 'expert: ' + cap, (15, 532 + cnt * gap), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
        cnt += 1
      for cap in anno['amateur_summary']:
        cv2.putText(res, 'amateur: ' + cap, (15, 532 + cnt * gap), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 1)
        cnt += 1
      cv2.imwrite(os.path.join(result_dir, input_fn, f'{op_name}.png'), res)


if __name__ == '__main__':
  # vis_dataset()
  vis_MIT_Fivek()





