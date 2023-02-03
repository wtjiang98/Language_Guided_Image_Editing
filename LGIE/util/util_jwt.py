
from cocoapi.PythonAPI.pycocotools import mask as cocomask
import numpy as np
import json

# with open('/mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip/masks/1ef146_1ef146_mask.json', 'r') as f:
#     mask = json.load(f)
#     rleObj = mask[0]
#     # rle = cocomask.frPyObjects(seg, mask[0]['size'][0], mask[0]['size'][1])
#     mask = cocomask.decode(rleObj)

unuse_op = ['crop', 'color_bg', 'black&white', 'rotate', 'flip', 'facet_blur', 'flip_obj', 'edge', 'rotate_obj']
unuse_gan_op = ['deform_obj', 'exposure', 'dehaze', 'gaussain_blur', 'denoise', 'radial_blur']

use_op = ['inpaint_obj', 'brightness', 'lightness', 'tint', 'hue', 'saturation', 'contrast', 'sharpness']   # v1 (4555 triplet)
# use_op = ['brightness', 'contrast', 'inpaint_obj']  # v3 (1755 triplet)
# use_op = ['brightness', 'contrast']     # v4 (178 triplet)

# op的id编号：0位无用，其他从1开始编号
op2ind = {op: ind+1 for ind, op in enumerate(use_op)}
# for op in unuse_op:
#     op2ind[op] = 0


def vis_dataset():
  with open('/mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/IER2.json', 'r') as f:
    # 把含有unuse_op的pair去掉
    anno = json.load(f)
    for the_op in use_op:
      cnt = 0
      sum = 0
      valid_image_cnt = 0
      fuck = 0

      for pair in anno:
        cur_ops = pair['operator']
        presum = sum
        precnt = cnt
        # sum += 1
        # for key in cur_ops:
        #     if key not in use_op:
        #         cnt += 1
        #         break

        # ************ 以下用于可视化数据 ************
        flag = True
        ids_ops = {}
        whole_img_other_op = False
        whole_img_that_op = False
        for key in cur_ops:
          if key in use_op:
            # if (cur_ops[key]['ids']) and (cur_ops[key]['local'] is False):
            #     fuck += 1
            # 当local为False，或local为True但ids为空（等同于local为False，有26个）
            if (cur_ops[key]['local'] is False) or ((not cur_ops[key]['ids']) and (cur_ops[key]['local'] is True)):
              if key != the_op:
                whole_img_other_op = True
              else:
                sum += 1
            else:
              for obj_id in cur_ops[key]['ids']:
                ids_ops[obj_id] = ids_ops.get(obj_id, []) + [key]
          else:
            flag = False
            break

        if flag:
          valid_image_cnt += 1
          for obj_id in ids_ops:
            obj_ops = ids_ops[obj_id]
            if the_op in obj_ops:
              if sum == presum:
                sum += 1
              if (len(obj_ops) == 1) and (not whole_img_other_op) and (cnt == precnt):
                cnt += 1
            # elif whole_img_that_op:
            #     sum += 1
      print(the_op)
      print(cnt, sum, cnt/sum)
    # print(fuck)
    # print(valid_image_cnt, sum, valid_image_cnt/sum)


def get_new_json():
  new_list = []
  with open('/mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/IER2.json', 'r') as f:
    # 把含有unuse_op的pair去掉
    anno = json.load(f)

    for pair in anno:
      cur_ops = pair['operator']
      flag = True
      for key in cur_ops:
          if key not in use_op:
              flag = False
              break

      if flag and cur_ops:  # 当cur_ops都为空，直接跳过
          new_list.append(pair)

    print(len(new_list))
    # write new json
    print(f'We now have {len(new_list)} images')
    with open('/mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_global.json', 'w') as f:
        json.dump(new_list, f)


if __name__ == '__main__':
  # vis_dataset()
  get_new_json()

