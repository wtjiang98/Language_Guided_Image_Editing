
import json
import cv2
import os
import numpy as np

data_path = '/mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected'
unuse_op = ['crop', 'color_bg', 'black&white', 'rotate', 'flip', 'facet_blur', 'flip_obj', 'edge', 'rotate_obj']
unuse_gan_op = ['deform_obj', 'exposure', 'dehaze', 'gaussain_blur', 'denoise', 'radial_blur']
use_op = ['inpaint_obj', 'brightness', 'lightness', 'tint', 'hue', 'saturation', 'contrast', 'sharpness']

IER2 = json.load(open(os.path.join(data_path, 'IER2.json'), 'r'))

def vis_diff():
    for n, img_pair in enumerate(IER2):
        cur_ops = img_pair['operator']
        flag = False
        for key in cur_ops:
            if key not in use_op:
                flag = True
                break
        if flag:
            continue

        input_img_name = img_pair['input'].replace('/', '_')
        output_img_name = img_pair['output'].replace('/', '_')
        input_img = cv2.imread(os.path.join(data_path, 'images', input_img_name))
        input_img = cv2.resize(input_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        output_img = cv2.imread(os.path.join(data_path, 'images', output_img_name))
        output_img = cv2.resize(output_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        res_img = np.abs(output_img.astype(np.int8) - input_img.astype(np.int8))
        res = res_img[:, :, 0] + res_img[:, :, 1] + res_img[:, :, 2]

        vis_map = np.zeros((input_img.shape[0], input_img.shape[1], 3), dtype=np.uint8)
        pos = np.where(res > 10)
        vis_map[pos[0], pos[1], :] = [0, 0, 255]
        vis_im = cv2.addWeighted(input_img, 0.6, vis_map, 0.4, 0)


        vis_im = cv2.resize(vis_im, (512, 512), interpolation=cv2.INTER_LINEAR)
        text_bg = np.zeros((60, 512 * 3, 3))
        result = np.concatenate([input_img, vis_im, output_img], 1)
        result = np.concatenate([result, text_bg], 0)
        text = img_pair['expert_summary'][0]

        cv2.putText(result, text, (40, 542), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(data_path, 'pair_vis', input_img_name+output_img_name.split('.')[0]+'.png'), result)


def vis_mask():

    img = cv2.imread(os.path.join('../data/IER2_for_zip/images', img_name))
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    vis_map = np.zeros((bimask.shape[0], bimask.shape[1], 3), dtype=np.uint8)
    pos = np.where(bimask == 1)
    vis_map[pos[0], pos[1], :] = [0, 0, 255]
    vis_map = cv2.resize(vis_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    vis_im = cv2.addWeighted(img, 0.6, vis_map, 0.4, 0)
    gt_im = cv2.imread(os.path.join('../data/IER2_for_zip/images', img_pair['output'].replace('/', '_')))
    gt_im = cv2.resize(gt_im, (512, 512), interpolation=cv2.INTER_LINEAR)
    text_bg = np.zeros((60, 512 * 3, 3))
    res = np.concatenate([img, vis_im, gt_im], 1)
    res = np.concatenate([res, text_bg], 0)
    text = img_pair['expert_summary'][0]
    cv2.putText(res, text, (40, 542), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
    cv2.imwrite(os.path.join('../data/gier_pair_vis/vis_mask', ref['file_name'].split('.')[0] + '.png'), res)