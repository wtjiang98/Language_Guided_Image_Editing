import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from util.vis_atten.color_maps import get_afmhot, gray2color, get_jet
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from matplotlib import cm


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))


    return colormap_int

def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

    # color_image = Image.fromarray(color_array)

    return color_array




image_index = 0
img_size = 256
atten_size = 64

def tensor2gray(x):
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    x = (x * 255).astype('uint8')
    return x

def attn_merge_im(im, attn, alpha):
    color_map = get_jet()
    attn_rgb = gray2color(attn, color_map)
    attn_im = np.clip(attn_rgb * alpha + im, 0, 255).astype('uint8')
    return attn_im

def rand_point(point_num):
    point_list = []
    all_point = [i for i in range(img_size * img_size)]
    points = np.random.choice(all_point, point_num)
    for i in range(point_num):
        tmp_x = points[i] // img_size
        tmp_y = points[i] % img_size
        point_list.append((tmp_x, tmp_y))
    return point_list



def save_attn_before_after(attn1, attn2):
    global image_index
    raw_path = '/home/htr/project/NeurlPS2019/code/PyTorch-Encoding/runs/raw_images'
    directory = '/home/htr/project/NeurlPS2019/code/PyTorch-Encoding/runs/pcontext/multi_nl_fcn/multi_simple_nonlocal_extend_aggre_1x_r50/vis'
    # print("visualize directory : ", directory)
    if not os.path.exists(directory):
        os.mkdir(directory)

    im = np.load(os.path.join(raw_path, '{}.npy'.format(str(image_index))))
    # print("load successfully", im.shape)
    avg = [.485, .456, .406]
    std = [.229, .224, .225]
    im = im.transpose(1, 2, 0) * std + avg
    im = im * 255
    raw_im = im.astype('uint8')
    raw_im = Image.fromarray(raw_im)
    raw_im = raw_im.resize((img_size, img_size))

    attn1 = attn1.sum(dim=1, keepdim=True)
    attn2 = attn2.sum(dim=1, keepdim=True)
    # print("attn shape", attn1.size())
    attn1_ims = F.interpolate(attn1, scale_factor=4, mode='bilinear').squeeze()
    attn2_ims = F.interpolate(attn2, scale_factor=4, mode='bilinear').squeeze()
    # print("attn shape", attn1_ims.size())

    alpha = 0.5
    tmp1 = attn1_ims.cpu().data.numpy()
    tmp1 = tensor2gray(tmp1).astype('uint8')
    mid_1 = np.median(tmp1)
    tmp1 = np.clip(tmp1, a_min=mid_1, a_max=100000).astype('uint8')
    tmp1 = attn_merge_im(raw_im, tmp1, alpha)

    tmp2 = attn2_ims.cpu().data.numpy()
    mid_2 = np.median(tmp2)
    tmp2 = np.clip(tmp2, a_min=mid_2, a_max=100000)
    tmp2 = tensor2gray(tmp2).astype('uint8')
    tmp2 = attn_merge_im(raw_im, tmp2, alpha)


    target = Image.new(mode='RGB', size=(240*3+10*2, 240), color=(255, 255, 255))
    target.paste(raw_im, (0, 0))
    target.paste(Image.fromarray(tmp1), (250, 0))
    target.paste(Image.fromarray(tmp2), (500, 0))
    target_dir = os.path.join(directory, 'before_after_median')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    # print(os.path.join(target_dir, '{}.png'.format(str(image_index))))
    target.save(os.path.join(target_dir, '{}.png'.format(str(image_index))))
    image_index += 1
    # input()
    # print("iamge_index:", image_index)

pic_cnt = 0


def vis_map(attn):
    """
    img: numpy
    x_sim: (1, HW, HW)
    x:  (1, c, fea_size, fea_size)
    :return:
    """
    # im = np.load(os.path.join(raw_path, '{}.npy'.format(str(image_index))))
    # # print("load successfully", im.shape)
    # avg = [.485, .456, .406]
    # std = [.229, .224, .225]
    # im = im.transpose(1, 2, 0) * std + avg
    # im = im * 255
    # raw_im = im.astype('uint8')
    # raw_im = Image.fromarray(raw_im)
    # raw_im = raw_im.resize((img_size, img_size))
    # x_sum = x.sum(dim=1, keepdim=True)   # [1, 1, 3600]
    #
    # alpha = 0.5
    # point_num = atten_size * atten_size
    # # point = rand_point(point_num)
    # # point = [i for i in range(0 + point_num)]
    # src_im = src_im.transpose(1, 2, 0).astype('uint8')
    # ref_im = ref_im.transpose(1, 2, 0).astype('uint8')
    # src_im = Image.fromarray(src_im)
    # ref_im = Image.fromarray(ref_im)
    #
    # # range(36*64, 38*64)
    #
    # for i in tqdm(range(17*64, 30*64)): # 扫过眼睛部分
    #     # if i < 4094: continue
    #     (point_x, point_y) = i // atten_size, i % atten_size        # 每个图生成64*64个
    #     index = point_x * atten_size + point_y
    #
    #     # tmp_attn = x_sim[0, index, :] * x_sum   # [1, 1, 3600]
    #     tmp_attn = x_sim[2, index]                                  # get a pixel from skin
    #     if tmp_attn.max() == 0:
    #         continue
    #     tmp_attn = tmp_attn.view(1, 1, atten_size, atten_size)      #  similarity matrix of one pixel   [1, 1, 64, 64]
    attn = attn.unsqueeze(1)
    tmp_attn = F.interpolate(attn, (img_size, img_size), mode='bilinear').squeeze()       # [256, 256]
    tmp_attn = tmp_attn.cpu().data.numpy()
    # hist, bin_edges = np.histogram(tmp_attn)
    # print("hist", hist)
    # print("bin_edges", bin_edges)
    tmp_attn = np.clip(tmp_attn, a_min=-100000, a_max=100000)
    tmp_attn = tensor2gray(tmp_attn).astype('uint8')

    color_map = get_jet()
    attn_rgb = gray2color(tmp_attn, color_map)
    attn_im = np.clip(attn_rgb, 0, 255).astype('uint8')

    return attn_im


def save_attn_point(src_im, ref_im, x_sim, x, task_path):
    """
    img: numpy
    x_sim: (1, HW, HW)
    x:  (1, c, fea_size, fea_size)
    :return:
    """
    global image_index, pic_cnt
    # raw_path = '/home/htr/project/NeurlPS2019/code/PyTorch-Encoding/runs/raw_images'
    # 由这里控制！！直接新建了一个目录
    res_path = os.path.join(task_path, f'debug_eye_vis_atten_{config.w_visual}', f'pic_{str(pic_cnt)}')
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # im = np.load(os.path.join(raw_path, '{}.npy'.format(str(image_index))))
    # # print("load successfully", im.shape)
    # avg = [.485, .456, .406]
    # std = [.229, .224, .225]
    # im = im.transpose(1, 2, 0) * std + avg
    # im = im * 255
    # raw_im = im.astype('uint8')
    # raw_im = Image.fromarray(raw_im)
    # raw_im = raw_im.resize((img_size, img_size))
    # x_sum = x.sum(dim=1, keepdim=True)   # [1, 1, 3600]

    alpha = 0.5
    point_num = atten_size * atten_size
    # point = rand_point(point_num)
    # point = [i for i in range(0 + point_num)]
    src_im = src_im.transpose(1, 2, 0).astype('uint8')
    ref_im = ref_im.transpose(1, 2, 0).astype('uint8')
    src_im = Image.fromarray(src_im)
    ref_im = Image.fromarray(ref_im)

    # range(36*64, 38*64)

    for i in tqdm(range(17*64, 30*64)): # 扫过眼睛部分
        # if i < 4094: continue
        (point_x, point_y) = i // atten_size, i % atten_size        # 每个图生成64*64个
        index = point_x * atten_size + point_y

        # tmp_attn = x_sim[0, index, :] * x_sum   # [1, 1, 3600]
        tmp_attn = x_sim[2, index]                                  # get a pixel from skin
        if tmp_attn.max() == 0:
            continue
        tmp_attn = tmp_attn.view(1, 1, atten_size, atten_size)      #  similarity matrix of one pixel   [1, 1, 64, 64]
        tmp_attn = F.interpolate(tmp_attn, (img_size, img_size), mode='bilinear').squeeze()       # [256, 256]
        tmp_attn = tmp_attn.cpu().data.numpy()
        # hist, bin_edges = np.histogram(tmp_attn)
        # print("hist", hist)
        # print("bin_edges", bin_edges)
        tmp_attn = np.clip(tmp_attn, a_min=0, a_max=100000)
        tmp_attn = tensor2gray(tmp_attn).astype('uint8')
        tmp_attn = attn_merge_im(ref_im, tmp_attn, alpha)

        src_im_cp = np.asarray(src_im).copy()
        src_im_cp[point_x * 4:(point_x + 1) * 4, point_y * 4:(point_y + 1) * 4, 0] = 255
        src_im_cp[point_x * 4:(point_x + 1) * 4, point_y * 4:(point_y + 1) * 4, 1] = 0
        src_im_cp[point_x * 4:(point_x + 1) * 4, point_y * 4:(point_y + 1) * 4, 2] = 0
        Image.fromarray(src_im_cp).save(os.path.join(res_path, f'src_{point_x}_{point_y}.png'))
        Image.fromarray(tmp_attn).save(os.path.join(res_path, f'atten_{point_x}_{point_y}.png'))
        print(os.path.join(res_path, f'atten_{point_x}_{point_y}.png'))
        image_index += 1

    pic_cnt += 1

    # target = Image.new(mode='RGB', size=(img_size*4+10*3, img_size*2+10), color=(255, 255, 255))
    # target.paste(src_im, (0, 0))
    # target.paste(Image.fromarray(attn_maps[0]), (250, 0))
    # target.paste(Image.fromarray(attn_maps[1]), (500, 0))
    # target.paste(Image.fromarray(attn_maps[2]), (750, 0))
    # target.paste(Image.fromarray(attn_maps[3]), (0, 250))
    # target.paste(Image.fromarray(attn_maps[4]), (250, 250))
    # target.paste(Image.fromarray(attn_maps[5]), (500, 250))
    # target.paste(Image.fromarray(attn_maps[6]), (750, 250))


# from config import *
if __name__ == '__main__':
    # set_ref_im(1)
    # print(get_ref_im())
    attn = torch.Tensor((1, 64, 64))
    vis_map(attn)





