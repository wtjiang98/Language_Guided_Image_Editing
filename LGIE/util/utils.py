import os
import json
import h5py
import pdb
import hashlib
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import cv2
import random
from torch.autograd import Variable

def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(cmd_path, op_path):
    vocab = {}
    with open(cmd_path, 'r') as f:
        cmd_vocab = json.load(f)
    with open(op_path, 'r') as f:
        op_vocab = json.load(f)
    vocab['command_token_to_idx'] = cmd_vocab
    vocab['operator_token_to_idx'] = op_vocab
    vocab['command_idx_to_token'] = invert_dict(cmd_vocab)
    vocab['operator_idx_to_token'] = invert_dict(op_vocab)
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['command_token_to_idx']['<NONE>'] == 0
    assert vocab['command_token_to_idx']['<START>'] == 1
    assert vocab['command_token_to_idx']['<END>'] == 2
    assert vocab['operator_token_to_idx']['<NONE>'] == 0
    assert vocab['operator_token_to_idx']['<START>'] == 1
    assert vocab['operator_token_to_idx']['<END>'] == 2
    return vocab


def load_embedding(path):
    data = h5py.File(path, 'r')
    glv = data['glove'][()]
    return torch.tensor(glv)

###################################
### functions for seq2seq model ###
###################################

def decode_str(x, y, vocab, start, end):
    """
    :param x: [ndarray] (bs, cmd_len)
    :param y: [ndarray] (bs, 1)
    :param vocab:
    :param start [ndarray] (bs, 1)
    :param end [ndarray] (bs, 1)
    :return: sents: list of none phrase. len(sents) = bs
    :return: ops: list of operator names. len(ops) = bs
    """
    def trim(arr):
        """
        :param arr: (len,)
        """
        s = np.where(arr == 1)[0]
        if s.any():
            s = s[0]
        else:
            s = 0
        e = np.where(arr == 2)[0]
        if e.any():
            e = e[0]
        else:
            e = len(arr) - 1
        assert e >= s, 'y is not correct to decode'
        out = arr[s + 1:e]
        return out

    bs = len(y)
    cmd_vocab = vocab['command_idx_to_token']
    op_vocab = vocab['operator_idx_to_token']
    sents = []
    ops_name = []
    for bs_i in range(bs):
        cmd = x[bs_i]
        if start == 0 or end == 0:
            cmd = []
        elif start == -1 or end == -1:
            cmd = trim(cmd)
        else:
            cmd = trim(cmd[start, end + 1])
        sent = ' '.join([cmd_vocab[idx] for idx in cmd])
        sents.append(sent)

        op = int(y[bs_i])
        op_name = op_vocab[op]
        ops_name.append(op_name)
    return sents, ops_name


def get_opname(idx, vocab):
    """
    change index to operator name
    :param idx: ndarray (bs, 1)
    :return: ops_name: list of names
    """
    bs, _ = idx.shape
    op_vocab = vocab['operator_idx_to_token']
    ops_name = []
    for i in range(bs):
        op_name = op_vocab[int(idx[i])]
        ops_name.append(op_name)
    return ops_name

def get_phrase(idx, vocab, start, end):
    """  Deprecated
    get phrase from input sentence index.
    if start = end = 0: global mask, return phrase ''
    if start = end = -1: invalid input, just return the whole sentence
    if start, end >= 0: select the phrase

    :param x: [ndarray] (bs, cmd_len)
    :param vocab:
    :param start [ndarray] (bs, 1)
    :param end [ndarray] (bs, 1)
    :return sents: list of noun phrases.
    """

    def trim(arr):
        """
        :param arr: (len,)
        """
        s = np.where(arr == 1)[0]
        if s.any():
            s = s[0]
        else:
            s = 0
        e = np.where(arr == 2)[0]
        if e.any():
            e = e[0]
        else:
            e = len(arr) - 1
        assert e >= s, 'y is not correct to decode'
        out = arr[s + 1:e]
        return out

    cmd_vocab = vocab['command_idx_to_token']
    bs, _ = idx.shape
    sents = []
    for bs_i in range(bs):
        cmd = idx[bs_i]
        if start == 0 or end == 0:
            cmd = []
        elif start == -1 or end == -1:
            cmd = trim(cmd)
        else:
            cmd = trim(cmd[start, end + 1])
        sent = ' '.join([cmd_vocab[idx] for idx in cmd])
        sents.append(sent)
    return sents

#################################
### functions for web display ###
#################################

def save_img(img, name):
    """
    save the image and name
    :param img: (3, h ,w)
    :return:
    """
    img = img.transpose((1, 2, 0)) * 255
    # RGB2BGR for cv2 saving
    img = img.astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(name, img)


def name_encode(name, trunc_num=4):
    sha1 = hashlib.sha1()
    sha1.update(name.encode('utf-8'))
    res = sha1.hexdigest()
    return res[:trunc_num]


def update_web_row_s(webpage, img_x, img_y, param, iter, operators, img_dir, rewards=None, isGT=False):
    """ update single row of the data and save the image result with supervision
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, valid_op_len, h, w)
    :param param: (1, valid_op_len, 1)
    :param iter: [int]
    :param operators: list of predicted operators
    :param: rewards [dict]: keys: 'rewards', 'operator_rewards', 'image_rewards', values (1, valid_op_len)
    :param isGT: [bool]
    """
    param = param[0]
    if rewards is not None:
        pass
    descs = ['input']
    descs += operators
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        reward_str = 'r:{:.2f} img_r:{:.2f} op_r:{:.2f}'.\
            format(rewards['rewards'][0][i-1], rewards['image_rewards'][0][i-1],
                   rewards['operator_rewards'][0][i-1]) if rewards is not None else ''
        param_str = ' (' + ', '.join(['{:.2f}'.format(v) for v in param[i - 1]]) + ') ' + reward_str
        descs[i] = descs[i] + param_str
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)

def update_web_row_u(webpage, img_x, img_y, iter, img_dir, isGT=False):
    """ update single row of the data and save the image result unsupervisely
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, 1, 3, h, w)
    :param iter: [int]
    :param img_dir: image root directory
    :param isGT: [bool]
    """
    descs = ['input']
    descs += ['output']
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)


def update_web_row_sm(webpage, img_x, img_y, iter, operators, img_dir, isGT=False):
    """update single row of the data and save the image result in semi-supervisely
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, 1, 3, h, w)
    :param iter: [int]
    :param operators: list of predicted operators (gt_op_len)
    :param img_dir: [str]
    :param isGT: [bool]
    """
    descs = ['input']
    descs += [';'.join(operators)]
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)



def update_web_row_attn(webpage, attns, cmd, ops, img_dir):
    """
    update the visualization of the attention
    :param webpage:
    :param attns: (1, op_len, sent_len), sent_len (<START> ... <END>), op_len(...<END>)
    :param cmd: list of word
    :param ops: list of word
    :param img_dir: image directory
    """
    attns = attns.cpu().numpy()[0] # (op_len, sent_len)
    op_len, cmd_len = attns.shape
    cmd = (['<START>'] + cmd + ['<END>'])[:cmd_len]
    ops = (ops + ['<END>'])[:op_len]
    save_name = name_encode(', '.join(cmd), 10) + '.png'
    save_path = os.path.join(img_dir, save_name)
    showAttention(cmd, ops, attns, save_path)
    webpage.add_images([save_name], ['attention'], [save_name], width=512)



# Visualize attention
def showAttention(input_sentence, output_words, attentions, save_path):
    """
    visualize attention
    :param input_sentence: list of token
    :param output_words: list of token
    :param attentions: (output_len, input_len) ndarray
    :return:
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(save_path)


def idx2cmd(x, vocab):
    x = x[0]
    try:
        end = np.where(x == 2)[0][0]
    except:
        pdb.set_trace()
    x = x[1:end]
    vocab = vocab['command_idx_to_token']
    sent = ' '.join([vocab[i] for i in x])
    return sent

def idx2op(x, vocab):
    """
    x symbol: no matter with or not with start symbol.
    :param x:
    :param vocab:
    :return:
    """
    x = x[0]
    start = np.where(x == 1)[0]
    start = start[0] + 1 if len(start) > 0 else 0
    end = np.where(x == 2)[0]
    end = end[0] if len(end) > 0 else len(x)
    x = x[start:end]
    vocab = vocab['operator_idx_to_token']
    ops = [vocab[i] for i in x]
    return ops

def update_web(webpage, x, y, img_x, img_y, img_pred, param_gt, param_pred, iter, symbol, vocab, img_dir, supervise, rewards=None, attns=None):
    """
    update one data
    :param: x (1, max_seq_len)
    :param: y (1, max_op_len)
    :param: img_x (1, 3, h, w) \in [0, 1]
    :param: img_y (1, gt_op_len, 3, h, w) \in [0, 1]. gt_op_len = 1 if not fully supervised, otherwise valid_op_len
    :param: img_pred (1, valid_op_len, 3, h, w) \in [0, 1]
    :param: param_gt (1, gt_op_len, 1) if fully supervised, otherwise None.
    :param: param_pred (1, valid_op_len, 1)
    :param: symbol (1, valid_op_len)
    :param: vocab [dict]
    :param: img_dir [str]
    :param: supervise [int]: 0: no gt_op, no gt_param; 1: has gt_op, no gt_param; 2: has gt_op, has gt_param
    :param: rewards [dict]: keys: 'rewards', 'operator_rewards', 'image_rewards', values (1, valid_op_len)
    """
    cmd = idx2cmd(x, vocab)
    operators_pred = idx2op(symbol, vocab)
    webpage.add_header('iter {:5d}: {}'.format(iter, cmd))
    update_web_row_s(webpage, img_x, img_pred, param_pred, iter, operators_pred, img_dir, rewards=rewards,
                     isGT=False)
    if supervise == 2:
        operators = idx2op(y, vocab)
        update_web_row_s(webpage, img_x, img_y, param_gt, iter, operators, img_dir, isGT=True)
    elif supervise == 1:
        operators = idx2op(y, vocab)
        update_web_row_sm(webpage, img_x, img_y, iter, operators, img_dir, isGT=True)
    elif supervise == 0:
        update_web_row_u(webpage, img_x, img_y, iter, img_dir, isGT=True)
    if attns is not None:
        update_web_row_attn(webpage, attns, cmd.split(), operators_pred, img_dir)


"""
Required By MAttNet Code
"""

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            try:
                param.grad.data.clamp_(-grad_clip, grad_clip)
            except:
                pdb.set_trace()

# adapt for gan
def clip_gradient_with_judge(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

from PIL import Image, ImageDraw, ImageFont
from panopticapi.utils import IdGenerator, rgb2id

def visualize_mask(masks, boxes, clss, scores):
    """
    :param masks: list of (h, w)
    :param boxes: list of (x1, y1, x2, y2)
    :param clss: (n,)
    :param scores: (n,)
    :return: Image object
    """
    categories = {el['id']: el for el in CATEGORIES}
    color_generator = IdGenerator(categories)
    h, w = masks[0].shape
    pan_format = np.zeros((h, w, 3), dtype=np.uint8)
    # add mask with color to pan_format
    for mask, cls in zip(masks, clss):
        pan_format[mask.astype(bool)] = color_generator.categories[cls]['color']
    pan_format = Image.fromarray(pan_format)
    draw = ImageDraw.Draw(pan_format)
    for mask, box, cls, score in zip(masks, boxes, clss, scores):
        mask = (mask * 255).astype(np.uint8)
        mask_h, mask_w = mask.shape
        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        c_areas = [cv2.contourArea(c) for c in contour]
        x1s, y1s, x2s, y2s = [], [], [], []
        for i, c in enumerate(contour):
            # eliminate the wrong masks patches
            if c_areas[i] < len(c) and c_areas[i] < max(c_areas) * 0.1 and CATEGORIES[cls]['isthing']:
                cv2.drawContours(cv2.UMat(mask), [c], -1, 0, -1)
                continue
            x1s.append(c[:, 0, 0].min())
            y1s.append(c[:, 0, 1].min())
            x2s.append(c[:, 0, 0].max())
            y2s.append(c[:, 0, 1].max())

            c = c.reshape(-1).tolist()
            if len(c) < 4:
                print('warning: invalid contour')
                continue
            draw.line(c, fill='white', width=2)

        box = [min(x1s), min(y1s), max(x2s), max(y2s)]

        # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", int(mask_h*0.04))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", int(mask_h * 0.04))

        draw.rectangle([box[0], box[1], box[2] + 1, box[3] + 1], outline=(255, 255, 255))
        draw.rectangle([box[0] + 1, box[1] + 1, box[2], box[3]], outline=tuple(color_generator.categories[cls]['color']))
        if score < 0.99:
            text_size = draw.textsize(str(score), font=font)
            # judge text location
            if (box[2] * box[3]) / (text_size[0] * text_size[1]) > 10: # text inside box
                x0, y0 = box[0], box[1]
            else: # text outside box
                x0, y0 = box[0], box[1] - text_size[1]
            draw.rectangle((x0 + 1, y0 + 1, x0 + text_size[0] + 1, y0 + text_size[1] + 1), outline=tuple(color_generator.categories[cls]['color']), fill=tuple(color_generator.categories[cls]['color']))
            draw.text((x0 + 1, y0), str(score), font=font)
    # pan_format = np.array(pan_format)

    # get palette
    unq_clss = np.unique(clss)
    colors = [CATEGORIES[cls_ind]['color'] for cls_ind in unq_clss]
    cls_names = [CATEGORIES[cls_ind]['name'] for cls_ind in unq_clss]
    palette = get_palette(colors, cls_names)
    return pan_format, palette

def get_palette(colors, names):
    """
    :return: Image object
    """
    img_h, img_w = 512, 512

    bh_max = 30
    bh = img_h // len(names)
    bh = bh if bh < bh_max else bh_max
    bw = 100

    im = Image.fromarray(np.ones((img_h, img_w, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(im)
    for i, (color, name) in enumerate(zip(colors, names)):
        y1, y2 = i * bh, (i + 1) * bh
        x1, x2 = 0, bw
        draw.rectangle([(x1, y1), (x2, y2)], fill=tuple(color))
        # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 20)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
        name = name.replace('-merged', '').replace('-other', '').replace('-stuff', '')
        draw.text((x2 + 5, y1), name, font=font, fill=(0, 0, 0))
    del draw
    return im

def parse_sent(desc):
    """
    parse sentence into tokens and do cleaning
    :param desc: sentence
    :return: tokens
    """
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


CATEGORIES = [{'supercategory': 'textile', 'color': [255, 255, 128], 'isthing': 0, 'id': 0, 'name': 'banner'}, {'supercategory': 'textile', 'color': [147, 211, 203], 'isthing': 0, 'id': 1, 'name': 'blanket'}, {'supercategory': 'building', 'color': [150, 100, 100], 'isthing': 0, 'id': 2, 'name': 'bridge'}, {'supercategory': 'raw-material', 'color': [168, 171, 172], 'isthing': 0, 'id': 3, 'name': 'cardboard'}, {'supercategory': 'furniture-stuff', 'color': [146, 112, 198], 'isthing': 0, 'id': 4, 'name': 'counter'}, {'supercategory': 'textile', 'color': [210, 170, 100], 'isthing': 0, 'id': 5, 'name': 'curtain'}, {'supercategory': 'furniture-stuff', 'color': [92, 136, 89], 'isthing': 0, 'id': 6, 'name': 'door-stuff'}, {'supercategory': 'floor', 'color': [218, 88, 184], 'isthing': 0, 'id': 7, 'name': 'floor-wood'}, {'supercategory': 'plant', 'color': [241, 129, 0], 'isthing': 0, 'id': 8, 'name': 'flower'}, {'supercategory': 'food-stuff', 'color': [217, 17, 255], 'isthing': 0, 'id': 9, 'name': 'fruit'}, {'supercategory': 'ground', 'color': [124, 74, 181], 'isthing': 0, 'id': 10, 'name': 'gravel'}, {'supercategory': 'building', 'color': [70, 70, 70], 'isthing': 0, 'id': 11, 'name': 'house'}, {'supercategory': 'furniture-stuff', 'color': [255, 228, 255], 'isthing': 0, 'id': 12, 'name': 'light'}, {'supercategory': 'furniture-stuff', 'color': [154, 208, 0], 'isthing': 0, 'id': 13, 'name': 'mirror-stuff'}, {'supercategory': 'structural', 'color': [193, 0, 92], 'isthing': 0, 'id': 14, 'name': 'net'}, {'supercategory': 'textile', 'color': [76, 91, 113], 'isthing': 0, 'id': 15, 'name': 'pillow'}, {'supercategory': 'ground', 'color': [255, 180, 195], 'isthing': 0, 'id': 16, 'name': 'platform'}, {'supercategory': 'ground', 'color': [106, 154, 176], 'isthing': 0, 'id': 17, 'name': 'playingfield'}, {'supercategory': 'ground', 'color': [230, 150, 140], 'isthing': 0, 'id': 18, 'name': 'railroad'}, {'supercategory': 'water', 'color': [60, 143, 255], 'isthing': 0, 'id': 19, 'name': 'river'}, {'supercategory': 'ground', 'color': [128, 64, 128], 'isthing': 0, 'id': 20, 'name': 'road'}, {'supercategory': 'building', 'color': [92, 82, 55], 'isthing': 0, 'id': 21, 'name': 'roof'}, {'supercategory': 'ground', 'color': [254, 212, 124], 'isthing': 0, 'id': 22, 'name': 'sand'}, {'supercategory': 'water', 'color': [73, 77, 174], 'isthing': 0, 'id': 23, 'name': 'sea'}, {'supercategory': 'furniture-stuff', 'color': [255, 160, 98], 'isthing': 0, 'id': 24, 'name': 'shelf'}, {'supercategory': 'ground', 'color': [255, 255, 255], 'isthing': 0, 'id': 25, 'name': 'snow'}, {'supercategory': 'furniture-stuff', 'color': [104, 84, 109], 'isthing': 0, 'id': 26, 'name': 'stairs'}, {'supercategory': 'building', 'color': [169, 164, 131], 'isthing': 0, 'id': 27, 'name': 'tent'}, {'supercategory': 'textile', 'color': [225, 199, 255], 'isthing': 0, 'id': 28, 'name': 'towel'}, {'supercategory': 'wall', 'color': [137, 54, 74], 'isthing': 0, 'id': 29, 'name': 'wall-brick'}, {'supercategory': 'wall', 'color': [135, 158, 223], 'isthing': 0, 'id': 30, 'name': 'wall-stone'}, {'supercategory': 'wall', 'color': [7, 246, 231], 'isthing': 0, 'id': 31, 'name': 'wall-tile'}, {'supercategory': 'wall', 'color': [107, 255, 200], 'isthing': 0, 'id': 32, 'name': 'wall-wood'}, {'supercategory': 'water', 'color': [58, 41, 149], 'isthing': 0, 'id': 33, 'name': 'water-other'}, {'supercategory': 'window', 'color': [183, 121, 142], 'isthing': 0, 'id': 34, 'name': 'window-blind'}, {'supercategory': 'window', 'color': [255, 73, 97], 'isthing': 0, 'id': 35, 'name': 'window-other'}, {'supercategory': 'plant', 'color': [107, 142, 35], 'isthing': 0, 'id': 36, 'name': 'tree-merged'}, {'supercategory': 'structural', 'color': [190, 153, 153], 'isthing': 0, 'id': 37, 'name': 'fence-merged'}, {'supercategory': 'ceiling', 'color': [146, 139, 141], 'isthing': 0, 'id': 38, 'name': 'ceiling-merged'}, {'supercategory': 'sky', 'color': [70, 130, 180], 'isthing': 0, 'id': 39, 'name': 'sky-other-merged'}, {'supercategory': 'furniture-stuff', 'color': [134, 199, 156], 'isthing': 0, 'id': 40, 'name': 'cabinet-merged'}, {'supercategory': 'furniture-stuff', 'color': [209, 226, 140], 'isthing': 0, 'id': 41, 'name': 'table-merged'}, {'supercategory': 'floor', 'color': [96, 36, 108], 'isthing': 0, 'id': 42, 'name': 'floor-other-merged'}, {'supercategory': 'ground', 'color': [96, 96, 96], 'isthing': 0, 'id': 43, 'name': 'pavement-merged'}, {'supercategory': 'solid', 'color': [64, 170, 64], 'isthing': 0, 'id': 44, 'name': 'mountain-merged'}, {'supercategory': 'plant', 'color': [152, 251, 152], 'isthing': 0, 'id': 45, 'name': 'grass-merged'}, {'supercategory': 'ground', 'color': [208, 229, 228], 'isthing': 0, 'id': 46, 'name': 'dirt-merged'}, {'supercategory': 'raw-material', 'color': [206, 186, 171], 'isthing': 0, 'id': 47, 'name': 'paper-merged'}, {'supercategory': 'food-stuff', 'color': [152, 161, 64], 'isthing': 0, 'id': 48, 'name': 'food-other-merged'}, {'supercategory': 'building', 'color': [116, 112, 0], 'isthing': 0, 'id': 49, 'name': 'building-other-merged'}, {'supercategory': 'solid', 'color': [0, 114, 143], 'isthing': 0, 'id': 50, 'name': 'rock-merged'}, {'supercategory': 'wall', 'color': [102, 102, 156], 'isthing': 0, 'id': 51, 'name': 'wall-other-merged'}, {'supercategory': 'textile', 'color': [250, 141, 255], 'isthing': 0, 'id': 52, 'name': 'rug-merged'}, {'supercategory': 'person', 'color': [220, 20, 60], 'isthing': 1, 'id': 53, 'name': 'person'}, {'supercategory': 'vehicle', 'color': [119, 11, 32], 'isthing': 1, 'id': 54, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'color': [0, 0, 142], 'isthing': 1, 'id': 55, 'name': 'car'}, {'supercategory': 'vehicle', 'color': [0, 0, 230], 'isthing': 1, 'id': 56, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'color': [106, 0, 228], 'isthing': 1, 'id': 57, 'name': 'airplane'}, {'supercategory': 'vehicle', 'color': [0, 60, 100], 'isthing': 1, 'id': 58, 'name': 'bus'}, {'supercategory': 'vehicle', 'color': [0, 80, 100], 'isthing': 1, 'id': 59, 'name': 'train'}, {'supercategory': 'vehicle', 'color': [0, 0, 70], 'isthing': 1, 'id': 60, 'name': 'truck'}, {'supercategory': 'vehicle', 'color': [0, 0, 192], 'isthing': 1, 'id': 61, 'name': 'boat'}, {'supercategory': 'outdoor', 'color': [250, 170, 30], 'isthing': 1, 'id': 62, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'color': [100, 170, 30], 'isthing': 1, 'id': 63, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'color': [220, 220, 0], 'isthing': 1, 'id': 64, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'color': [175, 116, 175], 'isthing': 1, 'id': 65, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'color': [250, 0, 30], 'isthing': 1, 'id': 66, 'name': 'bench'}, {'supercategory': 'animal', 'color': [165, 42, 42], 'isthing': 1, 'id': 67, 'name': 'bird'}, {'supercategory': 'animal', 'color': [255, 77, 255], 'isthing': 1, 'id': 68, 'name': 'cat'}, {'supercategory': 'animal', 'color': [0, 226, 252], 'isthing': 1, 'id': 69, 'name': 'dog'}, {'supercategory': 'animal', 'color': [182, 182, 255], 'isthing': 1, 'id': 70, 'name': 'horse'}, {'supercategory': 'animal', 'color': [0, 82, 0], 'isthing': 1, 'id': 71, 'name': 'sheep'}, {'supercategory': 'animal', 'color': [120, 166, 157], 'isthing': 1, 'id': 72, 'name': 'cow'}, {'supercategory': 'animal', 'color': [110, 76, 0], 'isthing': 1, 'id': 73, 'name': 'elephant'}, {'supercategory': 'animal', 'color': [174, 57, 255], 'isthing': 1, 'id': 74, 'name': 'bear'}, {'supercategory': 'animal', 'color': [199, 100, 0], 'isthing': 1, 'id': 75, 'name': 'zebra'}, {'supercategory': 'animal', 'color': [72, 0, 118], 'isthing': 1, 'id': 76, 'name': 'giraffe'}, {'supercategory': 'accessory', 'color': [255, 179, 240], 'isthing': 1, 'id': 77, 'name': 'backpack'}, {'supercategory': 'accessory', 'color': [0, 125, 92], 'isthing': 1, 'id': 78, 'name': 'umbrella'}, {'supercategory': 'accessory', 'color': [209, 0, 151], 'isthing': 1, 'id': 79, 'name': 'handbag'}, {'supercategory': 'accessory', 'color': [188, 208, 182], 'isthing': 1, 'id': 80, 'name': 'tie'}, {'supercategory': 'accessory', 'color': [0, 220, 176], 'isthing': 1, 'id': 81, 'name': 'suitcase'}, {'supercategory': 'sports', 'color': [255, 99, 164], 'isthing': 1, 'id': 82, 'name': 'frisbee'}, {'supercategory': 'sports', 'color': [92, 0, 73], 'isthing': 1, 'id': 83, 'name': 'skis'}, {'supercategory': 'sports', 'color': [133, 129, 255], 'isthing': 1, 'id': 84, 'name': 'snowboard'}, {'supercategory': 'sports', 'color': [78, 180, 255], 'isthing': 1, 'id': 85, 'name': 'sports ball'}, {'supercategory': 'sports', 'color': [0, 228, 0], 'isthing': 1, 'id': 86, 'name': 'kite'}, {'supercategory': 'sports', 'color': [174, 255, 243], 'isthing': 1, 'id': 87, 'name': 'baseball bat'}, {'supercategory': 'sports', 'color': [45, 89, 255], 'isthing': 1, 'id': 88, 'name': 'baseball glove'}, {'supercategory': 'sports', 'color': [134, 134, 103], 'isthing': 1, 'id': 89, 'name': 'skateboard'}, {'supercategory': 'sports', 'color': [145, 148, 174], 'isthing': 1, 'id': 90, 'name': 'surfboard'}, {'supercategory': 'sports', 'color': [255, 208, 186], 'isthing': 1, 'id': 91, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'color': [197, 226, 255], 'isthing': 1, 'id': 92, 'name': 'bottle'}, {'supercategory': 'kitchen', 'color': [171, 134, 1], 'isthing': 1, 'id': 93, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'color': [109, 63, 54], 'isthing': 1, 'id': 94, 'name': 'cup'}, {'supercategory': 'kitchen', 'color': [207, 138, 255], 'isthing': 1, 'id': 95, 'name': 'fork'}, {'supercategory': 'kitchen', 'color': [151, 0, 95], 'isthing': 1, 'id': 96, 'name': 'knife'}, {'supercategory': 'kitchen', 'color': [9, 80, 61], 'isthing': 1, 'id': 97, 'name': 'spoon'}, {'supercategory': 'kitchen', 'color': [84, 105, 51], 'isthing': 1, 'id': 98, 'name': 'bowl'}, {'supercategory': 'food', 'color': [74, 65, 105], 'isthing': 1, 'id': 99, 'name': 'banana'}, {'supercategory': 'food', 'color': [166, 196, 102], 'isthing': 1, 'id': 100, 'name': 'apple'}, {'supercategory': 'food', 'color': [208, 195, 210], 'isthing': 1, 'id': 101, 'name': 'sandwich'}, {'supercategory': 'food', 'color': [255, 109, 65], 'isthing': 1, 'id': 102, 'name': 'orange'}, {'supercategory': 'food', 'color': [0, 143, 149], 'isthing': 1, 'id': 103, 'name': 'broccoli'}, {'supercategory': 'food', 'color': [179, 0, 194], 'isthing': 1, 'id': 104, 'name': 'carrot'}, {'supercategory': 'food', 'color': [209, 99, 106], 'isthing': 1, 'id': 105, 'name': 'hot dog'}, {'supercategory': 'food', 'color': [5, 121, 0], 'isthing': 1, 'id': 106, 'name': 'pizza'}, {'supercategory': 'food', 'color': [227, 255, 205], 'isthing': 1, 'id': 107, 'name': 'donut'}, {'supercategory': 'food', 'color': [147, 186, 208], 'isthing': 1, 'id': 108, 'name': 'cake'}, {'supercategory': 'furniture', 'color': [153, 69, 1], 'isthing': 1, 'id': 109, 'name': 'chair'}, {'supercategory': 'furniture', 'color': [3, 95, 161], 'isthing': 1, 'id': 110, 'name': 'couch'}, {'supercategory': 'furniture', 'color': [163, 255, 0], 'isthing': 1, 'id': 111, 'name': 'potted plant'}, {'supercategory': 'furniture', 'color': [119, 0, 170], 'isthing': 1, 'id': 112, 'name': 'bed'}, {'supercategory': 'furniture', 'color': [0, 182, 199], 'isthing': 1, 'id': 113, 'name': 'dining table'}, {'supercategory': 'furniture', 'color': [0, 165, 120], 'isthing': 1, 'id': 114, 'name': 'toilet'}, {'supercategory': 'electronic', 'color': [183, 130, 88], 'isthing': 1, 'id': 115, 'name': 'tv'}, {'supercategory': 'electronic', 'color': [95, 32, 0], 'isthing': 1, 'id': 116, 'name': 'laptop'}, {'supercategory': 'electronic', 'color': [130, 114, 135], 'isthing': 1, 'id': 117, 'name': 'mouse'}, {'supercategory': 'electronic', 'color': [110, 129, 133], 'isthing': 1, 'id': 118, 'name': 'remote'}, {'supercategory': 'electronic', 'color': [166, 74, 118], 'isthing': 1, 'id': 119, 'name': 'keyboard'}, {'supercategory': 'electronic', 'color': [219, 142, 185], 'isthing': 1, 'id': 120, 'name': 'cell phone'}, {'supercategory': 'appliance', 'color': [79, 210, 114], 'isthing': 1, 'id': 121, 'name': 'microwave'}, {'supercategory': 'appliance', 'color': [178, 90, 62], 'isthing': 1, 'id': 122, 'name': 'oven'}, {'supercategory': 'appliance', 'color': [65, 70, 15], 'isthing': 1, 'id': 123, 'name': 'toaster'}, {'supercategory': 'appliance', 'color': [127, 167, 115], 'isthing': 1, 'id': 124, 'name': 'sink'}, {'supercategory': 'appliance', 'color': [59, 105, 106], 'isthing': 1, 'id': 125, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'color': [142, 108, 45], 'isthing': 1, 'id': 126, 'name': 'book'}, {'supercategory': 'indoor', 'color': [196, 172, 0], 'isthing': 1, 'id': 127, 'name': 'clock'}, {'supercategory': 'indoor', 'color': [95, 54, 80], 'isthing': 1, 'id': 128, 'name': 'vase'}, {'supercategory': 'indoor', 'color': [128, 76, 255], 'isthing': 1, 'id': 129, 'name': 'scissors'}, {'supercategory': 'indoor', 'color': [201, 57, 1], 'isthing': 1, 'id': 130, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'color': [246, 0, 122], 'isthing': 1, 'id': 131, 'name': 'hair drier'}, {'supercategory': 'indoor', 'color': [191, 162, 208], 'isthing': 1, 'id': 132, 'name': 'toothbrush'}]

#########################################################
# DDPG
#########################################################

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_extra_plane(x):
    """
    get the luminance of input x
    :param x: (bs, 3, h, w)
    :return: extra_plane (bs, 3, h, w)
    """

    lum = (x[:, 0] * 0.27 + x[:, 1] * 0.67 + x[:, 2] * 0.06 + 1e-5)

    luminance = torch.mean(lum, dim=(1, 2))
    contrast = torch.var(lum, dim=(1, 2), unbiased=False)
    i_max = torch.max(x, dim=1)[0]
    i_min = torch.min(x, dim=1)[0]
    sat = (i_max - i_min) / (torch.min(i_max + i_min, 2.0 - i_max - i_min) + 1e-2)
    saturation = torch.mean(sat, dim=(1, 2))
    extra_plane = torch.stack([luminance, contrast, saturation], 1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    return extra_plane

def pad_progress(img, progress):
    """
    pad progress channel
    :param img: (bs, c, h, w)
    :param progress: [float] step/max_step | (bs)
    :return: img: (bs, c+1, h, w)
    """
    bs, _, h, w = img.shape
    if type(progress) == float:
        p_plane = torch.ones(bs, 1, h, w).to(img.device) * progress
    else:
        p_plane = torch.ones(bs, 1, h, w).to(img.device) * progress.view(-1, 1, 1, 1)
    img = torch.cat((img, p_plane), 1)
    return img

def set_seed(seed):
    """set random seed"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def combine_feats(feats1, feats2):
    feats = []
    for feat1, feat2 in zip(feats1, feats2):
        feats.append(torch.cat([feat1, feat2]))
    return feats

def to_var(x):
    x = x.cuda()
    return Variable(x)

def to_tensor(x):
    x = torch.tensor(x).cuda()
    return x

def to_tensor_cpu(x):
    x = torch.tensor(x)
    return x

# adapt for gan
def get_single_data_from_batch(data, idx):
    result = []
    for item in data:
        result.append(item[idx])
    return result

def concat_single_data_to_batch(data1, data2):
    data = []
    for data1, data2 in zip(data1, data2):
        data.append(torch.cat([data1, data2]))
    return data

def expend_dim_for_data(data, dim=0):
    expended_data = []
    for item in data:
        expended_data.append(torch.unsqueeze(item, dim))
    return expended_data

def collate_fn_inference(data):
    data_res = None
    for data_i in data:
        data_res = data_i if data_res is None else concat_single_data_to_batch(data_res, data_i)
    return data_res

