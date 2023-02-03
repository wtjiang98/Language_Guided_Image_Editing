import sys
sys.path.append('../external/cocoapi/PythonAPI')
import json
import re
import string
from pycocotools import mask as cocomask
import numpy as np
import os
from PIL import Image
import pickle
import cv2

################################################# create vocabulary.txt
vocabulary = []

with open('../data/IER2_for_zip/IER2.json', 'r') as f:
    pairs = json.loads(f.read())

for n, pair in enumerate(pairs):
    for ins in pair['expert_summary']:
        # vocab = re.findall(r'\w+', ins)
        vocab = re.sub('[' + string.punctuation + ']', '', ins).split()
        vocabulary.extend(vocab)
    print(n, 'done')

vocabulary = list(set(vocabulary))

with open('../data/vocabulary_gier.txt', 'w') as f:
    f.write('<pad>\n<go>\n<eos>\n<unk>\n')
    for vocab in vocabulary:
        f.write(vocab+'\n')

print('vocabulary_gier.txt done')

################################################# create instance.json
vis_mask = False
instance_result = {
    'info': {'description': 'gier'},
    'categories': [
        {'supercategory': 'none', 'id': 1, 'name': 'foreground'}
    ]
}

refs_result = []

not_exist = ['apb5mr_apb5mr.jpg']
train_size = 5500
# train_size = 15

global_sent_id = 0
global_mask_id = 0
instance_result['annotations'] = []
instance_result['images'] = []

IER2 = json.load(open('../data/IER2_for_zip/IER2.json', 'r'))
for n, img_pair in enumerate(IER2):
    ref = {}
    img_info = {}
    img_name = img_pair['input'].replace('/', '_')
    print(n, 'dealing ' + img_name)
    if img_name in not_exist: continue
    img_info['file_name'] = img_name
    img = Image.open(os.path.join('../data/IER2_for_zip/images', img_name))
    img_info['height'] = img.height
    img_info['width'] = img.width
    img_info['id'] = n
    instance_result['images'].append(img_info)

    ref['image_id'] = img_info['id']
    h = img_info['height']
    w = img_info['width']
    if n < train_size:
        ref['split'] = 'train'
    else:
        ref['split'] = 'test'
    ref['sentences'] = []
    ref['sent_ids'] = []
    for sent in img_pair['expert_summary']:
        sent_info = {}
        sent_info['raw'] = sent
        sent_info['sent'] = sent
        sent_info['sent_id'] = global_sent_id
        ref['sent_ids'].append(global_sent_id)
        global_sent_id = global_sent_id + 1
        sent_info['tokens'] = re.sub('[' + string.punctuation + ']', '', sent).split()
        ref['sentences'].append(sent_info)

    ref['file_name'] = img_name
    ref['gt_im_name'] = img_pair['output'].replace('/', '_')
    ref['category_id'] = 1
    ref['ref_id'] = n

    annotations_mask = {}
    rles = json.load(open('../data/IER2_for_zip/masks/'+img_name.split('.')[0]+'_mask.json', 'r'))
    # with open('../data/IER2_for_zip/masks/'+img_name.split('.')[0]+'_mask.json', 'r') as f:
    #     rles = json.loads(f.read())
    mask_list = []
    whole_img = False
    for op_name in img_pair['operator'].keys():
        if not img_pair['operator'][op_name]['local']:
            bimask = np.ones((h, w), dtype=np.uint8)
            mask_rle = cocomask.encode(np.asfortranarray(bimask))
            whole_img = True
            break
        else:
            for mask_id in img_pair['operator'][op_name]['ids']:
                mask_rle = rles[mask_id]
                # mask_rle['counts'] = mask_rle['counts'].encode(encoding="utf-8")
                bimask = cocomask.decode(rles[mask_id])
                # if img_pair['operator'][op_name]['mask_mode'] == 'exclusive':
                #     bimask = (bimask - 1) * (-1)  # inverse
                boolmask = bimask > 0
                mask_list.append(boolmask)
    if not whole_img:
        boolmask = np.zeros((h, w), dtype=np.uint8) > 0
        for cur_boolmask in mask_list:
            boolmask = boolmask + cur_boolmask
        bimask = boolmask * 1
        mask_rle = cocomask.encode(np.asfortranarray(bimask.astype(np.uint8)))

    if vis_mask:
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

    mask_rle['counts'] = mask_rle['counts'].decode()  # bytes to str, for saving json file
    annotations_mask['segmentation'] = mask_rle
    # test = annotations_mask['segmentation'].encode(encoding="utf-8")  # str to bytes
    annotations_mask['image_id'] = ref['image_id']
    annotations_mask['category_id'] = 1
    annotations_mask['id'] = global_mask_id
    ref['ann_id'] = global_mask_id
    global_mask_id = global_mask_id + 1

    instance_result['annotations'].append(annotations_mask)
    refs_result.append(ref)
    print(n, 'done')

with open('../external/refer/data/gier/instances.json', 'w') as f:
    json.dump(instance_result, f)

with open('../external/refer/data/gier/refs(gc).p', 'wb') as f:
    pickle.dump(refs_result, f)


print('instance.json and ref.p done')