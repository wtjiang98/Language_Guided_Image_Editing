import torch.nn.functional as F
import json
import os
from PIL import Image
import cv2
import torchvision.transforms as transforms

source_dir = '/mnt/data1/jwt/VisualRelationships-master/dataset'
datasets = ['gier_test']

transforms_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

l1 = 0

for dataset in datasets:
    with open(os.path.join(source_dir, dataset, 'test.json'), 'r') as f:
        dataset_j = json.load(f)
    for item in dataset_j:
        img0_dir = item['img0'].replace('input_image', 'output_image')
        # img0_dir = item['img0']
        img1_dir = item['img1'].replace('output_image', 'input_image')
        img0 = cv2.imread(img0_dir)
        img1 = cv2.imread(img1_dir)
        img0 = transforms_img(img0)
        img1 = transforms_img(img1)
        l1 += F.l1_loss(img0, img1)
    print(item['uid'], l1)





