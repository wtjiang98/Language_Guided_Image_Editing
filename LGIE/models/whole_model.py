import torch
import torch.nn as nn
from models.grounding import JointMatching
from models.pix2pix_model import Pix2PixModel

class WholeModel(nn.Module):
    def __init__(self, opt_ground, opt, dataset):
        self.JointMatching = JointMatching(opt_ground)
        self.Pix2PixModel = Pix2PixModel(opt, dataset)

    def classify_forward(self, img, labels, prior):
        return self.JointMatching.classify_forward(self, img, labels, prior)

    def calssify_ground_forward(self, pfeat, lfeats, dif_lfeats, cxt_pfeat, cxt_lfeats, labels, attn):
        return self.JointMatching.forward(pfeat, lfeats, dif_lfeats, cxt_pfeat, cxt_lfeats, labels, attn)

    def edit_forward(self, data, mode):
        return self.Pix2PixModel(data, mode)
