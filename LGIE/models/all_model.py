import torch
import torch.nn as nn
from models.grounding import JointMatching


class ALLModel(nn.Module):
    def __init__(self, opt, dataset):
        super(ALLModel, self).__init__()
        self.JointMatching = JointMatching(opt)
        # if opt.netD == 'InpaintSA':
        #     from models.pix2pix_model_SNInpaint import Pix2PixModel
        #     self.pix2pix_model = Pix2PixModel(opt, dataset)
        # else:
        from models.pix2pix_model import Pix2PixModel
        self.pix2pix_model = Pix2PixModel(opt, dataset)

    def classify_forward(self, img, labels, prior):
        return self.JointMatching.classify_forward(img, labels, prior)

    def ground_forward(self, pfeat, lfeats, dif_lfeats, cxt_pfeat, cxt_lfeats, labels, attn):
        return self.JointMatching.forward(pfeat, lfeats, dif_lfeats, cxt_pfeat, cxt_lfeats, labels, attn)

    def edit_forward(self, data, mode):
        return self.pix2pix_model(data, mode)

    def create_optimizers(self, opt):
        return self.pix2pix_model.create_optimizers(opt)
