import pdb
import torch
from torch import nn
from torchvision.models.vgg import vgg16


class PerceptualLoss(nn.Module):
    """
    only use the last layer of vgg19 to compute perceptual loss
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

    def forward(self, src, tgt):
        """
        :param src: (bs, n, c, h, w) in [0, 1] RGB
        :param tgt: (bs, 1, c, h, w) in [0, 1]
        :return: dist: (bs, n)
        """
        mean = self.mean.to(src.device).view(1, 1, 3, 1, 1)
        std = self.std.to(src.device).view(1, 1, 3, 1, 1)
        src = (src - mean) / std
        tgt = (tgt - mean) / std
        _, n, c, h, w = src.shape
        src_feat = self.loss_network(src.view(-1, c, h, w))
        _, cc, hh, ww = src_feat.shape
        src_feat = src_feat.view(-1, n, cc, hh, ww)
        tgt_feat = self.loss_network(tgt.view(-1, c, h, w))
        tgt_feat = tgt_feat.view(-1, 1, cc, hh, ww)
        if type(src) == list:
            dist_list = [torch.pow(torch.pow(src_feat[i] - tgt_feat, 2).mean((2, 3, 4)), 0.5) for i in range(len(src))]
            dist = torch.stack(dist_list).mean(dim=0)
        else:
            dist = torch.pow(torch.pow(src_feat - tgt_feat, 2).mean((2, 3, 4)), 0.5)
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return dist
