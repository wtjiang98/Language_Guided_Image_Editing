import torch
import torch.nn as nn


def l2_norm(feat, dim=1):
  """
  L2 norm for features in certain dimension
  """
  # feat: [N, C, H, W]
  eps = 1e-12
  feat_norm = torch.norm(feat, dim=dim, keepdim=True) + eps
  # feat_norm: [N, 1, H, W] if dim == 1
  feat_l2_normed = feat / feat_norm
  return feat_l2_normed


class MutanHead(nn.Module):
    """
    Simplified Mutan head
    """
    def __init__(self, vis_ori_size, vis_reduced_size=256, lang_size=512, hid_size=256, reduce_sum=False, activation=nn.Tanh()):
        super(MutanHead, self).__init__()
        self.reduce_sum = reduce_sum
        self.vis_trans = nn.Sequential(
            nn.Conv2d(vis_ori_size, vis_reduced_size, 1),
            activation
        )
        self.lang_trans = nn.Sequential(
            nn.Conv2d(lang_size, hid_size, 1),
            activation
        )

    def forward(self, vis_feat, lang_feat):
        # Visual feature transform
        vis_sp_feat = vis_feat
        # vis_sp_feat: [B, vis_reduced_size + spatial_size, H, W]
        vis_trans_feat = self.vis_trans(vis_sp_feat)
        # vis_trans_feat: [B, vis_reduced_size, H, W]

        # # Language feature transform
        # lang_feat = lang_feat.unsqueeze(-1)
        # # lang_feat: [B, hid_size, 1]
        # lang_trans_feat = self.lang_trans(lang_feat)
        # # lang_trans_feat: [B, hid_size, 1]
        # lang_trans_feat = lang_trans_feat.unsqueeze(3)
        # # lang_trans_feat: [B, hid_size, 1, 1]

        lang_trans_feat = self.lang_trans(lang_feat)

        mutan_head_feat = vis_trans_feat * lang_trans_feat
        # mutan_head_feat: [B, vis_reduced_size, H, W]
        if self.reduce_sum:
            mutan_head_feat = mutan_head_feat.sum(1).unsqueeze(1)

        return mutan_head_feat


class MutanFusion(nn.Module):
    """
    Simplified Mutan fusion
    """
    def __init__(self, vis_ori_size, vis_reduced_size, lang_size, hid_size):
        super(MutanFusion, self).__init__()

        self.num_heads = 5
        self.mutan_heads = nn.ModuleList()

        for i in range(self.num_heads):
            self.mutan_heads.append(MutanHead(vis_ori_size, vis_reduced_size, lang_size, hid_size))

        self.tanh = nn.Tanh()

    def forward(self, vis_feat, lang_feat):
        mutan_heads_feats = []
        for i in range(self.num_heads):
            mutan_heads_feats.append(self.mutan_heads[i](vis_feat, lang_feat))

        mutan_feat = mutan_heads_feats[0]
        for i in range(1, self.num_heads):
            mutan_feat += mutan_heads_feats[i]
        mutan_feat = self.tanh(mutan_feat)
        # mutan_feat: [B, vis_reduced_size, H, W]
        mutan_feat = l2_norm(mutan_feat, dim=1)

        return mutan_feat