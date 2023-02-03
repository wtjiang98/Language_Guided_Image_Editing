from models.networks.sync_batchnorm import DataParallelWithCallback
from models.all_model import ALLModel
from models.max_margin_crit import MaxMarginCriterion
from util.utils import get_single_data_from_batch, expend_dim_for_data, concat_single_data_to_batch
import torch
import pdb

class AllTrainer():

    def __init__(self, opt, dataset, dataloaders=None):
        self.opt = opt
        self.dataset = dataset
        self.all_model = ALLModel(opt, dataset)
        self.mm_crit = MaxMarginCriterion(opt.visual_rank_weight, opt.lang_rank_weight, opt.margin)
        # if len(opt.gpu_ids) > 0:
        #     self.all_model = DataParallelWithCallback(self.all_model, device_ids=opt.gpu_ids)
        #     self.all_model_on_one_gpu = self.all_model.module
        # else:
        #     self.all_model_on_one_gpu = self.all_model

        if len(opt.gpu_ids) > 0:
            self.all_model.cuda()
            self.mm_crit.cuda()

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.all_model.create_optimizers(opt)
            self.old_lr = opt.lr
            self.optimizer_ground = torch.optim.Adam(self.all_model.JointMatching.parameters(), lr=opt.learning_rate, betas=(opt.optim_alpha, opt.optim_beta), eps=opt.optim_epsilon)

        self.mm_loss = 0

    def gen_batch_data(self, data):
        max_length = self.dataset.max_length
        batch_size = self.opt.batch_size
        # max_length = self.dataloader_local.dataset.max_length
        # batch_size = self.dataloader_local.batch_size
        datas = [data[i * int(len(data) / max_length): (i + 1) * int(len(data) / max_length)] for i in range(max_length)]
        processed_data = None
        for b in range(batch_size):
            for m in range(max_length):
                cur_data = get_single_data_from_batch(datas[m], b)
                if int(cur_data[-1]) == 1:
                    cur_data = expend_dim_for_data(cur_data)
                    processed_data = cur_data[: -1] if processed_data is None else concat_single_data_to_batch(processed_data, cur_data[: -1])
        return processed_data


    def run_classify(self, img_mix, req_mix, op_mix):
        # self.all_model.train()
        self.all_model.eval()
        prob_mix, attn_mix = self.all_model.classify_forward(img_mix, req_mix, op_mix)
        return prob_mix, attn_mix

    def run_ground(self, pfeat, lfeat, dif_lfeat, cxt_pfeat, cxt_lfeats, req, op_attn_l):
        scores, sub_attn, loc_attn, rel_attn, _, _, = self.all_model.ground_forward(pfeat, lfeat, dif_lfeat, cxt_pfeat, cxt_lfeats, req, op_attn_l)
        return scores, sub_attn, loc_attn, rel_attn

    def run_classify_inference(self, img, req, op):
        self.all_model.eval()
        prob, op_attn = self.all_model.classify_forward(img, req, op)
        return prob, op_attn

    def run_ground_inference(self, pfeats, lfeats, dif_lfeats, cxt_pfeats, cxt_lfeats, req, op_attn):
        scores, sub_attn, loc_attn, rel_attn, max_rel_ixs, weights = self.all_model.ground_forward(pfeats, lfeats, dif_lfeats, cxt_pfeats, cxt_lfeats, req, op_attn)
        self.all_model.train()
        return scores, sub_attn, loc_attn, rel_attn, max_rel_ixs, weights

    def compute_mm_loss(self, scores, opt):
        self.mm_loss = self.mm_crit(scores)

    def compute_bce_loss(self, prob_l, prob_g):
        bce_crit = torch.nn.BCELoss()
        bce_loss_l = bce_crit(prob_l, torch.ones_like(prob_l)) if prob_l is not None else 0
        bce_loss_g = bce_crit(prob_g, torch.zeros_like(prob_g)) if prob_g is not None else 0
        self.bce_loss = bce_loss_l + bce_loss_g

    def optimizer_zero_grad(self):
        self.optimizer_ground.zero_grad()

    def loss_backward(self):
        ground_loss = self.mm_loss + self.bce_loss
        ground_loss.backward(retain_graph=True)
        # ground_loss.backward()
        self.optimizer_ground.step()
        self.ground_loss = ground_loss

    def load_ground_checkpoint(self, checkpoint_path):
        self.all_model.JointMatching.load_checkpoint(checkpoint_path)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.all_model.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.all_model.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        if self.opt.lambda_gan > 0 or self.opt.lambda_gan_uncond > 0:
            return {**self.g_losses, **self.d_losses}
        else:
            return {**self.g_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch, opt):
        self.all_model.pix2pix_model.save(epoch, opt)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
