import numpy as np
import pdb
import time

import torch

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score, recall_score
from tabulate import tabulate


from models.loss import PerceptualLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Evaluator(object):
    def __init__(self, vocab):
        self.paramed_ops = ['brightness', 'contrast', 'blur', 'sharpness', 'color']
        self.vocab = vocab['operator_idx_to_token']
        self.reset()

    def reset(self):
        self.refs = []
        self.cands = []
        self.op_dict = {op: {'pred': [], 'true': []} for op in self.paramed_ops}

    def tensor2list(self, x):
        return x.cpu().numpy().tolist()

    def normalize(self, arr):
        """
        remove the redundant token in operator tensor
        :param arr: operator list (n,)
        :return: out: operator list (n,) or (n - 2,)
        """
        arr = np.array(arr)
        s = np.where(arr == 1)[0]
        if len(s) > 0:
            s = s[0]
        else:
            s = -1
        e = np.where(arr == 2)[0]
        if len(e) > 0:
            e = e[0]
        else:
            e = len(arr)
        assert e >= s, 'y is not correct to decode'
        out = arr[s + 1:e]
        return out.tolist()

    def update(self, x, y, px=None, py=None):
        """
        update the operator and parameter
        :param x: operators pred tensor (bs, n)
        :param y: operators gt tensor (bs, m): SPECIAL: start with 1
        :param px: parameter pred tensor (bs, n)
        :param py: parameter gt tensor (bs, m)
        """
        bs, _ = x.shape
        x = self.tensor2list(x)
        y = self.tensor2list(y)
        px = self.tensor2list(px)
        py = self.tensor2list(py) if py is not None else py
        for i in range(bs):
            x_i = self.normalize(x[i])
            y_i = self.normalize(y[i])
            self.cands.append(x_i)
            self.refs.append([y_i])
            if py is not None:
                for j in range(min(len(x_i), len(y_i))):
                    if x_i[j] == y_i[j]:
                        # if the operator is correct
                        op_idx = x_i[j]
                        op_name = self.vocab[op_idx]
                        if op_name in self.paramed_ops:
                            self.op_dict[op_name]['pred'].append(px[i][j])
                            self.op_dict[op_name]['true'].append(py[i][j])


    def eval_mse(self):
        """
        evaluate the mean square error
        :return:
        """
        mse = {}
        for (op, v) in self.op_dict.items():
            mse[op] = 0 if v['true'] == [] or v['pred'] == [] else mean_squared_error(v['true'], v['pred'])
        return mse

    def eval_bleu(self):
        """
        reference
        :param ref: references: list of list of list of token
        :param cand: candidates: list of list of token
        :return: bleu1, bleu2
        """

        bleu1 = 0 if self.refs == [] or self.cands == [] else corpus_bleu(self.refs, self.cands, weights=(1, 0, 0, 0))
        bleu2 = 0 if self.refs == [] or self.cands == [] else corpus_bleu(self.refs, self.cands, weights=(0, 1, 0, 0))
        return bleu1, bleu2


# for grounding evaluation
# IoU
def mask_iou(m1, m2):
    """
    compute iou between two masks
    :param m1: (h, w) \in {0,1}
    :param m2: (h, w) \in {0,1}
    :return: iou
    """
    inter = (m1 * m2).sum().item()
    union = (m1 + m2).clamp(0, 1).sum().item()
    iou = float(inter)/(union + 1e-6)
    return iou

def plot(x1, x2, x3, x4):
    t = np.arange(len(x1))
    order = np.argsort(x1)
    x1 = x1[order]
    x2 = x2[order]
    x3 = x3[order]
    x4 = x4[order]
    plt.plot(t, x1, '-b', label='in_dist')
    plt.plot(t, x2, '-r', label='out_dist')
    plt.plot(t, x3, '-g', label='decre_dist')
    plt.title('distance figure')
    plt.xlabel('index of image')
    plt.ylabel('distance')
    plt.legend()
    plt.savefig('distance.jpg')
    plt.close()


class ImageEvaluator(object):
    def __init__(self):
        self.perceptual_net = PerceptualLoss()
        self.perceptual_net.to(device)
        self.reset()

    def reset(self):
        self.out_dist = [] # L1 distance of output
        self.in_dist = [] # L1 distance of input

        self.perc_out_dist = []
        self.perc_in_dist = []

    def update(self, input, output, gt):
        """
        torch tensor
        :param input: (1, 3, h, w)
        :param output: (1, 3, h, w)
        :param gt: (1, 3, h, w)
        :return:
        """
        in_dist = torch.abs(input - gt).mean().item()
        out_dist = torch.abs(output - gt).mean().item()
        self.in_dist.append(in_dist)
        self.out_dist.append(out_dist)

        # calculate perceptual distance
        input = input.unsqueeze(0).to(device)
        output = output.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)

        perc_in_dist = self.perceptual_net(input, gt).item()
        perc_out_dist = self.perceptual_net(output, gt).item()
        self.perc_in_dist.append(perc_in_dist)
        self.perc_out_dist.append(perc_out_dist)


    def eval_perceptual(self):
        in_dists = np.array(self.perc_in_dist)
        out_dists = np.array(self.perc_out_dist)
        decre_dists = (in_dists - out_dists)
        mean_out_dist = np.mean(out_dists)
        mean_in_dist = np.mean(in_dists)
        mean_incre_dist = np.mean(decre_dists)
        return mean_in_dist, mean_out_dist, mean_incre_dist


    def eval_L1(self):
        in_dists = np.array(self.in_dist)
        out_dists = np.array(self.out_dist)
        decre_dists = (in_dists - out_dists)
        mean_out_dist = np.mean(out_dists)
        mean_in_dist = np.mean(in_dists)
        mean_incre_dist = np.mean(decre_dists)
        # plot(in_dists, out_dists, in_dists - out_dists, decre_dists)
        return mean_in_dist, mean_out_dist, mean_incre_dist

    def print_eval(self):
        in_L1, out_L1, L1_decre = self.eval_L1()
        in_perc, out_perc, perc_decre = self.eval_perceptual()
        print('input L1 dist {:.4f}, output L1 dist {:.4f}, L1 decre: {:.4f}'.format(in_L1, out_L1, L1_decre))
        print('input perc dist {:.4f}, output perc dist {:.4f}, perc decre: {:.4f}'.format(in_perc, out_perc, perc_decre))


class Ground_Evaluator(object):

    def __init__(self, ):
        self.reset()

    def reset(self):
        self.scores = [] # store all scores
        self.gts = [] # store all gts
        self.threshes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.ious = [[] for _ in range(len(self.threshes))]
        self.total_nums = [0 for _ in range(len(self.threshes))]
        self.correct_nums = [0 for _ in range(len(self.threshes))]

    def update(self, scores, gts, masks):
        """
        :param scores: (n,)
        :param gts: (n,) \in {0, 1}
        :param masks: list of n (h, w) \in {0, 1}
        :return:
        """
        self.scores.extend(scores)
        self.gts.extend(gts)
        scores = np.array(scores)
        gts = np.array(gts)
        if masks is not None:
            masks = np.array(masks)
            masks = torch.from_numpy(masks).cuda()
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            self.total_nums[i] += preds.sum()
            self.correct_nums[i] += (preds * gts).sum()
            if masks is not None:
                pred_mask = masks[np.where(preds > 0)[0]].sum(0).clamp(0, 1)
                gt_mask = masks[np.where(gts > 0)[0]].sum(0).clamp(0, 1)
                iou = mask_iou(pred_mask, gt_mask)
                self.ious[i].append(iou)

    def eval_acc(self):
        """
        eval accuracy in different threshold
        :return: acc: list of accuracy
        """
        # accuracy
        acc = [correct_num / (total_num + 1e-6) for (correct_num, total_num) in zip(self.correct_nums, self.total_nums)]
        return acc

    def eval_iou(self):
        """
        evaluate image mask iou in different threshold
        :return: iou: list of iou
        """
        if len(self.ious) == 0:
            return []
        iou = [float(np.mean(ious)) for ious in self.ious]
        pdb.set_trace()
        return iou

    def eval_roc(self):
        """
        regard it as multi-label classification, which have no effect about threshold
        :return: roc: float
        """
        roc = float(roc_auc_score(self.gts, self.scores))
        return roc

    def print_eval(self):
        acc = self.eval_acc()
        iou = self.eval_iou()
        roc = self.eval_roc()
        print('roc auc: {:4f}'.format(roc))
        acc_row = ['accuracy'] + acc
        iou_row = ['IoU'] + iou
        header = ['thresh'] + list(self.threshes)
        print(tabulate([acc_row, iou_row], headers=header, tablefmt='orgtbl'))


class GroundOperatorEvaluator(object):
    def __init__(self, ):
        self.reset()

    def reset(self):
        self.scores = [] # store all scores
        self.local_probs = [] # store all probs for local operation
        self.locals = [] # store gt for local operation
        self.gts = [] # store all gts choice of candidate
        self.threshes = (100*np.linspace(0, 1, 21)).astype(int) / 100  # only keep two decimal
        self.ious = [[] for _ in range(len(self.threshes))]
        self.total_nums = [0 for _ in range(len(self.threshes))]
        self.correct_nums = [0 for _ in range(len(self.threshes))]

    def update(self, prob, scores, gts, masks, local):
        """
        :param: prob: (1, 1) for one sample whether is local or global
        :param scores: (n,) for each candidate
        :param gts: (n,) \in {0, 1}
        :param masks: list of n (h, w) \in {0, 1}
        :return:
        """
        self.locals.append(local)
        self.local_probs.append(prob)
        if local and len(gts) > 0:  # local
            self.scores.extend(scores)
            self.gts.extend(gts)
            scores = np.array(scores)
            gts = np.array(gts)
            if masks is not None:
                masks = np.array(masks)
                masks = torch.from_numpy(masks).cuda()
            for i, thresh in enumerate(self.threshes):
                preds = (scores > thresh).astype(int)
                self.total_nums[i] += preds.sum()
                self.correct_nums[i] += (preds * gts).sum()
                if masks is not None:
                    pred_mask = masks[np.where(preds > 0)[0]].sum(0).clamp(0, 1)
                    gt_mask = masks[np.where(gts > 0)[0]].sum(0).clamp(0, 1)
                    iou = mask_iou(pred_mask, gt_mask)
                    self.ious[i].append(iou)

    def eval_ground_acc(self):
        """
        eval accuracy in different threshold
        :return: acc: list of accuracy
        """
        # accuracy
        acc = [correct_num / (total_num + 1e-6) for (correct_num, total_num) in zip(self.correct_nums, self.total_nums)]
        return acc

    def eval_local_acc(self):
        """
        evaluate the accuracy for local or global operation binary classification
        :return: acc: one of accuracy
        """
        # accuracy
        acc = ((np.array(self.local_probs) > 0.5) == np.array(self.locals)).sum() / len(self.locals)
        return acc


    def eval_iou(self):
        """
        evaluate image mask iou in different threshold
        :return: iou: list of iou
        """
        if len(self.ious) == 0:
            return []
        iou = [float(np.mean(ious)) for ious in self.ious]
        return iou


    def eval_ground_roc(self):
        """
        regard it as multi-label classification, which have no effect about threshold
        :return: roc: float
        """
        roc = float(roc_auc_score(self.gts, self.scores))
        return roc

    def eval_local_roc(self):
        """
        regard it as multi-label classification, which have no effect about threshold
        :return: roc: float
        """
        roc = float(roc_auc_score(self.locals, self.local_probs))
        return roc

    def eval_local_all(self):
        scores = np.array(self.local_probs)
        gts = np.array(self.locals)
        acc, recal, f1 = [], [], []
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            acc.append(accuracy_score(gts, preds))
            recal.append(recall_score(gts, preds))
            f1.append(f1_score(gts, preds))
        return acc, recal, f1

    def eval_ground_all(self):
        scores = np.array(self.scores)
        gts = np.array(self.gts)
        acc, recal, f1 = [], [], []
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            acc.append(accuracy_score(gts, preds))
            recal.append(recall_score(gts, preds))
            f1.append(f1_score(gts, preds))
        return acc, recal, f1

    def print_eval(self):
        local_acc, local_recal, local_f1 = self.eval_local_all()
        ground_acc, ground_recal, ground_f1 = self.eval_ground_all()

        iou = self.eval_iou()
        ground_roc = self.eval_ground_roc()
        local_roc = self.eval_local_roc()

        print('ground roc auc: {:4f}'.format(ground_roc))
        print('local roc auc: {:4f}'.format(local_roc))
        ground_acc_row = ['ground_accuracy'] + ground_acc
        ground_recal_row = ['ground_recal'] + ground_recal
        ground_f1_row = ['ground_f1'] + ground_f1
        local_acc_row = ['local_accuracy'] + local_acc
        local_recal_row = ['local_recal'] + local_recal
        local_f1_row = ['local_f1'] + local_f1
        iou_row = ['IoU'] + iou
        header = ['thresh'] + list(self.threshes)
        print(tabulate([local_acc_row, local_recal_row, local_f1_row, ground_acc_row, ground_recal_row, ground_f1_row, iou_row], headers=header, tablefmt='orgtbl'))


class MultiLabelEvaluator(object):
    def __init__(self, ):
        self.reset()

    def reset(self):
        self.scores = [] # store all scores
        self.gts = [] # store all gts
        self.threshes = np.linspace(0, 1, 21)
        self.total_nums = [0 for _ in range(len(self.threshes))]
        self.correct_nums = [0 for _ in range(len(self.threshes))]

    def update(self, scores, gts):
        """
        :param scores: (n,)
        :param gts: (n,) \in {0, 1}
        :param masks: list of n (h, w) \in {0, 1}
        :return:
        """
        self.scores.extend(scores)
        self.gts.extend(gts)
        scores = np.array(scores)
        gts = np.array(gts)
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            self.total_nums[i] += preds.sum()
            self.correct_nums[i] += (preds * gts).sum()

    def eval_acc(self):
        """
        eval accuracy in different threshold
        :return: acc: list of accuracy
        """
        scores = np.array(self.scores)
        gts = np.array(self.gts)
        acc = []
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            acc.append(accuracy_score(gts, preds))

        # accuracy
        # acc = [correct_num / (total_num + 1e-6) for (correct_num, total_num) in zip(self.correct_nums, self.total_nums)]
        return acc

    def eval_f1(self):
        """
        eval f1 score in different threshold
        micro (global) f1 score is considered
        :return: f1
        """
        # f1 score
        scores = np.array(self.scores)
        gts = np.array(self.gts)
        f1 = []
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            f1.append(f1_score(gts, preds))
        return f1

    def eval_recal(self):
        """
        eval recal score in different threshold
        micro reval score is considered
        :return: f1
        """
        scores = np.array(self.scores)
        gts = np.array(self.gts)
        recal = []
        for i, thresh in enumerate(self.threshes):
            preds = (scores > thresh).astype(int)
            recal.append(recall_score(gts, preds))
        return recal


    def eval_roc(self):
        """
        regard it as multi-label classification, which have no effect about threshold
        :return: roc: float
        """
        roc = float(roc_auc_score(self.gts, self.scores))
        return roc

    def print_eval(self):
        acc = self.eval_acc()
        recal = self.eval_recal()
        f1 = self.eval_f1()
        roc = self.eval_roc()
        print('roc auc: {:4f}'.format(roc))
        acc_row = ['accuracy'] + acc
        recall_row = ['recal'] + recal
        f1_row = ['f1'] + f1
        header = ['thresh'] + list(self.threshes)
        print(tabulate([acc_row, recall_row, f1_row], headers=header, tablefmt='orgtbl'))

