# -*- coding: utf-8 -*
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainAllOptions(BaseOptions):
  def initialize(self, parser):
    BaseOptions.initialize(self, parser)
    # for displays
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

    # for training
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

    # the default values for beta1 and beta2 differ by TTUR option
    opt, _ = parser.parse_known_args()
    if opt.no_TTUR:
      parser.set_defaults(beta1=0.5, beta2=0.999)

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

    # for discriminators
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
    parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
    parser.add_argument('--lambda_kld', type=float, default=0.05)
    parser.add_argument('--lambda_gan_uncond', type=float, default=0.0)

    # ground
    # Data input settings
    parser.add_argument('--dataset', type=str, default='IER', help='name of dataset')
    parser.add_argument('--start_from', type=str, default=None, help='continuing training from saved model')
    parser.add_argument('--num_worker', type=int, default=8, help='number of worker')
    # FRCN setting
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101', help='net_name: res101 or vgg16')
    parser.add_argument('--iters', default=1250000, type=int, help='iterations we trained for faster R-CNN')
    parser.add_argument('--tag', default='notime', help='on default tf, don\'t change this!')
    # Visual Encoder Setting
    parser.add_argument('--visual_sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--visual_fuse_mode', type=str, default='concat', help='concat or mul')
    parser.add_argument('--visual_init_norm', type=float, default=20, help='norm of each visual representation')
    parser.add_argument('--visual_use_bn', type=int, default=-1, help='>0: use bn, -1: do not use bn in visual layer')
    parser.add_argument('--visual_use_cxt', type=int, default=1, help='if we use contxt')
    parser.add_argument('--visual_cxt_type', type=str, default='frcn', help='frcn or res101')
    parser.add_argument('--visual_drop_out', type=float, default=0.2, help='dropout on visual encoder')
    parser.add_argument('--window_scale', type=float, default=2.5, help='visual context type')
    # Visual Feats Setting
    parser.add_argument('--with_st', type=int, default=1, help='if incorporating same-type objects as contexts')
    parser.add_argument('--num_cxt', type=int, default=5, help='how many surrounding objects do we use')
    # Language Encoder Setting
    parser.add_argument('--word_embedding_size', type=int, default=512, help='the encoding size of each token')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')
    parser.add_argument('--word_drop_out', type=float, default=0.5, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0.2, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode')
    # Joint Embedding setting
    parser.add_argument('--use_op_prior', type=int, default=1, help='if use the operator as prior')
    parser.add_argument('--jemb_drop_out', type=float, default=0.1, help='dropout in the joint embedding')
    parser.add_argument('--jemb_dim', type=int, default=512, help='joint embedding layer dimension')
    # Loss Setting
    parser.add_argument('--att_weight', type=float, default=1.0, help='weight on attribute prediction')
    parser.add_argument('--visual_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (neg_ref, sent)')
    parser.add_argument('--lang_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (ref, neg_sent)')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for ranking loss')
    # Optimization: General
    parser.add_argument('--max_iters', type=int, default=50000, help='max number of iterations to run')
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in number of images per batch')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--seq_per_ref', type=int, default=3, help='number of expressions per object during training')
    parser.add_argument('--learning_rate_decay_start', type=int, default=8000, help='at what iter to start decaying learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=8000, help='every how many iters thereafter to drop LR by half')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    # Optimization: Specific
    parser.add_argument('--warmup', type=int, default=1000, help='the step of training classification before grounding')
    # Evaluation/Checkpointing
    parser.add_argument('--num_sents', type=int, default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2000, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', type=str, default='../output', help='directory to save models')
    parser.add_argument('--language_eval', type=int, default=0, help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--losses_log_every', type=int, default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.')
    # test
    parser.add_argument('--threshold', type=float, default=0.25, help='threshold for filter objects')
    parser.add_argument('--operation_threshold', type=float, default=0.4, help='the threshold for operation prediction')
    parser.add_argument('--ground_threshold', type=float, default=0.25, help='the threshold for operation prediction')
    parser.add_argument('--local_threshold', type=float, default=0.5, help='the threshold for operation prediction')
    # misc
    parser.add_argument('--op_pred_trial', type=str, default='1', help='an id identifying this run/job.')
    parser.add_argument('--ground_trial', type=str, default='1', help='an id identifying this run/job.')
    parser.add_argument('--id', type=str, default='1', help='an id identifying this run/job.')
    parser.add_argument('--session', type=int, default=3, required=True, help='session for the setting of the experiment.')
    parser.add_argument('--seed', type=int, default=24, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use, -1 = use CPU')
    # parser.add_argument('--phase_ground', type=str, default='train', help='phase')
    # adapt for gan
    parser.add_argument('--filter_req', type=str, default='v0', help='setting for filtered request list')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='load saved checkpoint')
    parser.add_argument('--epoch', type=int, default=20, help='epochs for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='the epoch to start from')
    parser.add_argument('--best_score', type=float, default=None, help='the best score of training')
    parser.add_argument('--start_iter', type=int, default=1, help='the iteration to start from')
    parser.add_argument('--val_mode', type=int, default=1, help='use validation set (1) or not (0)')
    parser.add_argument('--train_loss_threshold', type=float, default=None, help='save model when train loss is smaller than this threshold')
    parser.add_argument('--total_loss', type=float, default=0, help='total loss of training')
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--op_vocab_size', type=int, default=None)
    parser.add_argument('--pan_dim', type=int, default=None)
    parser.add_argument('--ground_gt', action='store_true', help='use gt in grounding or not')
    parser.add_argument('--use_expert', action='store_true', help='only use expert summary or not')
    parser.add_argument('--exclude_colorbg', action='store_true', help='exclude colorbg operation or not')

    # add by jwt
    self.isTrain = True
    return parser


