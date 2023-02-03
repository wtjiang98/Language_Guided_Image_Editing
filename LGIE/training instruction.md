

## 1. 把 'LGIE/options/test_options.py' 中的
   
```python
parser.add_argument('--input_path', type=str, default='None', help='input_path of single image.')
```

挪到 'LGIE/options/base_options.py'中。最后的base_options.py如下：

```python
class BaseOptions():
  def __init__(self):
    self.initialized = False

  def initialize(self, parser):
    # experiment specifics
    parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
    parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    # for setting inputs
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
    parser.add_argument('--dataset_mode', type=str, default='coco')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
    parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
    parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

    # for displays
    parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

    # for generator
    parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--z_dim', type=int, default=256,
                        help="dimension of the latent z vector")

    # for instance-wise features
    parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
    parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

    # add by jwt
    parser.add_argument('--anno_path', type=str, default='', help='anno dict that specify pairs')
    parser.add_argument('--lang_encoder', type=str, default='bilstm', help='(static|bilstm)')
    parser.add_argument('--three_input_D', action='store_true', help='arch of D')
    parser.add_argument('--use_lang', action='store_true', help='arch of D')
    parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
    parser.add_argument('--lambda_MSE', type=float, default=0.0, help='weight for MSE loss')
    parser.add_argument('--lambda_unchange', type=float, default=1.0, help='weight for unchange loss')
    parser.add_argument('--predict_param', action='store_true', help='resnet to predict filter param')
    parser.add_argument('--buattn_norm', action='store_true', help='use norm in buattn')
    parser.add_argument('--use_buattn', action='store_true', help='use word attention')
    parser.add_argument('--ca_condition_dim', type=int, default=0, help='dim of ca operation')
    parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for gan loss')
    parser.add_argument('--use_op', action="store_true", default=False, help='whether to use operator annotation')
    parser.add_argument('--use_pattn', action='store_true', help='use word attention')
    parser.add_argument('--skiplayer', action='store_true', help='')
    parser.add_argument('--encoder_nospade', action='store_true', help='encoder_nospade')

    parser.add_argument('--spade_fusion', action='store_true', help='fuse seg with visual feature')
    parser.add_argument('--spade_attn', action='store_true', help='fuse seg with attention')
    parser.add_argument('--all_request', action='store_true', help='all_request')
    
    # 新加入的！
    parser.add_argument('--input_path', type=str, default='None', help='input_path of single image.')

    # add by wjy
    parser.add_argument('--ground', action='store_true', help='ground mode or not')

    self.initialized = True
    return parser
```


## 2. 运行LGIE/train.sh中唯一没有被注释的命令即可

```shell
# MYV2FusionInpaintV0Skip0V3Refine_SimpleSigmoidV1
# spade里用tanh，然后在W上加一个sigmoid就好
# 最后用的结果！！
python train.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 3 --batchSize 20 --nThreads 5 --lang_dim 256 \
--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn

```