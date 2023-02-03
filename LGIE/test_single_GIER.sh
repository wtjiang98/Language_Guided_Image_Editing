

# NOTE: change the path!

PROJECT_PATH="/mnt/data1/jwt/LGIE-Django"
PYTHON_PATH="/home/jwt/anaconda3/bin/python3"

cd $PROJECT_PATH/LGIE
$PYTHON_PATH test_single.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 --dataset_mode my \
--label_dir None --image_dir None --anno_path /learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 0 --batchSize 1 \
--nThreads 1 --lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade \
--spade_attn --how_many 1 --which_epoch 80 \
--input_path $1 --request "$2" --output_path $3

