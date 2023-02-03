

# *********************** test for 100 ***********************

python test.py \
--name base_FiveK_bilstm_feat1_vgg1 --dataset_mode my \
--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
--label_dir None --lang_dim 128 --gpu_ids 0 --batchSize 8


python test.py \
--name base_FiveK_bilstm_feat1_vgg1_gan0.1_v2 --dataset_mode my --label_dir None --three_input_D \
--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
--netG MY --gpu_ids 1 --batchSize 1 --nThreads 1 --lang_dim 128 --lang_encoder bilstm


# *********************** test for 104 ***********************

# 更新了一版代码到v2，将text_encoder也保存下来
python test.py \
--name base_FiveK_bilstm_feat1_vgg1_gan0.1_v2 --dataset_mode my --label_dir None --three_input_D \
--image_dir /mnt/data1/jwt/dataset/FiveK_for_zip/images \
--anno_path /mnt/data1/jwt/dataset/FiveK_for_zip/FiveK.json \
--netG MY --gpu_ids 1 --batchSize 10 --nThreads 10 --lang_dim 128 --lang_encoder bilstm


# *********************** test for 94 ***********************


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
python test.py \
--name LGIE_pattn_VGG5_l5_lang256 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn


# 试试512
python test.py \
--name base_FiveK_bilstm_feat1_vgg1_gan0.1_v2_512 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --load_size 562 --crop_size 512


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：不overfit了，看看效果
python test.py \
--name LGIE_pattn_VGG1_l5_lang256_v2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
python test.py \
--name LGIE_augpattn_VGG1_l5_lang256_v3 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn



# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#  L1 loss + vgg + GAN_uncondi + skip layer

python test.py \
--name LGIE_augpattn_VGG1_l10_uncond1_lang256_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --skiplayer


python test.py \
--name LGIEv2_augpattn_VGG1_l10_lang256_normIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade


python test.py \
--name LGIEv3_pattn_VGG1_l10_lang256_MYV2Fusion --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 2 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


python test.py \
--name LGIEv4_augpattn_VGG1_l10_lang256_normIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade


--norm_G spectralspadeinstance1x1 --encoder_nospade



python test.py \
--name LGIEv4_augpattn_VGG1_l10_lang256_normIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade


# MYV2FusionInpaint

python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionSInpaint

python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionSInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionSInpaint --gpu_ids 1 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# Inpaint

python test.py \
--name LGIEv4_pattn_VGG1_l10_Inpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG Inpaint --gpu_ids 0 --batchSize 1 --nThreads 1 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# FusionGatedInpaint
python test.py \
--name LGIEv4_pattn_VGG1_l10_MYV2FusionGated --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionGated --gpu_ids 0 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


python test.py \
--name LGIEv4_pattn_VGG1_l10_MYV2FusionInpaintRes --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintRes --gpu_ids 0 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion



python test.py \
--name LGIEv4_pattn_VGG1_unGAN1_l10_MYV2FusionInpaintSkip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip --gpu_ids 0 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion



# FiveK + MYV2FusionSInpaint
python test.py \
--name FiveK_pattn_VGG1_l15_lang256_MYV2FusionFiveK --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionFiveK --gpu_ids 0 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


python test.py \
--name FiveK_pattn_VGG1_l15_lang256_MYV2AttnFusionFiveK --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionFiveK --gpu_ids 3 --batchSize 4 --nThreads 1 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_attn



# MYV2FusionInpaint_test 看看是不是跟之前的一样
#python train.py \
#--name LGIEv4_pattn_VGG1_l5_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion

python test.py \
--name LGIEv4_pattn_VGG1_l5_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 3 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintSkip0: 只跳一小层
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintSkip0 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip0 --gpu_ids 1 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 15.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintSkip1: 跳两小层
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintSkip1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip1 --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 15.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintSkip0V2: 只跳一小层 V2: 不把inpaint变gray，并使得retouch避开inpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintSkip0V2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip0V2 --gpu_ids 3 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


#LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint
# MYV2FusionInpaintSkip1V2: 跳两小层  V2: 不把inpaint变gray，并使得retouch避开inpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintSkip1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip1V2 --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion




#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip1V2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip1V2 --gpu_ids 3 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintV0Skip1V2
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip1V2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip1V2 --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion



# MYV2FusionInpaintV0Skip1V2Batch
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip1V2Batch --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip1V2 --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion



python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_Attn --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 3 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn





# MYV2FusionInpaintV0Skip0V2Batch_RefineV2! 之前忘了加inpaint时候的self_attention
# AttnV2表示在做attention的时候需要reduce_sum
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_AttnV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 3 --batchSize 16 --nThreads 8 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn

python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_AttnV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 1 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 100


# MYV2FusionInpaintV0Skip0V3Simple!
# Simple_reduce_sum: 只用中间的一层做retouch
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 5 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn


python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 1 --batchSize 32 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 100



LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3Batch_Simple   MYV2FusionInpaintV0Skip0V3Simple

LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefineV2_Simple    MYV2FusionInpaintV0Skip0V3SimpleV1

LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3Refine_SimpleV1     MYV2FusionInpaintV0Skip0V3SimpleV1


FiveK_pattn_VGG1_l15_lang256_MYV2FusionInpaintV0Skip0V3SimpleFiveK  MYV2FusionInpaintV0Skip0V3SimpleFiveK

LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchSimpleRefine_allreq  MYV2FusionInpaintV0Skip0V3Simple

LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoid MYV2FusionInpaintV0Skip0V3Simple

LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 MYV2FusionInpaintV0Skip0V3Simple





--name FiveK_pattn_VGG1_l15_lang256_MYV2FusionInpaintV0Skip0V3SimpleFiveK --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json\
 --label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleFiveK --gpu_ids 0 \
 --batchSize 32 --nThreads 8 --lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 \
 --encoder_nospade --spade_attn --how_many 100 --which_epoch 24




