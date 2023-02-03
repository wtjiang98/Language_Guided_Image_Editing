

#python train_all_fixground_fast.py \
#--name 0_traingan_MYV2FusionInpaintV0Skip0V3SimpleALL --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleALL --gpu_ids 1 --batchSize 20 --nThreads 8 --lang_dim 256 \
#--display_freq 1000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 1 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 1 \
#--ground_gt

python train_fivek.py \
--name 0_Fivek_MYV2FusionInpaintV0Skip0V3SimpleFiveK_WoIRD --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/splits/train_sess_1.json \
--netG MYV2FusionInpaintV0Skip0V3SimpleFiveK --gpu_ids 2 --batchSize 20 --nThreads 8 --lang_dim 256 --display_freq 3000 --lambda_unchange 0.0 \
--niter 100 --save_epoch_freq 1 --lang_encoder bilstm --use_pattn --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 \
--lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --print_freq 100