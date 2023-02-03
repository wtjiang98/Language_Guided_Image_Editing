



#python test.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 --dataset_mode my --label_dir \
#/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks --image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json --label_nc 8 --no_instance \
#--contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 0,1,2 --batchSize 192 --nThreads 18 --lang_dim 256 \
#--lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --which_epoch 70


python test.py \
--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 --dataset_mode my --label_dir \
/mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks --image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images \
--anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json --label_nc 8 --no_instance \
--contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 1 --batchSize 60 --nThreads 8 --lang_dim 256 \
--lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --which_epoch 80


