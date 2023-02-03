

#echo $1
#echo $2
#echo $3

cd /mnt/data1/jwt/LGIE
/home/jwt/anaconda3/bin/python3 test_single.py \
--name 0_Fivek_MYV2FusionInpaintV0Skip0V3SimpleFiveK --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/splits/train_sess_1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleFiveK --gpu_ids 0 --batchSize 1 \
--nThreads 1 --lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade \
--spade_attn --how_many 1 --which_epoch 100 \
--input_path $1 --request "$2" --output_path $3

# 0_Fivek_MYV2FusionInpaintV0Skip0V3SimpleFiveK  --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/splits/train_sess_1.json
# FiveK_pattn_VGG1_l15_lang256_MYV2FusionInpaintV0Skip0V3SimpleFiveK  --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json
