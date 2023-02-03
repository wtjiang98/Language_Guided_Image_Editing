#python test_all.py \
#--name 0_traingan_expert_MYV2FusionInpaintV0Skip0V3Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 0 --batchSize 20 --nThreads 8 \
#--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 100 \
#--session 3 --id wjy_test --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 0 --ground_gt \
#--which_epoch 5 --use_expert --filter_req expert

#python test_all.py \
#--name 0_traingan_expert_excolorbg_MYV2FusionInpaintV0Skip0V3Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 8 \
#--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 10000 \
#--session 3 --id wjy_test --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2 \
#--ground_gt --which_epoch 100

#python test_all.py \
#--name 0_traingan_expert_excolorbg_MYV2FusionInpaintV0Skip0V3Simple_WoIRD --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 8 \
#--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --how_many 10000 \
#--session 3 --id wjy_test --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2 \
#--ground_gt --which_epoch 100

#python test_all.py \
#--name 0_traingan_expert_excolorbg_MYV2FusionInpaintV0Skip0V3Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 8 \
#--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 100 \
#--session 3 --id wjy_test --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2 \
#--ground_gt --which_epoch 90 --use_expert --exclude_colorbg --filter_req expert_excolorbg

python test_fivek.py \
--name 0_Fivek_MYV2FusionInpaintV0Skip0V3SimpleFiveK --dataset_mode fivek --label_dir None \
--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/splits/test_sess_1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleFiveK --gpu_ids 0 --batchSize 20 --nThreads 8 \
--lang_dim 256 --lang_encoder bilstm --use_pattn --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn --how_many 10000 --which_epoch 30