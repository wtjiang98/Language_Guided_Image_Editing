
# To train on the Facades or COCO dataset, for example.
#python train.py --name [experiment_name] --dataset_mode facades --dataroot [path_to_facades_dataset]
#python train.py --name [experiment_name] --dataset_mode coco --dataroot [path_to_coco_dataset]
#
## To train on your own custom dataset
#python train.py --name [experiment_name] --dataset_mode custom --label_dir [path_to_labels] -- image_dir [path_to_images] --label_nc [num_labels]
#
## coco
#--name test --dataset_mode coco --dataroot datasets/coco_stuff
#
## me
#--name test --dataset_mode custom --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip/masks --image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip/images
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip/IER2.json

# after modify  （包含dont care选项）
#python train.py \
#--name multihot --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 20 --nThreads 20

# 修正Discriminator的BUG
#python train.py \
#--name multihot_one_cap_50_v2 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 50 --three_input_D --display_freq 800

# 修正Discriminator的BUG, 只使用L1，不使用unchange
#python train.py \
#--name multihot_one_cap_50_v2_L1 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 20 --nThreads 20 --lang_dim 50 --three_input_D --display_freq 1200 --lambda_L1 1.0 --lambda_unchange 0.0 --niter 80

# L1=20, three_input_D
#python train.py \
#--name multihot_one_cap_50_v2_L1=20_Dinput3 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 20 --nThreads 20 --lang_dim 50 --three_input_D --display_freq 1200 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 80

# L1=20
#python train.py \
#--name multihot_one_cap_50_v2_L1=20 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 50 --display_freq 1200 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 80

# L1=20, three_input_D, opv2
#python train.py \
#--name multihot_one_cap_50_v3_L1=20 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v2.json \
#--label_nc 14 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 50 --display_freq 1200 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 80 --three_input_D


# L1=10, three_input_D, opv2, lang_dim=0
#python train.py \
#--name multihot_v3_L1=10 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v2.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 0 --display_freq 1200 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 50 --three_input_D


# L1=50, three_input_D, opv2, lang_dim=0
# 结果：不错，感觉比之前好。
#python train.py \
#--name multihot_v3_L1=50 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v2.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 20 --nThreads 20 --lang_dim 0 --display_freq 1200 --lambda_L1 50.0 --lambda_unchange 0.0 --niter 50 --three_input_D


# 是否是因为vgg和feat太大了，而GAN loss权重只是1/10而导致GAN loss无法下降？
# 结果：否。质量反而更差，并导致每张图左上角都有白色的斑点
#python train.py \
#--name multihot_v3_L1=20_feat=1_vgg=1 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v2.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 0 --display_freq 1200  --niter 80 --three_input_D \
#--lambda_L1 20.0 --lambda_unchange 0.0 --lambda_feat 1.0 --lambda_vgg 1.0


# L1=50, three_input_D, op v3, lang_dim=0
#python train.py \
#--name multihot_op=v3_L1=50 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v3.json \
#--label_nc 3 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1,2,3 \
#--batchSize 20 --nThreads 20 --lang_dim 0 --display_freq 600 --lambda_L1 50.0 --lambda_unchange 0.0 --niter 50 --three_input_D


# L1=20, three_input_D, op v4, lang_dim=0
#python train.py \
#--name multihot_op=v4_L1=20 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v4.json \
#--label_nc 2 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 20 --nThreads 20 --lang_dim 0 --display_freq 120 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 500 --three_input_D \
#--save_epoch_freq 100


# **** 要改动的地方：label_nc，代码中的列表use_op *****

# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=0
#python train.py \
#--name multihot_op=v1_L1=20 --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 4,5,6,7 \
#--batchSize 8 --nThreads 8 --lang_dim 0 --display_freq 120 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 500 --three_input_D \
#--save_epoch_freq 100

# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=50, predict_param
#python train.py \
#--name multihot_op=v1_L1=20_param --dataset_mode my --label_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/masks \
#--image_dir /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/images \
#--anno_path /mnt/data1/gc/LDIE/raw-dataset/IER2_for_zip_corrected/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1,2,3,5,6,7 \
#--batchSize 6 --nThreads 6 --lang_dim 50 --display_freq 120 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 500 --three_input_D \
#--save_epoch_freq 100 --predict_param


# **************** in 100-jwt ****************

## L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=50, predict_param, 使用buattn
#python train.py \
#--name multihot_op=v1_L1=20_param_buattn --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1 \
#--batchSize 2 --nThreads 2 --lang_dim 50 --display_freq 120 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 500 --three_input_D \
#--save_epoch_freq 100 --predict_param --buattn_norm --use_buattn

# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=50, predict_param
#python train.py \
#--name multihot_op=v1_L1=20_param --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0,1 \
#--batchSize 24 --nThreads 24 --lang_dim 50 --display_freq 120 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 1500 --three_input_D \
#--save_epoch_freq 100 --predict_param


# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=50, predict_param, 使用ca
# 结果： 非常糊，感觉非常不好

#python train.py \
#--name multihot_op=v1_L1=20_param_ca25 --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 \
#--batchSize 12 --nThreads 12 --lang_dim 50 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 --three_input_D \
#--save_epoch_freq 10 --predict_param --ca_condition_dim 25


# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=50, predict_param, 使用ca
#python train.py \
#--name multihot_op=v1_L1=10_param_ca25 --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 \
#--batchSize 12 --nThreads 12 --lang_dim 50 --display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 --three_input_D \
#--save_epoch_freq 10 --predict_param --ca_condition_dim 25

# L1=20, three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=0, predict_param
# 这里的v2表示改正了D的错误，v3表示每次重新获得input
#python train.py \
#--name multihot_op=v1_L1=20_param_v3 --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 \
#--batchSize 12 --nThreads 12 --lang_dim 0 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 --three_input_D \
#--save_epoch_freq 10 --predict_param

# L1=20, no! three_input_D, op v1(9个op，且对local和ids为空做了处理), lang_dim=0, predict_param
# 这里的v2表示改正了D的错误，v3表示每次重新获得input，v4表示D不接受真实图片作为输入
#python train.py \
#--name multihot_op=v1_L1=20_param_v4 --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 \
#--batchSize 12 --nThreads 12 --lang_dim 0 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --predict_param


# ******* 开始bilstm

#python train.py \
#--name base_bilstm --dataset_mode my --label_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 \
#--batchSize 12 --nThreads 12 --lang_dim 0 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10  --lang_encoder bilstm



# ************************ MIT_FiveK *************************


# L1=20, no_three_input_D, MIT_FiveK, only language, predict_param
#python train.py \
#--name multihot_op=v1_L1=20_param_FiveK --dataset_mode my --label_dir None \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 \
#--batchSize 12 --nThreads 12 --lang_dim 50 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --predict_param

# L1=20, three_input_D, MIT_FiveK, only language, predict_param
#python train.py \
#--name multihot_op=v1_L1=20_param_FiveK_3inputD --dataset_mode my --label_dir None \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --three_input_D \
#--batchSize 12 --nThreads 12 --lang_dim 50 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100  \
#--save_epoch_freq 10 --predict_param

# L1=20, three_input_D, MIT_FiveK, only language, predict_param, ca
#python train.py \
#--name multihot_op=v1_L1=20_param_FiveK_3inputD_ca --dataset_mode my --label_dir None \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 --three_input_D \
#--batchSize 12 --nThreads 12 --lang_dim 50 --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100  \
#--save_epoch_freq 10 --predict_param --ca_condition_dim 25




# ****************************  SenseTime Start **********************************

# /lustressd/jiangwentao/dataset/FiveK_for_zip
# /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip
# /mnt/data2/jwt/project-100/LDIE-Dataset/IER2_for_zip_final

# bilstm for MIT-FiveK noganfeat_novgg
#srun -p ha_vug --gres=gpu:1 --job-name=jwt_LGIE_noganFeat_novgg \
#python -u train.py \
#--name base_FiveK_bilstm_noganFeat_novgg --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/images \
#--anno_path /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 --lang_dim 128 \
# --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm \
#--no_ganFeat_loss --no_vgg_loss

#srun -p HA_senseAR --gres=gpu:1 --job-name=base_FiveK_bilstm_L1only \
#python -u train.py \
#--name base_FiveK_bilstm_ --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/images \
#--anno_path /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 7 --nThreads 7 --lang_dim 128 \
#--display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm \
#--no_ganFeat_loss --no_vgg_loss --lambda_gan 0.0


#srun -p HA_senseAR --gres=gpu:1 --job-name=base_FiveK_bilstm_feat2_vgg2_nogan \
#python -u train.py \
#--name base_FiveK_bilstm_feat2_vgg2_nogan_langdim256 --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/images \
#--anno_path /mnt/lustressd/jiangwentao/dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 7 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm \
#--lambda_feat 2 --lambda_vgg 2 --lambda_gan 0.0


############################### SenseTime End ***********************************


## bilstm for MIT-FiveK
# base_L1only
#python train.py \
#--name base_FiveK_bilstm_L1only --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0,1 --batchSize 2 --nThreads 2 --lang_dim 128 \
# --display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm \
#--no_ganFeat_loss --no_vgg_loss --lambda_gan 0.0

#
#
## bilstm for MIT-FiveK

# 更新了一版代码到v2，将text_encoder也保存下来
#python train.py \
#--name base_FiveK_bilstm_feat1_vgg1_gan0.1_v2 --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 1 --batchSize 10 --nThreads 10 --lang_dim 128 \
# --display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 2 --lang_encoder bilstm --lambda_feat 1 --lambda_vgg 1 --lambda_gan 0.1


## bilstm for MIT-FiveK
#python train.py \
#--name base_FiveK_bilstm --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/images \
#--anno_path /mnt/data2/jwt/project-100/LDIE-Dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 10 --nThreads 10 --lang_dim 128 \
# --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm

# ******** back to school 94-jwt ***********
# /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip

# 最初的版本:
#python train.py \
#--name base_FiveK_bilstm --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images \
#--anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 2 --batchSize 10 --nThreads 10 --lang_dim 128 \
# --display_freq 700 --lambda_L1 20.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 10 --lang_encoder bilstm


# 想法：试试只用L1会有什么效果。我感觉VGG loss还是有用的，因为VGG提取出来的feature代表了一些物体，如果只用L1则不能约束前后的物体一致
# 结论：
#python train.py \
#--name base_FiveK_bilstm_feat0_vgg0_gan0_l10_v2 --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images \
#--anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 12 --nThreads 12 --lang_dim 128 \
# --display_freq 700  --niter 80 --save_epoch_freq 2 --lang_encoder bilstm \
# --lambda_feat 0.0 --lambda_vgg 0.0 --lambda_gan 0.0 --lambda_L1 10.0 --lambda_unchange 0.0


# 想法：default + 512分辨率
# 结论：
#python train.py \
#--name base_FiveK_bilstm_feat1_vgg1_gan0.1_v2_512 --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images \
#--anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 1 --batchSize 2 --nThreads 2 --lang_dim 128 \
# --display_freq 700 --lambda_L1 10.0 --lambda_unchange 0.0 --niter 100 \
#--save_epoch_freq 2 --lang_encoder bilstm --lambda_feat 1 --lambda_vgg 1 --lambda_gan 0.1 \
#--load_size 562 --crop_size 512


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
#python train.py \
#--name LGIE_pattn_VGG5_l5_lang256 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0


#python train.py \
#--name LGIE_pattn_VGG2_l15_lang256 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 8 --nThreads 8 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 2 --lambda_gan 0 --lambda_L1 15.0 --lambda_feat 0



# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集
#python train.py \
#--name LGIE_pattn_VGG1_l5_lang256_v2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 3 --batchSize 8 --nThreads 8 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0


## 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
## v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#python train.py \
#--name LGIE_augpattn_VGG1_l5_lang256_v3 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 2 --batchSize 8 --nThreads 8 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0


## 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
## v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#python train.py \
#--name LGIE_augpattn_VGG1_l5_lang256_v3_skiplayer --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 2 --batchSize 8 --nThreads 8 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0 --skiplayer


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#python train.py \
#--name LGIE_augpattn_VGG1_l5_lang256_uncond1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 7 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 1 --lambda_L1 5.0 --lambda_feat 1 --lambda_gan_uncond 1 --three_input_D


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#python train.py \
#--name LGIE_augpattn_VGG1_l5_lang256_uncond1_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 --batchSize 6 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 1 --lambda_L1 5.0 --lambda_feat 1 --lambda_gan_uncond 1 --three_input_D --skiplayer


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
# 只有L1 loss + vgg + skip layer
#python train.py \
#--name LGIE_augpattn_VGG1_l5_lang256_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 7 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --three_input_D --skiplayer


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#  L1 loss + vgg + GAN_uncondi + skip layer
#python train.py \
#--name LGIE_augpattn_VGG1_l10_uncond1_lang256_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 1 --batchSize 7 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 1 --lambda_gan_uncond 1 --skiplayer


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#  L1 loss + vgg + GAN_uncondi + skip layer
#python train.py \
#--name LGIE_augpattn_VGG1_l10_uncond1_lang256_skip_normIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 7 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 1 --lambda_gan_uncond 1 --skiplayer --norm_G spectralspadeinstance1x1


# 给LGIE dataset加上MAttNet类似的attn用以区分language中inpaint和retouch的成分
# v2：分了训练和测试集；v3：使用了augPhraseAug，并从语言中预测了weight
#  L1 loss + vgg
# 新v2: inpaint放中间做，retouch放后面做；encoder里没有spade
#python train.py \
#--name LGIEv2_augpattn_VGG1_l10_lang256_normIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MY --gpu_ids 0 --batchSize 10 --nThreads 10 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade


# SimpleINGenerator
#python train.py \
#--name LGIEv2_pattn_VGG1_l10_lang512_SimpleIN --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG SimpleIN --gpu_ids 1 --batchSize 12 --nThreads 12 --lang_dim 512 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade

# SimpleIN, affine=True for Encoder
#python train.py \
#--name LGIEv3_pattn_VGG1_l10_lang512_SimpleIN_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG SimpleIN --gpu_ids 2 --batchSize 7 --nThreads 7 --lang_dim 512 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --skiplayer


# MYV2，优化embed计算速度；把不在的phrase强行置为0，看看是否是因为这里导致图像变灰
#python train.py \
#--name LGIEv3_pattn_VGG1_l10_lang512_MYV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2 --gpu_ids 1 --batchSize 8 --nThreads 8 --lang_dim 512 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade

# MYV2，优化embed计算速度；把不在的phrase不值为零，与上面做对比
#python train.py \
#--name LGIEv3_pattn_VGG1_l10_lang512_MYV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2 --gpu_ids 3 --batchSize 8 --nThreads 8 --lang_dim 512 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade


# MYV2Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
#python train.py \
#--name LGIEv3_pattn_VGG1_l10_lang256_MYV2Fusion --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2Fusion --gpu_ids 1 --batchSize 6 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2Fusion: 在每个spade的前面加上fusion（channel统一到lang_dim），把当前层的language feature和visual feature融合了之后再得到gamma beta
#python train.py \
#--name LGIEv3_pattn_VGG1_l10_lang256_MYV2Fusion_V2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2Fusion --gpu_ids 3 --batchSize 4 --nThreads 4 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
# LGIEv4 试试多卡
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2Fusion --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2Fusion --gpu_ids 0,1,2 --batchSize 18 --nThreads 18 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2Fusion: 在每个spade的前面加上fusion（channel统一到norm_nc），把当前层的language feature和visual feature融合了之后再得到gamma beta
# LGIEv4：单卡版本
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2Fusion_single --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2Fusion --gpu_ids 1 --batchSize 6 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion

# MYV2FusionInpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion --continue_train


# MYV2FusionSInpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionSInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionSInpaint --gpu_ids 2,3 --batchSize 12 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# Inpaint：只使用gated conv，看看图片糊到底是什么原因
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_Inpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG Inpaint --gpu_ids 0,1,2,3 --batchSize 88 --nThreads 20 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# InpaintV2：只使用gated conv，看看图片糊到底是什么原因
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_InpaintV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG InpaintV2 --gpu_ids 0,1,2,3 --batchSize 60 --nThreads 20 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionGated：只使用gated conv，看看图片糊到底是什么原因
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_MYV2FusionGated --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionGated --gpu_ids 0,1,2,3 --batchSize 20 --nThreads 12 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaint SkipLGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint_skip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion --skiplayer



# MYV2FusionInpaint + SN Discriminator
#python train.py \
#--name LGIEv4_pattn_VGG1_l5_GAN5_lang256_MYV2FusionInpaint_InpaintSA --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1,2,3 --batchSize 52 --nThreads 16 --lang_dim 256 --display_freq 700 \
#--lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --netD InpaintSA \
#--lambda_vgg 1 --lambda_gan 5.0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


#python train.py \
#--name LGIEv4_pattn_VGG1_l5_GAN5_lang256_MYV2FusionInpaint_InpaintSA_D5 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1,2,3 --batchSize 52 --nThreads 16 --lang_dim 256 --display_freq 700 \
#--lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --netD InpaintSA \
#--lambda_vgg 1 --lambda_gan 5.0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion --D_steps_per_G 5



#python train.py \
#--name LGIEv4_pattn_VGG0_l5_GAN1_lang256_MYV2FusionInpaint_InpaintSA_D5 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 24 --nThreads 8 --lang_dim 256 --display_freq 700 \
#--lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --netD InpaintSA \
#--lambda_vgg 0 --lambda_gan 1.0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion --D_steps_per_G 5



#python train.py \
#--name LGIEv4_pattn_VGG0_l5_GAN1_lang256_MYV2FusionInpaint_InpaintSA_D5_origanloss --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 2,3 --batchSize 24 --nThreads 8 --lang_dim 256 --display_freq 700 \
#--lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --netD InpaintSA \
#--lambda_vgg 0 --lambda_gan 1.0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion --D_steps_per_G 5




# MYV2FusionGated：只使用gated conv，看看图片糊到底是什么原因
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_GAN1_MYV2FusionGated_netD --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionGated --gpu_ids 0,1,2 --batchSize 20 --nThreads 12 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --netD InpaintSA \
#--lambda_vgg 1 --lambda_gan 1 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion



# MYV2FusionInpaintRes
# 在糊的版本上加入res shortcut
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_MYV2FusionInpaintRes --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintRes --gpu_ids 0,1,2,3 --batchSize 48 --nThreads 12 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaint + ori GAN loss
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_unGAN2_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 2 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaint + ori GAN loss

#python train.py \
#--name LGIEv4_pattn_VGG1_l10_unGAN2_D10_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 2,3 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 2 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion --D_steps_per_G 10


# MYV2FusionInpaint + ori GAN loss
#python train.py \
#--name LGIEv4_pattn_l20_unGAN1_D5_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0 --batchSize 14 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 0 --lambda_gan 0 --lambda_L1 20.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion --D_steps_per_G 5




# MYV2FusionInpaint + ori GAN loss
#python train.py \
#--name LGIEv4_pattn_l20_unGAN1_D1_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 1 --batchSize 14 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 0 --lambda_gan 0 --lambda_L1 20.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion --D_steps_per_G 1




# MYV2FusionInpaintRes
# 在糊的版本上加入res shortcut
#python train.py \
#--name LGIEv4_pattn_l10_unGAN1_MYV2FusionInpaintRes --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintRes --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 0 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintRes
# 在糊的版本上加入res shortcut
#python train.py \
#--name LGIEv4_pattn_l10_unGAN1_D5_MYV2FusionInpaintRes --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintRes --gpu_ids 3 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 0 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion --D_steps_per_G 5


# MYV2FusionInpaintSkip
# 在糊的版本上加入 skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_MYV2FusionInpaintSkip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip --gpu_ids 3 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintSkip
# 在糊的版本上加入 skip layer, MSE loss
#python train.py \
#--name LGIEv4_pattn_VGG1_MSE10_MYV2FusionInpaintSkip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn --spade_fusion \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 0 --lambda_MSE 10 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade



# MYV2FusionInpaintSkip
# 在糊的版本上加入 skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_unGAN1_l10_MYV2FusionInpaintSkip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip --gpu_ids 3 --batchSize 12 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintSkip
# 在糊的版本上加入 skip layert
#python train.py \
#--name LGIEv4_pattn_VGG1_unGAN1_l10_MYV2FusionInpaintSkip --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip --gpu_ids 2 --batchSize 12 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn  \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 1 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# FiveK
#python train.py \
#--name base_FiveK_bilstm_feat0_vgg0_gan0_l10_v2 --dataset_mode my --label_dir None --three_input_D \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images \
#--anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MY --gpu_ids 0 --batchSize 12 --nThreads 12 --lang_dim 128 \
# --display_freq 700  --niter 80 --save_epoch_freq 2 --lang_encoder bilstm \
# --lambda_feat 0.0 --lambda_vgg 0.0 --lambda_gan 0.0 --lambda_L1 10.0 --lambda_unchange 0.0


# FiveK + MYV2FusionSInpaint
#python train.py \
#--name FiveK_pattn_VGG1_l10_lang256_MYV2FusionSInpaint --dataset_mode fivek --label_dir None \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MYV2FusionSInpaint --gpu_ids 1 --batchSize 12 --nThreads 6 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# FiveK + MYV2FusionSInpaint
#python train.py \
#--name FiveK_pattn_VGG1_l15_lang256_MYV2FusionFiveK --dataset_mode fivek --label_dir None \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MYV2FusionFiveK --gpu_ids 1,2 --batchSize 14 --nThreads 7 --lang_dim 256 --display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 \
#--lang_encoder bilstm --use_pattn --lambda_vgg 1 --lambda_gan 0 --lambda_L1 15.0 --lambda_feat 0 --lambda_gan_uncond 0 \
#--norm_G spectralspadeinstance1x1 --encoder_nospade --spade_fusion


# FiveK + MYV2AttnFusionInpaint
#python train.py \
#--name FiveK_pattn_VGG1_l15_lang256_MYV2AttnFusionFiveK --dataset_mode fivek --label_dir None \
#--image_dir /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/images --anno_path /mnt/data1/jwt/LGIE-dataset/FiveK_for_zip/FiveK.json \
#--netG MYV2FusionFiveK --gpu_ids 0,1,2 --batchSize 30 --nThreads 15 --lang_dim 256 --display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 \
#--lang_encoder bilstm --use_pattn --lambda_vgg 1 --lambda_gan 0 --lambda_L1 15.0 --lambda_feat 0 --lambda_gan_uncond 0 \
#--norm_G spectralspadeinstance1x1 --encoder_nospade --spade_attn --continue_train



# MYV2FusionInpaint_test 看看是不是跟之前的一样
#python train.py \
#--name LGIEv4_pattn_VGG1_l5_lang256_MYV2FusionInpaint --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion



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


# MYV2FusionInpaintSkip1V2: 跳两小层  V2: 不把inpaint变gray，并使得retouch避开inpaint
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintSkip1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintSkip1V2 --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaint
# 还原最原始的 模糊设定，看看是否正确（结果：不正确）
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaint_test --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaint --gpu_ids 0,1 --batchSize 26 --nThreads 13 --lang_dim 256 \
#--display_freq 700 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintV0
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0 --gpu_ids 0 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintV0Skip0
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0 --gpu_ids 1 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
# --lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


# MYV2FusionInpaintV0Skip1
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip1 --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion



# MYV2FusionInpaintV0Skip0V2
# 从最接近的setting出发，开始加skip layer
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V2 --gpu_ids 3 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 1500 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion


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



# MYV2FusionInpaintV0Skip0V2Batch_Refine! 之前忘了加inpaint时候的self_attention
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3Batch_Refine --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 2 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion



# MYV2FusionInpaintV0Skip0V2Batch
#python train.py \
#--name LGIEv4_pattn_VGG1_l5_lang256_MYV2FusionInpaintV0Skip0V3Batch --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 1 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 5.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_fusion



# MYV2FusionInpaintV0Skip0V2Batch_Refine! 之前忘了加inpaint时候的self_attention
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_Attn --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 3 --batchSize 14 --nThreads 7 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn



# MYV2FusionInpaintV0Skip0V2Batch_RefineV2! 之前忘了加inpaint时候的self_attention
# AttnV2表示在做attention的时候需要reduce_sum
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_AttnV2 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3 --gpu_ids 3 --batchSize 16 --nThreads 8 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn



# MYV2FusionInpaintV0Skip0V3Simple!
# Simple_reduce_sum: 只用中间的一层做retouch
#python train.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 5 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn


#python train_all.py \
#--name check_wjy_all --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 0 --batchSize 2 --nThreads 5 --lang_dim 256 \
#--display_freq 100 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 10 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0


#python train_all_fixground.py \
#--name check_wjy_all_fixground --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 0 --batchSize 2 --nThreads 5 --lang_dim 256 \
#--display_freq 100 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 10 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0


#python train.py \
#--name check_wjy --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 20 --nThreads 0 --lang_dim 256 \
#--display_freq 3000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 12 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn

#python check_fixground.py \
#--name check_fixground --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 2 --batchSize 2 --nThreads 5 --lang_dim 256 \
#--display_freq 100 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 10 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2

#python train_all_fixground.py \
#--name check_wjy_all_fixground_MYV2FusionInpaintV0Skip0V3SimpleALL_time --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleALL --gpu_ids 2 --batchSize 20 --nThreads 5 --lang_dim 256 \
#--display_freq 100 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 10 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2

#python train_all_fixground_fast.py \
#--name check_wjy_all_fixground_fast_MYV2FusionInpaintV0Skip0V3SimpleALL --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3SimpleALL --gpu_ids 2 --batchSize 20 --nThreads 8 --lang_dim 256 \
#--display_freq 1000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 2

# MYV2FusionInpaintV0Skip0V3Refine_SimpleSigmoidV1
# spade里用tanh，然后在W上加一个sigmoid就好
#python train_all_fixground_fast.py \
#--name LGIEv4_pattn_VGG1_l10_lang256_MYV2FusionInpaintV0Skip0V3BatchRefine_SimpleSigmoidV1 --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 3 --batchSize 20 --nThreads 8 --lang_dim 256 \
#--display_freq 1000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 10 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 3 \
#--continue_train

#python train_all_fixground_fast.py \
#--name check_wjy --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 3 --batchSize 20 --nThreads 8 --lang_dim 256 \
#--display_freq 100 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 1 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 3 \
#--ground_gt --continue_train

#python wjy_tools.py \
#--name wjy_tools --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
#--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
#--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 3 --batchSize 20 --nThreads 8 --lang_dim 256 \
#--display_freq 1000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 1 --lang_encoder bilstm --use_pattn \
#--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
#--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 3 \
#--ground_gt

python train_all_fixground_fast.py \
--name 0_traingan_MYV2FusionInpaintV0Skip0V3Simple --dataset_mode my --label_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/masks \
--image_dir /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/images --anno_path /mnt/data1/jwt/LGIE-dataset/IER2_for_zip_final/learnable_op_v1.json \
--label_nc 8 --no_instance --contain_dontcare_label --netG MYV2FusionInpaintV0Skip0V3Simple --gpu_ids 3 --batchSize 20 --nThreads 8 --lang_dim 256 \
--display_freq 1000 --lambda_unchange 0.0 --niter 100 --save_epoch_freq 1 --lang_encoder bilstm --use_pattn \
--lambda_vgg 1 --lambda_gan 0 --lambda_L1 10.0 --lambda_feat 0 --lambda_gan_uncond 0 --norm_G spectralspadebatch1x1 --encoder_nospade --spade_attn \
--session 3 --id wjy_test --print_freq 100 --load_checkpoint /mnt/data1/jwt/LGIE/output/IER_ground_trial_1/best_dict.pth --learning_rate 0.0 --gpuid 3 \
--ground_gt

