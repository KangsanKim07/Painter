#!/bin/bash

DATA_PATH=/c1/kangsan/Painter/datasets
name=new1227
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=12358 \
	--use_env semi_main_train.py  \
    --batch_size 8 \
    --accum_iter 4  \
    --model painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 2 \
    --data_path $DATA_PATH/ \
    --json_path  \
    $DATA_PATH/VOC2012/VOC2012_training_image_semantic.json \
    --val_json_path \
    $DATA_PATH/VOC2012/VOC2012_validation_image_semantic.json \
    --output_dir /c1/kangsan/Painter/models/$name \
    --log_dir /c1/kangsan/Painter/models/$name/logs \
    --finetune /c1/kangsan/Painter/models/new1227/checkpoint-14.pth
    # --finetune /c1/kangsan/Painter/checkpoint/painter_vit_large.pth \
    # --log_wandb \

