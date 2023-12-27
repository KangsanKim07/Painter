# !/bin/bash

set -x

JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT="study_room_0005b/rgb_00094"

MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
CKPT_PATH="/c1/kangsan/Painter/checkpoint/${CKPT_FILE}"
DST_DIR="models_inference/kitti_depth_inference_${PROMPT}"
LORA_PATH="/c1/kangsan/Painter/models/painter_vit_large_kitti/checkpoint-1.pth"

# inference
python eval/kitti/painter_inference_depth.py \
  --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} --lora_path ${LORA_PATH}

# python eval/kitti/eval_with_pngs.py \
#   --pred_path ${DST_DIR} \
#   --gt_path datasets/nyu_depth_v2/official_splits/test/ \
#   --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
