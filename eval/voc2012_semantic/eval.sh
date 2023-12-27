# !/bin/bash

set -x

NUM_GPUS=1
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=11

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

# CKPT_PATH="/c1/kangsan/Painter/checkpoint/${CKPT_FILE}"
CKPT_PATH="/c1/kangsan/Painter/models/new1227/checkpoint-14.pth"
DST_DIR="/c1/kangsan/Painter/models_inference/VOC2012_semseg_inference/demo${PROMPT}"

# inference
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=29504 --use_env \
  eval/voc2012_semantic/painter_inference_segm.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE}

# postprocessing and eval
python eval/voc2012_semantic/VOC2012SemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
