#!/bin/bash

PROMPT_VERSION=v1

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --model_name_or_path /root/autodl-tmp/models/llava-v1.6-vicuna-7b \
    --version $PROMPT_VERSION \
    --data_path /root/autodl-tmp/LLAVA/playground/data/ScienceQA/scienceqa_train_llava.json \
    --image_folder /root/autodl-tmp/LLAVA/playground/data/ScienceQA/images \
    --vision_tower /root/autodl-tmp/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --image_aspect_ratio anyres \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /root/autodl-tmp/outputs/llava_scienceqa_lora_r64_lr2e5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none