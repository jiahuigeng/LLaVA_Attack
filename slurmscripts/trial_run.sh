#!/bin/bash
#
#SBATCH --job-name=olive-trial
#SBATCH --output=/storage/ukp/work/thytran/sweetolive/LLaVA_Attack/slurmlogs/%x-%j.output
#SBATCH --mail-user=thy.tran@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a6000"

# Activate and install basic packages.
source /storage/ukp/work/$USER/slurm_cmds/load_env.sh
source /storage/ukp/work/$USER/slurm_cmds/load_conda.sh
conda activate llava
export PYTHONPATH="${PYTHONPATH}:/storage/ukp/work/thytran/sweetolive/LLaVA_Attack"

echo "Start running ...."
export WANDB_PROJECT=sweetolive
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k/llava_v1_5_mix665k---textvqa-trial.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$SLURM_JOB_ID-llava-v1.5-13b-task-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
