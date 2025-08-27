#!/bin/bash
#SBATCH -J longbench
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 8:00:00
# SBATCH -w gpu08
# SBATCH -c 8
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --mem=64G

. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

. "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate LAQ


export CUDA_VISIBLE_DEVICES=0

    
method=LAQ # Support PyramidKV, SnapKV, H2O, StreamingLLM
model_provider=Mistral # Support LLaMA3, Mistral, qwen2

max_capacity_prompts=96
attn_implementation=flash_attention_2
# model_path=/home/syji/model/LLM-Research/Llama-3.2-1B-Instruct
# model_path=/home/syji/model/LLM-Research/Meta-Llama-3-8B-Instruct
# model_path=/home/syji/model/share_model/Meta-Llama-3.1-8B-Instruct
model_path=/home/syji/model/AI-ModelScope/Mistral-7B-Instruct-v0.2


lookahead_method=snapkv
max_lookahead_size=8
window_size=32 # for LAQ, this window_size is used in the lookahead stage, and it is usually set to 32.
stage2_window_size=8


TAG=test

mkdir -p results_needle/logs

(
python -u run_needle_in_haystack.py --s_len 800 --e_len 32001 \
    --model_provider ${model_provider} \
    --model_name ${model_path} \
    --attn_implementation ${attn_implementation} \
    --step 800 \
    --method $method \
    --max_capacity_prompt $max_capacity_prompts \
    --model_version ${model_provider}_${method}_${max_capacity_prompts}_${TAG} \
    --lookahead_method ${lookahead_method} \
    --max_lookahead_size ${max_lookahead_size} \
    --window_size ${window_size} \
    --stage2_window_size ${stage2_window_size}

) 2>&1  | tee results_needle/logs/${model_provider}_${method}_${max_capacity_prompts}_${TAG}.log

