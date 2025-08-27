#!/bin/bash
#SBATCH -J longbench_LAQ
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
#SBATCH -p xxx
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module use --append /share/public/apps/modulefiles
module load cuda/12.4.1

. "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate LAQ



export CUDA_VISIBLE_DEVICES=0


method=LAQ # Support LAQ

max_capacity_prompts=128 # 128 256 512 

attn_implementation=flash_attention_2 # Support "flash_attention_2"

# model_path=/home/syji/model/AI-ModelScope/Mistral-7B-Instruct-v0.2
model_path=/your/path/to/Mistral-7B-Instruct-v0.2


save_dir=results/


lookahead_max_capacity_prompts="${max_capacity_prompts}"

lookahead_method=snapkv # snapkv in paper, but LAQ is orthogonal to methods such as SnapKV and PyramidKV.

lookahead_window_size=32 # This window_size is used for the lookahead_method.

max_lookahead_size=8
stage2_window_size=8 # This window_size is used in the decoding stage, and it is set to 0 for LAQ, 8 for LAQ.

datasets="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

python3 run_longbench_LAQ.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --lookahead_max_capacity_prompts ${lookahead_max_capacity_prompts} \
    --lookahead_method ${lookahead_method} \
    --max_lookahead_size ${max_lookahead_size} \
    --lookahead_window_size ${lookahead_window_size} \
    --stage2_window_size ${stage2_window_size} \
    --datasets ${datasets}


model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')

eval_path="${save_dir}/${model_name}_${max_capacity_prompts}"
python3 eval.py --results_dir ${eval_path}
echo "eval_path: $eval_path"


