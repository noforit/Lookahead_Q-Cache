#!/bin/bash
#SBATCH -J longbench_baseline
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

method=SnapKV # Support PyramidKV, SnapKV, H2O, StreamingLLM


max_capacity_prompts=256 

attn_implementation=flash_attention_2 # Support "flash_attention_2", "sdpa", "eager".

model_path=/your/path/to/Mistral-7B-Instruct-v0.2

save_dir=results/

window_size=32
datasets="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"


python3 run_longbench_baselines.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --window_size ${window_size} \
    --datasets ${datasets}

model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')


eval_path="${save_dir}/${model_name}_${max_capacity_prompts}"
python3 eval.py --results_dir ${eval_path}
echo "eval_path: $eval_path"