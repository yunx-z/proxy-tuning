model=$1
alpha_strategy=$2
dataset=$3

if [[ "$model" == *"small"* ]]; then
    gpu_cnt=1
else
    gpu_cnt=4
fi

# if [[ "$dataset" == *"MATH"* ]]; then
#     do_sample="False"
# else
#     do_sample="True"
# fi

do_sample="True"
max_generation_token=16384

job_name="${model}--${alpha_strategy}--${dataset}"
echo """#!/bin/bash

#SBATCH --account=wangluxy_owned1
#SBATCH --job-name=${job_name}       # Name of the job
#SBATCH --output=logs/gl/${job_name}--%j.log   # File to which the output will be written
#SBATCH --error=logs/gl/${job_name}--%j.log     # File to which the error will be written
#SBATCH --time=07-00:00:00           # Wall time limit of the job (e.g., 1 hour)
#SBATCH --partition=spgpu2           # Partition (or queue) name
#SBATCH --nodes=1
#SBATCH --gres=gpu:${gpu_cnt}              # Request 1 GPU
#SBATCH --ntasks=1                # Number of tasks, typically set to 1 for single GPU jobs
#SBATCH --cpus-per-gpu=4         # Number of CPU cores per task
#SBATCH --mem=43GB                 # Amount of memory per node (e.g., 16 GB)

echo \"My job ID is \$SLURM_JOB_ID\"
echo \"Running on host \$(hostname)\"
echo \"Starting at \$(date)\"

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate proxy-tuning-llama

batch_sz=1
dataset=\"${dataset}\"
# Evaluating DExperts with chat expert
my_data_dir=\"data/eval/\${dataset}/\"

large_size=\"32\"
small_size=\"14\"
large_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${large_size}B\"
small_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${small_size}B\"
large_base_model=\"Qwen/Qwen2.5-\${large_size}B-Instruct\"
small_base_model=\"Qwen/Qwen2.5-\${small_size}B\"

# large_size=\"72\"
# small_size=\"1.5\"
# large_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${large_size}B\"
# small_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${small_size}B\"
# large_base_model=\"Qwen/Qwen2.5-Math-\${large_size}B-Instruct\"
# small_base_model=\"Qwen/Qwen2.5-Math-\${small_size}B\"

# large_size=\"70\"
# small_size=\"8\"
# large_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Llama-\${large_size}B\"
# small_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Llama-\${small_size}B\"
# large_base_model=\"meta-llama/Llama-3.3-\${large_size}B-Instruct\"
# small_base_model=\"meta-llama/Llama-3.1-\${small_size}B\"

experiment_id=\$(uuidgen)

if [[ \"${model}\" == \"DExperts\" ]]; then
	results_dir=\"results/\${dataset}/dexperts-S\${small_size}B-L\${large_size}B/${alpha_strategy}/\${experiment_id}\"
	echo \"Results dir: \${results_dir}\"
	python -m eval.gsm.run_eval \
	    --max_new_tokens ${max_generation_token} \
	    --data_dir \${my_data_dir} \
	    --save_dir \${results_dir} \
	    --base_model_name_or_path \${large_base_model} \
	    --expert_model_name_or_path \${small_expert_model} \
	    --anti_expert_model_name_or_path \${small_base_model} \
	    --alpha_strategy \"${alpha_strategy}\" \
	    --do_sample ${do_sample} \
	    --eval_batch_size \${batch_sz}
elif [[ \"${model}\" == \"SBLELB\" ]]; then
	results_dir=\"results/\${dataset}/SBLELB-S\${small_size}B-L\${large_size}B/${alpha_strategy}/\${experiment_id}\"
	echo \"Results dir: \${results_dir}\"
	python -m eval.gsm.run_eval \
	    --max_new_tokens ${max_generation_token} \
	    --data_dir \${my_data_dir} \
	    --save_dir \${results_dir} \
	    --base_model_name_or_path \${small_base_model} \
	    --expert_model_name_or_path \${large_expert_model} \
	    --anti_expert_model_name_or_path \${large_base_model} \
	    --alpha_strategy \"${alpha_strategy}\" \
	    --do_sample ${do_sample} \
	    --eval_batch_size \${batch_sz}
elif [[ \"${model}\" == \"TwoBody\" ]]; then
	results_dir=\"results/\${dataset}/TwoBody-S\${small_size}B-L\${large_size}B/${alpha_strategy}/\${experiment_id}\"
	echo \"Results dir: \${results_dir}\"
	python -m eval.gsm.run_eval \
	    --max_new_tokens ${max_generation_token} \
	    --data_dir \${my_data_dir} \
	    --save_dir \${results_dir} \
	    --base_model_name_or_path \${large_base_model} \
	    --expert_model_name_or_path \${small_expert_model} \
	    --alpha_strategy \"${alpha_strategy}\" \
	    --do_sample ${do_sample} \
	    --eval_batch_size \${batch_sz}
else
	results_dir=\"results/\${dataset}/\${${model}//\//_}/\${experiment_id}\"
	echo \"Results dir: \${results_dir}\"
	python -m eval.gsm.run_eval \
	    --max_new_tokens ${max_generation_token} \
	    --data_dir \${my_data_dir} \
	    --data_dir \${my_data_dir} \
	    --save_dir \${results_dir} \
	    --base_model_name_or_path \${${model}} \
	    --do_sample ${do_sample} \
	    --eval_batch_size \${batch_sz}
fi
echo \"Finished at \$(date)\"
""" > slurm/${job_name}.sh
sbatch slurm/${job_name}.sh
