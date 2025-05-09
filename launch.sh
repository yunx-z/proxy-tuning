model=$1
dataset=$2
experiment_id=$3

alpha_strategy="warmup"

if [[ "$model" == *"small"* ]]; then
    gpu_cnt=1
    batch_sz=1
else
    gpu_cnt=2
    batch_sz=1
fi

if [[ "$dataset" == *"train"* ]]; then
	experiment_id=0
elif [[ "$experiment_id" = "" ]]; then 
	experiment_id=$(uuidgen)
fi

# if [[ "$dataset" == "MATH_hard_test" ]]; then
#     do_sample="False"
# else
#     do_sample="True"
# fi
do_sample="True"

max_generation_token=16384

job_name="${model}--${dataset}--${experiment_id}"
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
##SBATCH --exclude=gl1506
##SBATCH --dependency=afterany:24617369:24617368

echo \"My job ID is \$SLURM_JOB_ID\"
echo \"Running on host \$(hostname)\"
echo \"Starting at \$(date)\"

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate proxy-tuning-llama

batch_sz=${batch_sz}
experiment_id=${experiment_id}
dataset=\"${dataset}\"
# Evaluating DExperts with chat expert
my_data_dir=\"data/eval/\${dataset}/\"

large_size=\"32\"
small_size=\"1.5\"
small_rft_expert_model=\"checkpoints/small_rft_expert_model\"
small_rft_lora_expert_model=\"checkpoints/small_rft_lora_expert_model\"
small_pft_expert_model=\"checkpoints/small_pft_expert_model\"
small_pft_lora_expert_model=\"checkpoints/small_pft_lora_expert_model\"
small_short_ft_lora_expert_model=\"checkpoints/small_short_ft_lora_expert_model\"
small_early_pft_expert_model=\"checkpoints/small_early_pft_expert_model\"
small_50_pft_expert_model=\"checkpoints/small_50_pft_expert_model\"
small_distill_expert_model=\"checkpoints/small_distill_expert_model\"
small_rl_expert_model=\"agentica-org/DeepScaleR-\${small_size}B-Preview\"
small_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${small_size}B\"

if [[ \"${model}\" == *\"dexperts\"* ]]; then
	large_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${large_size}B\"
	if [[ \"${model}\" == \"dexperts-rft\" ]]; then
		small_expert_model=\${small_rft_expert_model}
	elif [[ \"${model}\" == \"dexperts-distill\" ]]; then
		small_expert_model=\${small_distill_expert_model}
	elif [[ \"${model}\" == \"dexperts-pft\" ]]; then
		small_expert_model=\${small_pft_expert_model}
	elif [[ \"${model}\" == \"dexperts-early-pft\" ]]; then
		small_expert_model=\${small_early_pft_expert_model}
	elif [[ \"${model}\" == \"dexperts-50-pft\" ]]; then
		small_expert_model=\${small_50_pft_expert_model}
	elif [[ \"${model}\" == \"dexperts-rft-lora\" ]]; then
		small_expert_model=\${small_rft_lora_expert_model}
	elif [[ \"${model}\" == \"dexperts-pft-lora\" ]]; then
		small_expert_model=\${small_pft_lora_expert_model}
	elif [[ \"${model}\" == \"dexperts-short-ft-lora\" ]]; then
		small_expert_model=\${small_short_ft_lora_expert_model}
	else
		small_expert_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${small_size}B\"
	fi
	large_base_model=\"Qwen/Qwen2.5-\${large_size}B\"
	small_base_model=\"Qwen/Qwen2.5-Math-\${small_size}B\"
elif [[ \"${model}\" == *\"RLproxy\"* ]]; then
	large_expert_model=\"\"
	small_expert_model=\${small_rl_expert_model}
	small_base_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${small_size}B\"
	large_base_model=\"deepseek-ai/DeepSeek-R1-Distill-Qwen-\${large_size}B\"
fi

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


if [[ \"${model}\" == *\"dexperts\"* || \"${model}\" == *\"RLproxy\"* ]]; then
	results_dir=\"results/\${dataset}/${model}-S\${small_size}B-L\${large_size}B/${alpha_strategy}/\${experiment_id}\"
	echo \"Results dir: \${results_dir}\"
	while true; do
		if [ -f \"\${results_dir}/predictions.jsonl\" ]; then
			line_count=\$(wc -l < \"\${results_dir}/predictions.jsonl\")
			if [ \"\$line_count\" -ge 30 ]; then
				echo \"predictions.jsonl has 30 or more lines. Exiting loop.\"
				break
			fi
			echo \"predictions.jsonl has \$line_count lines, still running evaluation...\"
		else
			echo \"predictions.jsonl does not exist yet. Running evaluation...\"
		fi
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

		sleep 5 # Wait for a few seconds before checking again
	done
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
