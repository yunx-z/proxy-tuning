module load cuda/12.6.3
module load gcc/13.2.0

batch_sz=1
dataset="aime2024"
# Evaluating DExperts with chat expert
my_data_dir="data/eval/${dataset}/"

large_size="7"
small_size="1.5"
large_base_model="Qwen/Qwen2.5-Math-${large_size}B"
large_expert_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-${large_size}B"
small_base_model="Qwen/Qwen2.5-Math-${small_size}B"
small_expert_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-${small_size}B"

models=(${large_base_model} ${large_expert_model} ${small_expert_model} ${small_base_model})

mkdir -p results/${dataset}/

# Evaluating DExperts
results_dir="results/${dataset}/dexperts-${large_size}B"
echo "Results dir: ${results_dir}"
python -m eval.gsm.run_eval \
    --max_examples 1 \
    --data_dir ${my_data_dir} \
    --save_dir ${results_dir} \
    --base_model_name_or_path ${large_base_model} \
    --expert_model_name_or_path ${small_expert_model} \
    --anti_expert_model_name_or_path ${small_base_model} \
    --eval_batch_size ${batch_sz}

for model in "${models[@]}"; do
	results_dir="results/${dataset}/${model//\//_}"
	echo "Results dir: ${results_dir}"
	python -m eval.gsm.run_eval \
	    --max_examples 1 \
	    --data_dir ${my_data_dir} \
	    --save_dir ${results_dir} \
	    --model_name_or_path ${model} \
	    --eval_batch_size ${batch_sz}
done
