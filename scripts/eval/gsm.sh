module load cuda/12.6.3
module load gcc/13.2.0

batch_sz=1
dataset="MATH500"
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
# "constant" "cycle100" "random0.5" "random0.2" 
for alpha_strategy in "random0.8"; do
	results_dir="results/${dataset}/dexperts-${large_size}B/${alpha_strategy}"
	echo "Results dir: ${results_dir}"
	python -m eval.gsm.run_eval \
	    --max_examples 2 \
	    --max_new_tokens 1024 \
	    --data_dir ${my_data_dir} \
	    --save_dir ${results_dir} \
	    --base_model_name_or_path ${large_base_model} \
	    --expert_model_name_or_path ${small_expert_model} \
	    --anti_expert_model_name_or_path ${small_base_model} \
	    --alpha_strategy ${alpha_strategy} \
	    --eval_batch_size ${batch_sz}
done

for model in "${models[@]}"; do
	results_dir="results/${dataset}/${model//\//_}"
	echo "Results dir: ${results_dir}"
	python -m eval.gsm.run_eval \
	    --max_examples 2 \
            --max_new_tokens 1024 \
	    --data_dir ${my_data_dir} \
	    --save_dir ${results_dir} \
    	    --base_model_name_or_path ${model} \
	    --eval_batch_size ${batch_sz}
done
