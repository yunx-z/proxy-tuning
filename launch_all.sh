# dataset_list=(MATH_hard_train_{00..48})
dataset_list=("aime2024") # "aime2025" "amc23" "MATH_hard_test")

for dataset in "${dataset_list[@]}"; do
	if [[ "$dataset" == *"MATH_hard_train"* ]]; then
	    total_runs=1 # training data sampling
	else
	    total_runs=8 # evaluation
	fi
	# total_runs=1

	# 1-4 spgpu 5-8 spgpu2
	# for i in {5..8}; do
	for i in $(seq 1 $total_runs); do
		bash launch.sh "dexperts-short-ft-lora" "$dataset" "$i"
		# spgpu2
		# bash launch.sh "dexperts" "$dataset" "$i"
		# bash launch.sh "dexperts-distill" "$dataset" "$i"
		# bash launch.sh "dexperts-rft" "$dataset" "$i"
		# bash launch.sh "dexperts-rft-lora" "$dataset" "$i"
		# bash launch.sh "dexperts-pft-lora" "$dataset" "$i"
		# bash launch.sh "RLproxy" "$dataset" "$i"
		# bash launch.sh "small_rl_expert_model" "$dataset" "$i"

		# models=("small_expert_model" "large_base_model" "large_expert_model")
		# models=("small_rft_expert_model" "small_pft_expert_model" "small_distill_expert_model")
		# for model in "${models[@]}"; do
		# 	bash launch.sh "$model" "$dataset" "$i"
		# done

		# spgpu
		# alpha_strategys=("constant0.25" "constant0.5" "constant2.0")
		# for alpha_strategy in "${alpha_strategys[@]}"; do
		# 	bash launch.sh "DExperts" "${alpha_strategy}" "$dataset"
		# done

		# alpha_strategys=("expert_logit_only" "expert_keyword_logits_only")
		# for alpha_strategy in "${alpha_strategys[@]}"; do
		# 	bash launch.sh "TwoBody" "${alpha_strategy}" "$dataset"
		# done
	done
done
