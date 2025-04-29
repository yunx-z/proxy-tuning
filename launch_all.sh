# for dataset in "aime2024" "aime2025" "MATH100"; do
# dataset_list=(MATH_hard_train_{00..48})
dataset_list=("MATH_hard_test") # "aime2025" "MATH_hard_test")

for dataset in "${dataset_list[@]}"; do
	if [[ "$dataset" == *"MATH_hard_train"* ]]; then
	    total_runs=1 # training data sampling
	else
	    total_runs=16 # evaluation
	fi
	# total_runs=1

	for i in $(seq 1 $total_runs); do
		# spgpu2
		bash launch.sh "DExperts" "constant" "$dataset" "$i"

		models=("small_expert_model" "large_base_model" "large_expert_model")
		for model in "${models[@]}"; do
			bash launch.sh "$model" "" "$dataset" "$i"
		done

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
