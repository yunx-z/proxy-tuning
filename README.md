# Elicit Long Chain-of-Thought Reasoning 

## Setup

1. Create a conda environment with `python=3.9` and install packages via `pip install -r requirements.txt`
2. Adjust the batch_sz for generation [here](https://github.com/yunx-z/proxy-tuning/blob/main/launch.sh#L10)
3. This [script](https://github.com/yunx-z/proxy-tuning/blob/main/launch_all.sh) launches 16 runs for aime24/25/MATH_hard_test with the proxy-tuned model.
4. Run `python -m eval.compute_metrics`  to get pass@1 on all finished generations.
