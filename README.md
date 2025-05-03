# Elicit Long Chain-of-Thought Reasoning 

## Setup

1. Create a conda environment with `python=3.9` and install packages via `pip install -r requirements.txt`
2. Adjust the batch_sz for generation [here](https://github.com/yunx-z/proxy-tuning/blob/main/launch.sh#L10)
3. This [script](https://github.com/yunx-z/proxy-tuning/blob/main/launch_all.sh) launches 16 runs for aime24/25/MATH_hard_test with the proxy-tuned model.
4. Run `python -m eval.compute_metrics`  to get pass@1 on all finished generations.

Note on Output and Resuming:
- The model's predictions are saved incrementally to a file named `predictions.jsonl` within the specified {save_dir}. This happens after each batch of generations is finished.
- If the `predictions.jsonl` file already contains a certain number of predictions (let's say N), rerunning the launch command will automatically resume from the (N+1)th question, preventing redundant computations.

The output directory will look like this:
```
results/aime2024/dexperts-S1.5B-L32B/constant/
├── 1
│   ├── example_prompt.txt
│   ├── logits.log
│   ├── metrics.json
│   └── predictions.jsonl
├── 2
│   ├── example_prompt.txt
│   ├── logits.log
│   ├── metrics.json
│   └── predictions.jsonl
...
```
