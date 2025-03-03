import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load tokenizers
base_model_name = "Qwen/Qwen2.5-Math-7B"  # Change to actual model path or name
expert_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Change to actual model path or name

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
expert_tokenizer = AutoTokenizer.from_pretrained(expert_model_name)

# Define the number of tokens to compare
num_tokens_to_check = 151936

# Extract token strings
base_tokens = base_tokenizer.convert_ids_to_tokens(list(range(num_tokens_to_check)))
expert_tokens = expert_tokenizer.convert_ids_to_tokens(list(range(num_tokens_to_check)))

print("len(base_tokens)", len(base_tokens))
print("len(expert_tokens)", len(expert_tokens))
print("base_tokens", base_tokens)
print("expert_tokens", expert_tokens)

# Compare tokens
mismatches = []
for idx, (base_tok, expert_tok) in enumerate(zip(base_tokens, expert_tokens)):
    if base_tok != expert_tok:
        mismatches.append((idx, base_tok, expert_tok))

# Print results
if not mismatches:
    print(f"The vocabularies are fully aligned for the first {num_tokens_to_check} tokens.")
else:
    print(f"Found {len(mismatches)} mismatches in the first {num_tokens_to_check} tokens.")
    for idx, base_tok, expert_tok in mismatches:  # Show first 10 mismatches
        print(f"Mismatch at index {idx}: base='{base_tok}' vs expert='{expert_tok}'")


# Load model and tokenizer
for model_name in tqdm([base_model_name, expert_model_name]):  # Replace with your model (e.g., "facebook/opt-1.3b", "meta-llama/Llama-2-7b-hf")
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Ensure model is on the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Input prompt
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate text
    output_ids = model.generate(input_ids, max_new_tokens=10000, do_sample=False)

    # Decode with special tokens kept
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Print result
    print("Generated Text:\n", output_text)

