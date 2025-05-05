import sys
import time
import random
import os
import json
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from collections import defaultdict
from modeling.utils import top_k_top_p_filtering
# Import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def _get_topk_info(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: int = 10):
    """
    Returns the top-k tokens and logits (as Python lists) for inspection.
    Assumes logits is 1D, i.e. shape [vocab_size].
    """
    top_values, top_indices = torch.topk(logits, k)
    top_values = top_values.tolist()
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    return top_tokens, top_values

class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0, # Note: alpha is mostly controlled by alpha_strategy now
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        log_file: Optional[str] = None,
        alpha_strategy: str = None,
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base_model_name_or_path = base_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.antiexpert_model_name_or_path = antiexpert_model_name_or_path
        # Ensure model_kwargs is a dict, handle None case
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.base = self.load_model(self.base_model_name_or_path, self.model_kwargs)
        self.expert = None # Lazy load in generate if needed
        self.antiexpert = None # Lazy load in generate if needed

        self.tokenizer = tokenizer
        self.device = self.base.device # Assume base model determines device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = bool(expert_model_name_or_path and 'chat' in expert_model_name_or_path.lower())

        if self.use_chat_format_for_expert:
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"
            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"
            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

        # State machine variables (only used if alpha_strategy='override_annealing')
        self.phase = "S1_ZERO"
        self.phase_step_count = 0
        self.alpha = 1.0 # Default alpha, dynamically changed by strategy

        self.log_file = log_file
        if self.log_file and os.path.exists(self.log_file):
             try: # Add try-except for potential permission issues etc.
                 os.remove(self.log_file)
             except OSError as e:
                 print(f"Warning: Could not remove existing log file {self.log_file}: {e}", file=sys.stderr)


        self.alpha_strategy = alpha_strategy
        self.MAX_EPISODE = 3 # Used only for 'ppt' strategy


    def load_model(self, model_name_or_path, _model_kwargs):
        if model_name_or_path:
            print(f"Loading model: {model_name_or_path}")
            # Ensure model loads to the correct device consistent with self.device
            a_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **_model_kwargs
            )
            print(model_name_or_path, "max_position_embeddings:", a_model.config.max_position_embeddings, flush=True)
            return a_model.eval()
        else:
            return None

    # Keep this method as it was used internally by the original generate
    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        # This method is effectively replaced by the concurrent execution
        # inside the new generate method. It might not be directly called anymore.
        # If needed elsewhere, it remains, but the generate loop won't use it.
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict) if self.expert else None
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict) if self.antiexpert else None
        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""
        # Ensure input_ids are on CPU for decode if necessary, though decode might handle device tensors
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        cleaned_prompts = []
        if self.chat_response_prefix:
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt", add_special_tokens=True # Assuming True is desired
        )
        # Move tokenizer outputs to the correct device
        chat_inputs = {k: v.to(self.device) for k, v in chat_inputs.items()}
        return chat_inputs


    def _update_phase(self, overriding_event_occurred: bool, extra_prompt_appended: bool):
        """
        State machine update logic (unchanged from original)
        """
        if extra_prompt_appended:
            self.phase = "S1_FINAL"
            self.phase_step_count = 0
        elif self.phase == "S1_ZERO":
            if self.phase_step_count >= 100:
                self.phase = "S2_ONE"
                self.phase_step_count = 0
        elif self.phase == "S2_ONE":
            if overriding_event_occurred:
                self.phase = "S1_ZERO" # Changed from S2_ONE_COUNTDOWN in original? Retaining S1_ZERO based on provided code.
                self.phase_step_count = 0
        elif self.phase == "S2_ONE_COUNTDOWN": # This state seems unused if S2_ONE goes to S1_ZERO
             if self.phase_step_count >= 100:
                 self.phase = "S1_ZERO"
                 self.phase_step_count = 0

        # Now set alpha/mode based on current phase
        if self.phase in ["S1_FINAL", "S1_ZERO"]:
            self.alpha = 0
        else: # S2_ONE, S2_ONE_COUNTDOWN
            self.alpha = 1

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        # This seems unused in the provided generate, but kept for potential external use
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))
        return analysis_data

    # Helper function for concurrent execution
    def _run_model_forward(self, model, input_ids, model_kwargs):
        """Prepares inputs and runs a single model forward pass."""
        if model is None:
            return None
        forward_begin_time = time.perf_counter()
        inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        with torch.no_grad(): # Ensure inference mode
             outputs = model(**inputs, return_dict=True)
        forward_end_time = time.perf_counter()
        # print(f"{model_kwargs['model_name']} forward time: {forward_end_time - forward_begin_time} s")
        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor = None,
        stopping_criteria = None,
        return_logits_for_analysis: bool = False,
        **kwargs # Base kwargs passed to all models initially
    ):
        """
        Modified generate method to:
         - Use concurrent execution for base, expert, and antiexpert models.
         - dynamically toggle alpha between 0 and 1 based on a finite-state machine
         - log JSON lines of top-10 tokens/logits each step to self.log_file
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        # Ensure input_ids are on the correct device
        input_ids = input_ids.to(self.device)

        # Initialize kwargs for each model based on the shared kwargs
        # Use deepcopy if kwargs contain mutable objects that shouldn't be shared unintentionally,
        # but usually past_key_values are replaced entirely, so shallow copy is fine.
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()
        base_kwargs["model_name"] = self.base_model_name_or_path
        expert_kwargs["model_name"] = self.expert_model_name_or_path
        antiexpert_kwargs["model_name"] = self.antiexpert_model_name_or_path

        # --- Lazy Loading of Expert/Antiexpert ---
        # Load models only if they haven't been loaded yet and are needed by the strategy
        # (Assume they are always potentially needed unless alpha is constantly 0,
        # but strategy can change, so load them if path is provided)
        needs_experts = self.expert_model_name_or_path or self.antiexpert_model_name_or_path
        if needs_experts:
            if self.expert is None and self.expert_model_name_or_path:
                self.expert = self.load_model(self.expert_model_name_or_path, self.model_kwargs)
            if self.antiexpert is None and self.antiexpert_model_name_or_path:
                self.antiexpert = self.load_model(self.antiexpert_model_name_or_path, self.model_kwargs)
        #-------------------------------------------

        # --- Prepare Expert Inputs (Chat Formatting) ---
        if self.expert and self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs['input_ids']
            # Ensure attention_mask from chat_inputs is used if present
            if 'attention_mask' in chat_inputs:
                 expert_kwargs['attention_mask'] = chat_inputs['attention_mask']
            # Make sure initial base/antiexpert attention masks match input_ids shape if not provided
            if 'attention_mask' not in base_kwargs and 'attention_mask' not in antiexpert_kwargs:
                 initial_mask = torch.ones_like(input_ids)
                 base_kwargs['attention_mask'] = initial_mask
                 antiexpert_kwargs['attention_mask'] = initial_mask.clone() # Use clone if modifying independently
            elif 'attention_mask' in base_kwargs and 'attention_mask' not in antiexpert_kwargs:
                 antiexpert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()


        else:
            expert_input_ids = input_ids # Use same input_ids if no chat format
            # Ensure all models have consistent initial attention masks if provided/needed
            if 'attention_mask' in base_kwargs:
                 expert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()
                 antiexpert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()
            elif 'attention_mask' not in base_kwargs:
                 # Create default mask if none provided in kwargs
                 initial_mask = torch.ones_like(input_ids)
                 base_kwargs['attention_mask'] = initial_mask
                 expert_kwargs['attention_mask'] = initial_mask.clone()
                 antiexpert_kwargs['attention_mask'] = initial_mask.clone()
            else: # base_kwargs has mask, but others don't yet
                 expert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()
                 antiexpert_kwargs['attention_mask'] = base_kwargs['attention_mask'].clone()

        # --- Generation Loop Setup ---
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id], device=input_ids.device) if self.tokenizer.eos_token_id else None

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        # --- Local helper to write JSON lines (Unchanged) ---
        def write_log(step: int, base_logits_1d: torch.Tensor,
                      dexperts_logits_1d: Optional[torch.Tensor], alpha: float, phase: str, episode: int, next_token: str):
            if base_logits_1d is None or not self.log_file: # Added check for self.log_file
                return
            base_top10_tokens, base_top10_vals = _get_topk_info(base_logits_1d, self.tokenizer, k=10)
            if dexperts_logits_1d is not None:
                dexperts_top10_tokens, dexperts_top10_vals = _get_topk_info(dexperts_logits_1d, self.tokenizer, k=10)
            else:
                dexperts_top10_tokens, dexperts_top10_vals = None, None

            log_obj = {
                "step": step, "phase": phase, "episode": episode, "alpha": alpha, "next_token": next_token,
                "base_top10_tokens": base_top10_tokens, "dexperts_top10_tokens": dexperts_top10_tokens,
                "base_top10_logits": base_top10_vals, "dexperts_top10_logits": dexperts_top10_vals,
            }
            try:
                 with open(self.log_file, "a", encoding="utf-8") as f:
                     f.write(json.dumps(log_obj) + "\n")
            except Exception as e:
                 print(f"Error writing to log file {self.log_file}: {e}", file=sys.stderr)


        # --- Main Generation Loop ---
        gen_steps = 0
        allowed_gen_steps = max_new_tokens
        extra_prompt_appended = False
        curr_episode = 0 # For 'ppt' strategy
        overriding_event_occurred = False # For 'override_annealing' strategy (reset each step?)
        self.phase = "S1_ZERO" # Reset phase machine state at the start of generation
        self.phase_step_count = 0

        # Determine max workers needed (base + expert + antiexpert)
        num_workers = 1 + (1 if self.expert else 0) + (1 if self.antiexpert else 0)

        # Use context manager for the executor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while gen_steps < allowed_gen_steps:
                step_begin_time = time.perf_counter()

                # --- Update Alpha based on Strategy ---
                # (Logic copied from original, potentially move some outside loop if constant)
                if extra_prompt_appended:
                    self.alpha = 0
                else:
                    # Simplified alpha calculation based on strategy
                    current_alpha_strategy = self.alpha_strategy if self.alpha_strategy else "constant" # Default to constant 1.0

                    if current_alpha_strategy.startswith("constant"):
                        try:
                            self.alpha = float(current_alpha_strategy.replace("constant", ""))
                        except ValueError:
                            self.alpha = 1.0 # Default if format is wrong
                    elif current_alpha_strategy == "warmup":
                        self.alpha = 0.0 if gen_steps < 100 else 1.0
                    elif current_alpha_strategy.startswith("cycle"):
                        T = int(current_alpha_strategy.replace("cycle", "") or 100) # Default cycle length 100
                        self.alpha = 1.0 if (gen_steps // T) % 2 != 0 else 0.0 # Cycle between 0 and 1
                    elif current_alpha_strategy.startswith("2cycles"):
                        items = current_alpha_strategy.split('-')
                        T0 = int(items[1]) if len(items) > 1 else 400
                        T1 = int(items[2]) if len(items) > 2 else 100
                        # Ensure Ts are multiples of 100? Original code asserted this.
                        # T0 = (T0 // 100) * 100
                        # T1 = (T1 // 100) * 100
                        cycle_len_steps = T0 + T1
                        step_in_cycle = gen_steps % cycle_len_steps
                        self.alpha = 0.0 if step_in_cycle < T0 else 1.0
                    elif current_alpha_strategy.startswith("random"):
                        prob = float(current_alpha_strategy.replace("random", "") or 0.5)
                        self.alpha = 1.0 if random.random() < prob else 0.0
                    elif current_alpha_strategy == "override_annealing":
                        # Note: overriding_event_occurred needs to be determined *after* logits comparison
                        self._update_phase(overriding_event_occurred, extra_prompt_appended) # self.alpha is set inside _update_phase
                    elif current_alpha_strategy == "ppt":
                        if curr_episode > self.MAX_EPISODE:
                            self.alpha = 0
                            # Optimization: potentially unload experts (consider implications if generate is called again)
                            # If unloading, ensure they are reloaded at the start of generate if needed.
                            # self.expert = None; self.antiexpert = None; torch.cuda.empty_cache() # Be cautious with this
                        else:
                            self.alpha = 1.0 # Assume alpha=1 during active PPT episodes
                    elif current_alpha_strategy in ["expert_logit_only", "expert_keyword_logits_only"]:
                        self.alpha = 1.0 # Alpha is used differently here, just enables expert pathway
                    else: # Default or unknown strategy
                        self.alpha = 1.0

                # --- Submit Model Forward Passes Concurrently ---
                futures = {}
                # 1. Base model
                futures['base'] = executor.submit(self._run_model_forward, self.base, input_ids, base_kwargs)

                # 2. Expert model (if exists)
                if self.expert:
                    futures['expert'] = executor.submit(self._run_model_forward, self.expert, expert_input_ids, expert_kwargs)

                # 3. Antiexpert model (if exists)
                if self.antiexpert:
                    futures['antiexpert'] = executor.submit(self._run_model_forward, self.antiexpert, input_ids, antiexpert_kwargs) # Uses base input_ids

                # --- Retrieve Results ---
                # Use a dictionary to store results as they complete
                results = {}
                try:
                    for future in as_completed(futures.values()):
                         # Find the key corresponding to this future
                         model_name = None
                         for name, f in futures.items():
                             if f == future:
                                 model_name = name
                                 break
                         if model_name:
                             results[model_name] = future.result() # Get output or exception
                         else:
                              # Should not happen if futures dict is managed correctly
                              print("Warning: Future completed but could not find corresponding model name.", file=sys.stderr)

                except Exception as e:
                    print(f"Error during concurrent model execution: {e}", file=sys.stderr)
                    # Handle error appropriately, maybe raise it or try to fallback/stop generation
                    raise e # Re-raise the exception

                # --- Extract Logits and Check for Errors ---
                base_outputs = results.get('base')
                expert_outputs = results.get('expert')
                antiexpert_outputs = results.get('antiexpert')

                if base_outputs is None:
                     raise RuntimeError("Base model forward pass failed.") # Base model is essential

                base_next_token_logits = base_outputs.logits[..., -1, :]
                expert_next_token_logits = expert_outputs.logits[..., -1, :] if expert_outputs else None
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :] if antiexpert_outputs else None

                # --- Determine Final Logits ---
                dexperts_next_token_logits = None # Initialize
                effective_alpha = self.alpha # Use the alpha determined by the strategy

                # Condition check needs self.expert AND self.antiexpert if using subtraction logic
                if effective_alpha != 0 and expert_next_token_logits is not None and antiexpert_next_token_logits is not None:
                    # Standard DExperts calculation
                    vocab_size = base_next_token_logits.shape[-1] # Use shape from logits directly
                    # Clone base logits
                    dexperts_next_token_logits = base_next_token_logits.clone()
                    # Apply DExperts modification safely up to common vocab size
                    # common_vocab = min(vocab_size, expert_next_token_logits.shape[-1], antiexpert_next_token_logits.shape[-1])
                    common_vocab = self.tokenizer.vocab_size
                    delta_logits = expert_next_token_logits[..., :common_vocab] - antiexpert_next_token_logits[..., :common_vocab]
                    delta_clipped = torch.clamp(delta_logits, -10, 10)
                    dexperts_next_token_logits[..., :common_vocab] = (
                        base_next_token_logits[..., :common_vocab] +
                        effective_alpha * delta_clipped
                    )
                    next_token_logits = dexperts_next_token_logits
                    # print("input_ids.shape", input_ids.shape[-1], flush=True)
                    # print("base_logits.std()", base_next_token_logits[..., :common_vocab].std(), flush=True)
                    # print("delta_logits.std()", delta_logits.std(), flush=True)

                # Condition check for expert_logit_only / expert_keyword_logits_only strategies
                elif effective_alpha != 0 and expert_next_token_logits is not None and antiexpert_next_token_logits is None:
                    vocab_size = base_next_token_logits.shape[-1]
                    common_vocab = min(vocab_size, expert_next_token_logits.shape[-1])
                    dexperts_next_token_logits = base_next_token_logits.clone() # Start with base

                    if self.alpha_strategy == "expert_logit_only":
                         dexperts_next_token_logits[..., :common_vocab] = (
                             base_next_token_logits[..., :common_vocab] +
                             effective_alpha * expert_next_token_logits[..., :common_vocab] # Add expert logits
                         )
                    elif self.alpha_strategy == "expert_keyword_logits_only":
                        special_tokens = ["Wait", "Alternatively", "Hmm"]
                        # Handle potential tokenization differences (e.g., leading space) robustly if needed
                        # This assumes tokenizer.convert_tokens_to_ids handles them correctly.
                        special_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
                        # Filter out None or invalid IDs
                        special_token_ids = [tid for tid in special_token_ids if tid is not None and tid != self.tokenizer.unk_token_id]

                        mask = torch.zeros(common_vocab, device=expert_next_token_logits.device)
                        valid_ids = [tid for tid in special_token_ids if tid < common_vocab]
                        if valid_ids: # Only create mask if there are valid IDs in range
                            mask[valid_ids] = 1

                        # Apply mask to expert logits before adding
                        filtered_expert_logits = expert_next_token_logits[..., :common_vocab] * mask
                        dexperts_next_token_logits[..., :common_vocab] = (
                            base_next_token_logits[..., :common_vocab] +
                            effective_alpha * filtered_expert_logits
                        )
                    else:
                        # Fallback or other strategies might just use base if only expert exists but strategy doesn't match
                         next_token_logits = base_next_token_logits

                    if dexperts_next_token_logits is not None : # If calculated above
                        next_token_logits = dexperts_next_token_logits
                    # else next_token_logits remains base_next_token_logits

                else: # alpha is 0 OR required experts are missing
                    next_token_logits = base_next_token_logits # Use base only

                # --- Check for Overriding Event (for 'override_annealing' strategy) ---
                overriding_event_occurred = False # Reset for this step
                if self.alpha_strategy == "override_annealing" and self.alpha != 0 and dexperts_next_token_logits is not None:
                     base_top1 = torch.argmax(base_next_token_logits[0], dim=-1)
                     dexperts_top1 = torch.argmax(dexperts_next_token_logits[0], dim=-1)
                     if base_top1.item() != dexperts_top1.item():
                         overriding_event_occurred = True # This will be used in the *next* iteration's _update_phase call


                # --- Logits Processing ---
                if logits_processor:
                    next_token_logits = logits_processor(input_ids, next_token_logits)

                # --- Sampling ---
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    # Ensure top_k_top_p_filtering handles potential NaNs/Infs from division if temp is 0?
                    # Usually temperature > 0.
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p) # Assumes top_k is not used here

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # --- Handle Finished Sequences ---
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                # --- Logging ---
                # Log based on the first sequence in the batch for simplicity
                next_token_str = self.tokenizer.decode(next_tokens[0])
                base_logits_1d_log = base_next_token_logits[0].detach().cpu() if base_next_token_logits is not None else None
                dexperts_logits_1d_log = dexperts_next_token_logits[0].detach().cpu() if dexperts_next_token_logits is not None else None

                write_log(
                     step=gen_steps,
                     base_logits_1d=base_logits_1d_log,
                     dexperts_logits_1d=dexperts_logits_1d_log,
                     alpha=self.alpha, # Log the alpha used in *this* step
                     phase=self.phase, # Log the phase active in *this* step
                     episode=curr_episode,
                     next_token=next_token_str,
                 )

                # --- Update Analysis Data (if requested) ---
                if return_logits_for_analysis:
                    # Gather step stats
                    next_token_logits_dict = {"dexperts": next_token_logits, "base": base_next_token_logits}
                    if expert_next_token_logits is not None:
                        next_token_logits_dict["expert"] = expert_next_token_logits
                    if antiexpert_next_token_logits is not None:
                        next_token_logits_dict["antiexpert"] = antiexpert_next_token_logits

                    # Store them (consider moving detach/cpu here if memory is an issue)
                    analysis_data["tokens"].append([self.tokenizer.decode(t) for t in next_tokens])
                    analysis_data["token_ids"].append(next_tokens.clone()) # Clone if tensor might be modified later
                    for m_name, val in next_token_logits_dict.items():
                        if val is not None:
                            analysis_data[f"logits_{m_name}"].append(val.unsqueeze(dim=1).clone()) # Clone logits

                # --- Update input_ids for next iteration ---
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if self.expert: # Update expert_input_ids only if expert exists
                     if self.use_chat_format_for_expert:
                         expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)
                     else:
                         expert_input_ids = input_ids # Keep aligned if no chat format

                # --- Update Model Kwargs (KV Cache, Attention Mask) ---
                # IMPORTANT: Use the outputs retrieved from the futures
                if base_outputs:
                    base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs, input_ids.shape[0])
                if expert_outputs:
                    expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs, expert_input_ids.shape[0])
                if antiexpert_outputs:
                    # Antiexpert uses base input_ids shape for mask update
                    antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs, input_ids.shape[0])


                # --- Check Stopping Criteria ---
                # 1. EOS Token
                if eos_token_id_tensor is not None:
                    if not torch.is_tensor(unfinished_sequences): # Ensure it's a tensor
                         unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

                    # Correctly check against multiple EOS tokens if eos_token_id_tensor has more than one
                    eos_candidates = next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    is_eos = eos_candidates.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0) # 0 if EOS is found
                    unfinished_sequences = unfinished_sequences.mul(is_eos) # Update unfinished sequences

                    if unfinished_sequences.max() == 0:
                         break # All sequences finished

                # 2. Custom Stopping Criteria
                # if stopping_criteria is not None and stopping_criteria(input_ids, None): # score typically not needed here
                #      break

                # --- Check for 'Thinking Token' (for 'ppt' strategy) ---
                # (This logic seems independent of concurrency)
                # Check the first sequence's next token
                # next_token_str was already decoded for logging
                if self.alpha_strategy == "ppt" and self.is_thinking_token(next_token_str):
                      curr_episode += 1


                # --- Force Answer Generation (End of Max Tokens) ---
                gen_steps += 1 # Increment generation step counter *before* the check

                if gen_steps == max_new_tokens and not extra_prompt_appended:
                     extra_prompt = "\nI'm not allowed to think more so I have to conclude that the final answer is:"
                     extra_input = self.tokenizer(extra_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(input_ids.device)

                     # Expand to batch size
                     if extra_input.shape[0] == 1 and input_ids.shape[0] > 1:
                         extra_input = extra_input.expand(input_ids.shape[0], -1)

                     # Append to input_ids
                     input_ids = torch.cat([input_ids, extra_input], dim=-1)
                     if self.expert: # Append to expert_input_ids too
                         expert_input_ids = torch.cat([expert_input_ids, extra_input], dim=-1)

                     # Update attention masks for all models
                     extra_len = extra_input.shape[1]
                     for current_kwargs in [base_kwargs, expert_kwargs, antiexpert_kwargs]:
                         if "attention_mask" in current_kwargs and torch.is_tensor(current_kwargs["attention_mask"]):
                             extra_attention = torch.ones((current_kwargs["attention_mask"].shape[0], extra_len),
                                                          device=input_ids.device,
                                                          dtype=current_kwargs["attention_mask"].dtype)
                             current_kwargs["attention_mask"] = torch.cat([current_kwargs["attention_mask"], extra_attention], dim=-1)
                             # Also update cache_position if used explicitly
                             if "cache_position" in current_kwargs and torch.is_tensor(current_kwargs["cache_position"]):
                                  current_kwargs["cache_position"] += extra_len


                     allowed_gen_steps += 100 # Extend generation limit
                     extra_prompt_appended = True
                     self.phase = "S1_FINAL" # Trigger final phase (alpha=0)
                     self.phase_step_count = 0 # Reset step count for final phase


                # --- Update Phase Step Count (for 'override_annealing') ---
                if self.alpha_strategy == "override_annealing":
                    self.phase_step_count += 1 # Increment steps *after* all step logic
                step_end_time = time.perf_counter()
                # print(f"step time: {step_end_time - step_begin_time} s")

        # --- End of Loop ---

        if return_logits_for_analysis:
            # Concatenate logits across steps
            for k in list(analysis_data.keys()): # Iterate over keys copy
                if k.startswith('logits_') and analysis_data[k]:
                    try:
                         analysis_data[k] = torch.cat(analysis_data[k], dim=1)
                    except Exception as e:
                         print(f"Error concatenating analysis data for {k}: {e}", file=sys.stderr)
                         # Decide how to handle: remove key, keep list, etc.
                         del analysis_data[k] # Example: remove problematic key
                elif not analysis_data[k]: # Remove empty keys
                    del analysis_data[k]
            return input_ids, analysis_data

        return input_ids


    # Modified to handle batch size and potential missing attention_mask/cache_position
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        batch_size: int # Pass batch size explicitly
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs and torch.is_tensor(model_kwargs["attention_mask"]):
            attention_mask = model_kwargs["attention_mask"]
            # Ensure the new ones match the batch size inferred from attention_mask
            current_batch_size = attention_mask.shape[0]
            new_mask = attention_mask.new_ones((current_batch_size, 1))
            model_kwargs["attention_mask"] = torch.cat([attention_mask, new_mask], dim=-1)

            # Update cache_position if present and used (common in recent HF)
            # It should reflect the new sequence length after adding one token
            # if "cache_position" in model_kwargs and torch.is_tensor(model_kwargs["cache_position"]):
            #      # Assuming cache_position tracks the index of the *next* token to be processed
            #      model_kwargs["cache_position"] = model_kwargs["attention_mask"].shape[1] -1 + model_kwargs["past_key_values"][0][0].shape[-2] +1 # Adjust based on how cache_position is exactly used by the model
            # Alternative if cache_position is just the length
            model_kwargs["cache_position"] = torch.tensor([model_kwargs["attention_mask"].shape[1]], device=outputs.logits.device)

        # If attention_mask wasn't initially present, we shouldn't add it here unless required.
        # The logic at the start of generate should handle initial mask creation.

        return model_kwargs


    # Unchanged from original
    def is_thinking_token(self, token):
        return token.strip() in ["Wait", "Alternatively", "Hmm"]
