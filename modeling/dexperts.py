import sys
import random
import os
import json
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
# TODO: remove these references 
# from transformers.generation.utils import (
#     ModelOutput,
#     top_k_top_p_filtering,
#     StoppingCriteriaList,
#     LogitsProcessorList
# )
from collections import defaultdict

# https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(_logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # TODO: convert next_token_logits dim to 1 then convert back
    assert _logits.shape[0] == 1 # assume greedy decoding
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = _logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits.unsqueeze(0)

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
        alpha: float = 1.0,
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

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.base.eval()
        if expert_model_name_or_path:
            self.expert = AutoModelForCausalLM.from_pretrained(
                expert_model_name_or_path, **model_kwargs
            )
            self.expert.eval()
        else:
            self.expert = None
        if antiexpert_model_name_or_path:
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
                antiexpert_model_name_or_path, **model_kwargs
            )
            self.antiexpert.eval()
        else:
            self.antiexpert = None


        self.tokenizer = tokenizer
        # Although the original 'alpha' constructor argument is retained, we  
        # will not use it verbatim for generation. Instead, alpha is controlled  
        # by the finite state machine described below.
        self.initial_alpha = alpha 
        # self.alpha = alpha
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = True if expert_model_name_or_path and 'chat' in expert_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

        # State machine variables  
        # The phases:  
        #   "S1_ZERO": alpha=0 => system 1 for 100 steps from the start or after counting down 
        #   "S2_ONE": alpha=1 => system 2 (wait for overriding event)  
        #   "S2_ONE_COUNTDOWN": alpha=1 => system 2 for 100 steps after overriding event  
        self.phase = "S1_ZERO"  
        self.phase_step_count = 0  # how many steps so far in the current phase  
        self.alpha = 1.0         # actual alpha in effect  
  
        # If provided, we will log JSON lines to this file  
        self.log_file = log_file  
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        self.alpha_strategy = alpha_strategy

  

    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        # print('DExperts expert prompt', flush=True)
        # print(chat_prompts[0], flush=True)
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def _update_phase(self, overriding_event_occurred: bool, extra_prompt_appended: bool):  
        """  
        State machine update logic:  
          - S1_ZERO: alpha=0 => system 1 for 100 steps; then go to S2_ONE  
          - S2_ONE: alpha=1 => system 2, wait for overriding event -> move to S2_ONE_COUNTDOWN  
          - S2_ONE_COUNTDOWN: alpha=1 => system 2 for 100 steps -> move to S1_ZERO  
          - S1_FINAL: alpha=0 => system 1 for 100 steps for answer generation -> end
        """  
        if extra_prompt_appended:
            self.phase = "S1_FINAL"
            self.phase_step_count = 0

        elif self.phase == "S1_ZERO":  
            # alpha=0  
            if self.phase_step_count >= 100:  
                self.phase = "S2_ONE"  
                self.phase_step_count = 0  
  
        elif self.phase == "S2_ONE":  
            # alpha=1  
            if overriding_event_occurred:  
                # start countdown  
                # self.phase = "S2_ONE_COUNTDOWN"  
                self.phase = "S1_ZERO"  
                self.phase_step_count = 0  
  
        elif self.phase == "S2_ONE_COUNTDOWN":  
            # alpha=1  
            if self.phase_step_count >= 100:  
                # after 100 steps, go to S1_ZERO  
                self.phase = "S1_ZERO"  
                self.phase_step_count = 0  
  
        # Now set alpha/mode based on current phase  
        if self.phase in ["S1_FINAL", "S1_ZERO"]:  
            self.alpha = 0  
        else:  
            self.alpha = 1  

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

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
        **kwargs
    ):
        """  
        Modified generate method to:  
          - dynamically toggle alpha between 0 and 1 based on a finite-state machine  
          - log JSON lines of top-10 tokens/logits each step to self.log_file  
        """  
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # prepare inputs for expert model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        # local helper to write JSON lines  
        def write_log(step: int, base_logits_1d: torch.Tensor,  
                      dexperts_logits_1d: Optional[torch.Tensor], alpha: float, phase: str):  
            """  
            Logs one dictionary entry to self.log_file (if not None).  
            base_logits_1d: shape [vocab_size]  
            dexperts_logits_1d: shape [vocab_size] or None  
            """  
            if base_logits_1d is None:  
                return  
            # top-10 from base model  
            base_top10_tokens, base_top10_vals = _get_topk_info(base_logits_1d, self.tokenizer, k=10)  
            if dexperts_logits_1d is not None:  
                dexperts_top10_tokens, dexperts_top10_vals = _get_topk_info(dexperts_logits_1d, self.tokenizer, k=10)  
            else:  
                dexperts_top10_tokens, dexperts_top10_vals = None, None  
  
            log_obj = {  
                "step": step,  
                "phase": phase,  
                "alpha": alpha,  
                "base_top10_tokens": base_top10_tokens,  
                "dexperts_top10_tokens": dexperts_top10_tokens,  
                "base_top10_logits": base_top10_vals,  
                "dexperts_top10_logits": dexperts_top10_vals,  
            }  
            if self.log_file is not None:  
                with open(self.log_file, "a", encoding="utf-8") as f:  
                    f.write(json.dumps(log_obj) + "\n")  


        # We'll use a while loop that runs until we've generated the allowed number of tokens.
        gen_steps = 0                   # Count of tokens generated by the model.
        allowed_gen_steps = max_new_tokens  # Initially allow max_new_tokens.
        extra_prompt_appended = False   # Flag to ensure extra prompt is added only once.

        while gen_steps < allowed_gen_steps:
            if extra_prompt_appended:
                self.alpha = 0
            else:
                if self.alpha_strategy is None or self.alpha_strategy == "constant":
                    self.alpha = 1
                elif self.alpha_strategy.startswith("cycle"):
                    # cycle100
                    T = int(self.alpha_strategy.replace("cycle", "")) 
                    self.alpha = (gen_steps // T) % 2
                elif self.alpha_strategy.startswith("2cycles"):
                    # 2cycles-400-100
                    items = self.alpha_strategy.split('-')
                    T0 = int(items[1])
                    T1 = int(items[2])
                    assert T0 % 100 == 0
                    assert T1 % 100 == 0
                    if (gen_steps // 100) % (T0 // 100 + T1 // 100) < (T0 // 100):
                        self.alpha = 0
                    else:
                        self.alpha = 1
                elif self.alpha_strategy.startswith("random"):
                    # random0.5
                    prob = float(self.alpha_strategy.replace("random", ""))
                    self.alpha = 1 if random.random() < prob else 0
                elif self.alpha_strategy == "override_annealing":
                    self._update_phase(overriding_event_occurred, extra_prompt_appended)  
                else:
                    raise ValueError(f"invalid alpha_strategy: {self.alpha_strategy}")

            # ------------------------------------------------  
            # 1) Prepare base inputs (always needed for forward)  
            # ------------------------------------------------  
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)  

            # ------------------------------------------------  
            # 2) Compute base logits  
            # ------------------------------------------------  
            base_outputs = self.base(**base_inputs, return_dict=True)  
            base_next_token_logits = base_outputs.logits[..., -1, :]  
  
            # prepare inputs for experts  
            # note: for anti-expert we use input_ids (not chat formatting)  
            if self.expert and self.antiexpert:
                expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)  
                antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)  
                # It won't save computation by skipping expert generation when alpha is zero,
                # because once alpha is changed to 1, we still need to compute key-value for all previously skipped tokens

                # forward pass for experts  
                expert_outputs = self.expert(**expert_inputs, return_dict=True)  
                antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=True)  
                expert_next_token_logits = expert_outputs.logits[..., -1, :]  
                antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]  

            # else:  
            # alpha=0 => no difference  
  
            # ------------------------------------------------  
            # 3) Determine final next_token_logits for sampling  
            # ------------------------------------------------  
            if self.alpha == 1 and self.expert and self.antiexpert:  

                vocab_size = self.tokenizer.vocab_size + 3 # include eos, user, assistant special token
                # Copy the base logits so that tokens beyond vocab_size remain unchanged.
                dexperts_next_token_logits = base_next_token_logits.clone()

                # Only update the first vocab_size logits.
                dexperts_next_token_logits[:, :vocab_size] = (
                    base_next_token_logits[:, :vocab_size] +
                    self.initial_alpha * (expert_next_token_logits[:, :vocab_size] - antiexpert_next_token_logits[:, :vocab_size])
                )

                next_token_logits = dexperts_next_token_logits 
            else:  
                dexperts_next_token_logits = None  
                next_token_logits = base_next_token_logits  # alpha=0 => use base only  
  
            # pre-process logits if needed  
            if logits_processor:  
                next_token_logits = logits_processor(input_ids, next_token_logits)  
  
            # apply temperature / top-p  
            if temperature != 1.0:  
                next_token_logits = next_token_logits / temperature  
            if top_p < 1.0:  
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)  
  
            # ------------------------------------------------  
            # 4) Sample or Greedy  
            # ------------------------------------------------  
            if do_sample:  
                probs = F.softmax(next_token_logits, dim=-1)  
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  
            else:  
                next_tokens = torch.argmax(next_token_logits, dim=-1)  
  
            # If a sequence is finished, set its next token to pad  
            next_tokens = (  
                next_tokens * unfinished_sequences  
                + self.tokenizer.pad_token_id * (1 - unfinished_sequences)  
            )  
  
            # ------------------------------------------------  
            # 5) Log the top-10 tokens for base & dexperts  
            #    (only use the first batch element to keep the logs simple)  
            # ------------------------------------------------  
            base_logits_1d = base_next_token_logits[0].detach()  
            if dexperts_next_token_logits is not None:  
                dexperts_logits_1d = dexperts_next_token_logits[0].detach()  
            else:  
                dexperts_logits_1d = None  
  
            write_log(  
                step=gen_steps,  
                base_logits_1d=base_logits_1d,  
                dexperts_logits_1d=dexperts_logits_1d,  
                alpha=self.alpha,  
                phase=self.phase,  
            )  

            if return_logits_for_analysis:  
                # gather step stats  
                next_token_logits_dict = {  
                    "dexperts": next_token_logits,  
                    "base": base_next_token_logits,  
                }  
                if self.expert and self.antiexpert:
                    next_token_logits_dict["expert"] = expert_next_token_logits  
                    next_token_logits_dict["antiexpert"] = antiexpert_next_token_logits  
  
                # store them  
                analysis_data["tokens"].append([self.tokenizer.decode(t) for t in next_tokens])  
                analysis_data["token_ids"].append(next_tokens)  
                for m_name, val in next_token_logits_dict.items():  
                    analysis_data[f"logits_{m_name}"].append(val.unsqueeze(dim=1))  

            # ------------------------------------------------  
            # 6) Update input_ids for next step  
            # ------------------------------------------------  
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  
            # also extend expert_input_ids  
            if self.use_chat_format_for_expert:  
                expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)  
            else:  
                expert_input_ids = input_ids  # keep them aligned if no chat formatting  
  
            # update model kwargs to include past / attention_mask  
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)  
            if self.expert and self.antiexpert:
                expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)  
                antiexpert_kwargs = self._update_model_kwargs_for_generation(  
                    antiexpert_outputs, antiexpert_kwargs  
                )  

            # ------------------------------------------------  
            # 7) Check stopping criteria  
            # ------------------------------------------------  
            if stopping_criteria and stopping_criteria(input_ids, None):  
                break  
  
            # if eos_token found, mark the sequence as finished  
            unfinished_sequences = unfinished_sequences.mul(  
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)  
                .ne(eos_token_id_tensor.unsqueeze(1))  
                .prod(dim=0)  
            )  
            if unfinished_sequences.max() == 0:  
                # all sequences finished  
                break  

            # ------------------------------------------------  
            # 8) Check if "overriding event" happened  
            #    (only if alpha=1 => system 2)  
            # ------------------------------------------------  
            overriding_event_occurred = False  
            if self.alpha == 1 and self.expert and self.antiexpert:  
                # compare base model's argmax vs. dexperts argmax on first example  
                base_top1 = torch.argmax(base_next_token_logits[0], dim=-1)  
                dexperts_top1 = torch.argmax(dexperts_next_token_logits[0], dim=-1)  
                if base_top1.item() != dexperts_top1.item():  
                    overriding_event_occurred = True  

            # ------------------------------------------------  
            # 9) force answer generation when max_generaiton_token limit is hit
            # ------------------------------------------------  
            gen_steps += 1
 
            # If we've generated exactly max_new_tokens and haven't added the extra prompt, append it.
            if gen_steps == max_new_tokens and not extra_prompt_appended:
                extra_prompt = "\nI'm not allowed to think more so I have to conclude that the final answer is:"
                extra_input = self.tokenizer(extra_prompt, return_tensors="pt").input_ids.to(input_ids.device)
                # Ensure the extra prompt is broadcasted to all sequences if needed.
                if extra_input.shape[0] == 1 and input_ids.shape[0] > 1:
                    extra_input = extra_input.expand(input_ids.shape[0], -1)
                input_ids = torch.cat([input_ids, extra_input], dim=-1)
                expert_input_ids = torch.cat([expert_input_ids, extra_input], dim=-1)

                # Also update attention masks if they exist.
                if "attention_mask" in base_kwargs:
                    extra_attention = torch.ones(extra_input.shape, device=input_ids.device, dtype=base_kwargs["attention_mask"].dtype)
                    base_kwargs["attention_mask"] = torch.cat([base_kwargs["attention_mask"], extra_attention], dim=-1)
                if self.expert and self.antiexpert:
                    if "attention_mask" in expert_kwargs:
                        extra_attention = torch.ones(extra_input.shape, device=input_ids.device, dtype=expert_kwargs["attention_mask"].dtype)
                        expert_kwargs["attention_mask"] = torch.cat([expert_kwargs["attention_mask"], extra_attention], dim=-1)
                    if "attention_mask" in antiexpert_kwargs:
                        extra_attention = torch.ones(extra_input.shape, device=input_ids.device, dtype=antiexpert_kwargs["attention_mask"].dtype)
                        antiexpert_kwargs["attention_mask"] = torch.cat([antiexpert_kwargs["attention_mask"], extra_attention], dim=-1)

                # Increase the allowed generation steps by 100.
                allowed_gen_steps += 100
                extra_prompt_appended = True

  
            # track how many steps in the current phase  
            self.phase_step_count += 1  



        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values
        # https://github.com/huggingface/transformers/issues/36151
        kwargs["cache_position"] = torch.tensor([kwargs["attention_mask"].shape[1]]).to(outputs.logits.device)

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs
