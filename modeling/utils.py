import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits: torch.Tensor,
                          top_k: int = 0,
                          top_p: float = 0.0,
                          filter_value: float = -float('Inf'),
                          min_tokens_to_keep: int = 1) -> torch.Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                         Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            filter_value: value to assign to filtered logits.
            min_tokens_to_keep: minimum number of tokens we must keep (default 1).

        Returns:
            Logits with filtered elements set to filter_value.
            Shape: (batch size, vocabulary size)
    """
    if top_k == 0 and top_p == 0.0:
        return logits # No filtering needed

    batch_size, vocab_size = logits.size()

    if top_k > 0:
        # Safety check: ensure top_k is not larger than vocab size
        top_k = min(max(top_k, min_tokens_to_keep), vocab_size)
        # Keep at least min_tokens_to_keep (default 1) tokens
        # top_k = max(top_k, min_tokens_to_keep)

        # Find the top_k values and their indices for each batch item
        # We need the k-th value to threshold against
        # topk returns (values, indices)
        # values has shape (batch_size, top_k)
        topk_values, _ = torch.topk(logits, top_k, dim=-1)

        # Get the k-th value for each batch item (shape: batch_size, 1)
        # The [..., -1, None] keeps the dimension for broadcasting
        kth_value = topk_values[..., -1, None]

        # Create a mask for values less than the k-th value
        indices_to_remove = logits < kth_value
        # Apply the filter
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (nucleus filtering)
        # Create a mask for values where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Ensure we keep at least min_tokens_to_keep tokens
        if min_tokens_to_keep > 1:
            # Set the first min_tokens_to_keep indices to False (i.e., keep them)
             sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        else:
            # Shift the indices to the right to keep also the first token above the threshold
            # Important: The shift ensures that the token that *crosses* the threshold is kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            # Always keep the highest probability token (the first one)
            sorted_indices_to_remove[..., 0] = False

        # Create a final mask for the original logits tensor
        # Initialize a mask of False with the same shape as logits
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        # Use scatter_ to place True values in indices_to_remove at the locations
        # specified by sorted_indices corresponding to where sorted_indices_to_remove is True
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        # Apply the filter
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

# Example Usage:
if __name__ == '__main__':
    # Example logits (batch_size=2, vocab_size=5)
    example_logits = torch.tensor([
        [0.1, 0.5, 0.2, 0.9, 0.3],  # Batch item 1
        [0.8, 0.1, 0.7, 0.2, 0.6]   # Batch item 2
    ], dtype=torch.float32)

    print("Original Logits:\n", example_logits)

    # --- Test Top-K ---
    top_k_val = 2
    filtered_logits_k = top_k_top_p_filtering(example_logits.clone(), top_k=top_k_val, top_p=0.0)
    print(f"\nFiltered Logits (Top-K={top_k_val}):\n", filtered_logits_k)
    # Expected Item 1: Keep 0.9 (idx 3) and 0.5 (idx 1) -> [-inf, 0.5, -inf, 0.9, -inf]
    # Expected Item 2: Keep 0.8 (idx 0) and 0.7 (idx 2) -> [0.8, -inf, 0.7, -inf, -inf]

    # --- Test Top-P ---
    top_p_val = 0.8
    filtered_logits_p = top_k_top_p_filtering(example_logits.clone(), top_k=0, top_p=top_p_val)
    print(f"\nFiltered Logits (Top-P={top_p_val}):\n", filtered_logits_p)
    # Expected Item 1:
    #   Sorted: [0.9, 0.5, 0.3, 0.2, 0.1]
    #   Softmax approx: [0.44, 0.27, 0.16, 0.14, 0.12]
    #   Cumsum approx: [0.44, 0.71, 0.87, 1.01, 1.13]
    #   Keep indices where cumsum <= 0.8 (after shift): indices 3, 1 (values 0.9, 0.5)
    #   Result: [-inf, 0.5, -inf, 0.9, -inf]
    # Expected Item 2:
    #   Sorted: [0.8, 0.7, 0.6, 0.2, 0.1]
    #   Softmax approx: [0.37, 0.33, 0.30, 0.12, 0.11]
    #   Cumsum approx: [0.37, 0.70, 1.00, 1.12, 1.23]
    #   Keep indices where cumsum <= 0.8 (after shift): indices 0, 2 (values 0.8, 0.7) - Note: 0.6 crosses the threshold
    #   Result: [0.8, -inf, 0.7, -inf, 0.6] -> Wait, check shift logic. Shift keeps the one *crossing* the threshold. So it should keep index 4 (value 0.6) as well.
    #   Revised Expected Item 2: Keep indices 0, 2, 4 (values 0.8, 0.7, 0.6) -> [0.8, -inf, 0.7, -inf, 0.6]

    # --- Test Combined ---
    top_k_comb = 4
    top_p_comb = 0.75
    filtered_logits_combined = top_k_top_p_filtering(example_logits.clone(), top_k=top_k_comb, top_p=top_p_comb)
    print(f"\nFiltered Logits (Top-K={top_k_comb}, Top-P={top_p_comb}):\n", filtered_logits_combined)
    # Expected Item 1 (top_k=4): Keeps 0.9, 0.5, 0.3, 0.2 -> [-inf, 0.5, 0.2, 0.9, 0.3]
    # Expected Item 1 (top_p=0.75 applied after top_k):
    #   Logits: [-inf, 0.5, 0.2, 0.9, 0.3]
    #   Sorted: [0.9, 0.5, 0.3, 0.2, -inf]
    #   Softmax approx (on kept): [0.44, 0.27, 0.16, 0.14] -> Softmax([-inf, 0.5, 0.2, 0.9, 0.3])
    #   Softmax([0.9, 0.5, 0.3, 0.2]) approx [0.44, 0.27, 0.16, 0.14]
    #   Softmax([-inf, 0.5, 0.2, 0.9, 0.3]) approx [0, 0.26, 0.13, 0.43, 0.16] (recalc needed)
    # Let's recalculate softmax for [-inf, 0.5, 0.2, 0.9, 0.3]: Values are [1.65, 1.22, 2.46, 1.35]. Sum=6.68. Probs approx [0.25, 0.18, 0.37, 0.20] for indices [1, 2, 3, 4].
    # Sorted logits: [0.9, 0.5, 0.3, 0.2, -inf] -> indices [3, 1, 4, 2, 0]
    # Sorted Probs: [0.37, 0.25, 0.20, 0.18, 0]
    # Cumsum: [0.37, 0.62, 0.82, 1.00, 1.00]
    # Keep where <= 0.75 (after shift): Keep first two -> indices 3, 1 (Values 0.9, 0.5)
    # Result Item 1: [-inf, 0.5, -inf, 0.9, -inf]

    # Expected Item 2 (top_k=4): Keeps 0.8, 0.7, 0.6, 0.2 -> [0.8, -inf, 0.7, 0.2, 0.6]
    # Expected Item 2 (top_p=0.75 applied after top_k):
    #   Logits: [0.8, -inf, 0.7, 0.2, 0.6]
    #   Softmax([0.8, -inf, 0.7, 0.2, 0.6]): Values [2.23, 0, 2.01, 1.22, 1.82]. Sum=7.28. Probs approx [0.31, 0, 0.28, 0.17, 0.25] for indices [0, 1, 2, 3, 4]
    #   Sorted logits: [0.8, 0.7, 0.6, 0.2, -inf] -> indices [0, 2, 4, 3, 1]
    #   Sorted Probs: [0.31, 0.28, 0.25, 0.17, 0]
    #   Cumsum: [0.31, 0.59, 0.84, 1.00, 1.00]
    #   Keep where <= 0.75 (after shift): Keep first two -> indices 0, 2 (Values 0.8, 0.7)
    #   Result Item 2: [0.8, -inf, 0.7, -inf, -inf]

    # --- Test min_tokens_to_keep ---
    min_tokens = 3
    filtered_logits_min = top_k_top_p_filtering(example_logits.clone(), top_k=1, top_p=0.1, min_tokens_to_keep=min_tokens)
    print(f"\nFiltered Logits (Top-K=1, Top-P=0.1, min_tokens={min_tokens}):\n", filtered_logits_min)
    # Expected Item 1: top_k=1 keeps 0.9. top_p=0.1 keeps 0.9. min_tokens=3 forces keeping top 3: 0.9, 0.5, 0.3
    #   Result Item 1: [-inf, 0.5, -inf, 0.9, 0.3]
    # Expected Item 2: top_k=1 keeps 0.8. top_p=0.1 keeps 0.8. min_tokens=3 forces keeping top 3: 0.8, 0.7, 0.6
    #   Result Item 2: [0.8, -inf, 0.7, -inf, 0.6]
