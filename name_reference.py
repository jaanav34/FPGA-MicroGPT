import torch
import torch.nn.functional as F

def generate_golden_reference(state_dict, uchars, start_char='b', max_len=16):
    """
    Simulates the FPGA's greedy decoding to verify hardware correctness.
    """
    # 1. Setup mapping
    stoi = {ch: i for i, ch in enumerate(uchars)}
    itos = {i: ch for i, ch in enumerate(uchars)}
    vocab_size = len(uchars) + 1 # +1 for BOS
    bos_id = vocab_size - 1
    
    # 2. Convert state_dict to tensors
    wte = torch.tensor(state_dict['wte'])
    wpe = torch.tensor(state_dict['wpe'])
    lm_head = torch.tensor(state_dict['lm_head'])
    
    # Start with the user-provided character or BOS
    current_token = stoi.get(start_char, bos_id)
    generated_ids = []
    
    print(f"--- Starting Python Golden Reference (Start: '{start_char}') ---")
    
    for pos in range(max_len):
        # --- Simulating microgpt_top.sv Logic ---
        # tok_emb + pos_emb
        x = wte[current_token] + wpe[pos]
        
        # Note: This is a simplified forward pass for verification
        # In a real check, you'd run the full TransformerLayer forward here
        # logits = model(current_token_tensor) 
        
        # Simplified: applies only the LM head projection, not the full transformer stack
        logits = x @ lm_head.t()
        
        # 3. Greedy Argmax (Matches TOP_ARGMAX in SV)
        next_token = torch.argmax(logits).item()
        
        if next_token == bos_id:
            print(f"  [Pos {pos}] Predicted: BOS (Stopping)")
            break
            
        char = itos.get(next_token, '?')
        generated_ids.append(next_token)
        print(f"  [Pos {pos}] Predicted Token ID: {next_token} (char: {char})")
        
        # Advance
        current_token = next_token

    return generated_ids

# Example usage:
generate_golden_reference(trained_state_dict, uchars, start_char='b')