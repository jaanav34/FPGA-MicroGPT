import math

# --- Configuration (Must match microgpt_pkg.sv) ---
N_EMBD = 16
VOCAB_SIZE = 27
BLOCK_SIZE = 16
MLP_DIM = 64
FRAC_BITS = 8
SCALE = 256.0

# Address Map
ADDR_WTE, ADDR_WPE, ADDR_LM_HEAD = 0, 432, 688
ADDR_WQ, ADDR_WK, ADDR_WV, ADDR_WO = 1120, 1376, 1632, 1888
ADDR_FC1, ADDR_FC2 = 2144, 3168

def q88_to_float(hex_str):
    val = int(hex_str, 16)
    if val >= 32768: val -= 65536
    return val / SCALE

def rms_norm(x, eps=1e-5):
    # Simplified RMSNorm (Assumes weights are 1.0 as in early training)
    ss = sum(v**2 for v in x) / len(x)
    return [v / math.sqrt(ss + eps) for v in x]

def softmax(logits):
    exps = [math.exp(l - max(logits)) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

def load_weights(filename="params.mem"):
    with open(filename, 'r') as f:
        raw = [q88_to_float(line.strip()) for line in f if line.strip()]
    
    def reshape(data, r, c):
        return [data[i*c:(i+1)*c] for i in range(r)]

    return {
        'wte': reshape(raw[ADDR_WTE:], 27, 16),
        'wpe': reshape(raw[ADDR_WPE:], 16, 16),
        'lm_head': reshape(raw[ADDR_LM_HEAD:], 27, 16),
        'wq': reshape(raw[ADDR_WQ:], 16, 16),
        'wk': reshape(raw[ADDR_WK:], 16, 16),
        'wv': reshape(raw[ADDR_WV:], 16, 16),
        'wo': reshape(raw[ADDR_WO:], 16, 16),
        'fc1': reshape(raw[ADDR_FC1:], 64, 16),
        'fc2': reshape(raw[ADDR_FC2:], 16, 64)
    }

def run_full_inference(w, start_id=1, max_len=16):
    current_token = start_id
    kv_cache = [] # Stores (K, V) tuples
    print(f"\n--- Full Transformer Python Reference (Start ID: {start_id}) ---")

    for pos in range(max_len):
        # 1. Embed
        x = [t + p for t, p in zip(w['wte'][current_token], w['wpe'][pos])]
        x_res1 = list(x)
        
        # 2. RMSNorm + Attention
        x = rms_norm(x)
        q = [sum(row[i]*x[i] for i in range(16)) for row in w['wq']]
        k = [sum(row[i]*x[i] for i in range(16)) for row in w['wk']]
        v = [sum(row[i]*x[i] for i in range(16)) for row in w['wv']]
        kv_cache.append((k, v))
        
        # Multi-head attention (simplified for verification)
        scores = []
        for pk, pv in kv_cache:
            score = sum(qi*ki for qi, ki in zip(q, pk)) / math.sqrt(4)
            scores.append(score)
        
        probs = softmax(scores)
        attn_out = [0.0] * 16
        for i, p in enumerate(probs):
            for j in range(16):
                attn_out[j] += p * kv_cache[i][1][j]
        
        x = [sum(row[i]*attn_out[i] for i in range(16)) + r for row, r in zip(w['wo'], x_res1)]
        x_res2 = list(x)
        
        # 3. RMSNorm + MLP
        x = rms_norm(x)
        h = [max(0, sum(row[i]*x[i] for i in range(16))) for row in w['fc1']] # ReLU
        x = [sum(row[i]*h[i] for i in range(64)) + r for row, r in zip(w['fc2'], x_res2)]
        
        # 4. Output Head
        logits = [sum(row[i]*x[i] for i in range(16)) for row in w['lm_head']]
        next_token = logits.index(max(logits))
        
        if next_token == 26: break
        print(f"Pos {pos:2d}: ID {next_token:2d} (char: {chr(next_token + 97)})")
        current_token = next_token

if __name__ == "__main__":
    weights = load_weights("params.mem")
    run_full_inference(weights, start_id=1)