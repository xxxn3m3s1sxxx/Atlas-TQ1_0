"""Compare TQ1 kernel output vs pure Python for ONE output row."""
import ctypes, struct, numpy as np, pickle, os
from safetensors import safe_open
os.environ['PYTHONIOENCODING'] = 'utf-8'

with open(r'C:\atlas\ref_activations.pkl', 'rb') as f:
    ref_data = pickle.load(f)
act = ref_data['activations']
ref_pn = act['L0_post_norm_in'][0]  # [3072]
ref_gate = act['L0_gate_out'][0]    # [23040], reference output

# Load weights from safetensors (small: just unpack third of the rows)
with safe_open(r'C:\models\Falcon3-7B-Instruct-1.58bit\model.safetensors',
               framework='pt', device='cpu') as f:
    w_u8 = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy()
    weight_scale = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale").item()

print(f"weight_scale={weight_scale:.4f}")

# Unpack just 1 packed row → 4 actual rows
ur_test = 0
packed_row = w_u8[ur_test:ur_test+1]  # (1, 3072) = 4 actual rows
t0 = (packed_row & 3).astype(np.float32) - 1.0  # row 0
t1 = ((packed_row >> 2) & 3).astype(np.float32) - 1.0  # row 1
t2 = ((packed_row >> 4) & 3).astype(np.float32) - 1.0  # row 2
t3 = ((packed_row >> 6) & 3).astype(np.float32) - 1.0  # row 3

print(f"Unpacked 4 rows from packed row {ur_test}")

# Pure Python forward for these 4 rows
def pure_forward(x, w_row):
    """x: [3072], w_row: [3072] → scalar output"""
    max_abs = max(np.max(np.abs(x)), 1e-5)
    input_scale = 127.0 / max_abs
    x_q = np.round(x * input_scale).clip(-128, 127)
    y = np.dot(x_q, w_row)  # scalar
    return y / (input_scale * weight_scale)

out_py = np.array([pure_forward(ref_pn, t0[0]), pure_forward(ref_pn, t1[0]),
                   pure_forward(ref_pn, t2[0]), pure_forward(ref_pn, t3[0])])
print(f"Pure Python out: {out_py}")

# Now via TQ1 kernel (need atlas model, avoid full load)
# Instead: decode TQ1 bytes from atlas and compute dot product manually
with open(r'C:\atlas\falcon3-7b-tq1.atlas', 'rb') as f:
    f.seek(64)
    dir_data = f.read(451 * 12)
    idx = 5  # gate_proj index
    entry = dir_data[idx*12 : (idx+1)*12]
    offset = struct.unpack('<I', entry[1:5])[0]
    row_dim = struct.unpack('<I', entry[5:9])[0]
    ppr = entry[9] | (entry[10] << 8) | (entry[11] << 16)
    
    print(f"Atlas: offset={offset} rows={row_dim} ppr={ppr}")
    
    # Read scale + row 0 data
    f.seek(offset)
    scale_raw = struct.unpack('<H', f.read(2))[0]
    scale = np.frombuffer(struct.pack('<H', scale_raw), dtype=np.float16)[0].item()
    print(f"Scale: {scale:.4f} (weight_scale from safetensors: {weight_scale:.4f})")
    
    # Read row 0 TQ1 data
    row0_bytes = np.frombuffer(f.read(ppr), dtype=np.uint8)
    
    # Decode TQ1 row 0: each byte → 5 ternary values
    w_tq1_row0 = np.zeros(3075, dtype=np.float32)  # full padded length
    for bi, b in enumerate(row0_bytes):
        tmp = int(b)
        for k in range(5):
            m = tmp % 3
            w_tq1_row0[bi*5 + k] = -1.0 if m == 0 else (1.0 if m == 2 else 0.0)
            tmp //= 3
    w_tq1_row0 = w_tq1_row0[:3072]  # remove padding
    
    # Compare TQ1 row 0 with HF row 0
    match = np.allclose(w_tq1_row0, t0[0])
    print(f"TQ1 row 0 matches HF: {match}")

# TQ1 forward (manual, matching kernel exactly)
def tq1_forward(x, w_tq1_row):
    """Simulate TQ1 kernel exactly."""
    max_abs = max(np.max(np.abs(x)), 1e-5)
    input_scale = 127.0 / max_abs
    x_q = np.round(x * input_scale).clip(-128, 127)
    raw_sum = np.dot(x_q, w_tq1_row)
    # Kernel dequant: out *= max_abs / (127 * scale)
    return raw_sum * max_abs / (127.0 * scale)

out_tq1 = np.array([tq1_forward(ref_pn, w_tq1_row0)])
print(f"TQ1 forward out: {out_tq1[0]:.6f}")
print(f"Ref gate[0]: {float(ref_gate[0]):.6f}")

# Compare
error = abs(out_tq1[0] - ref_gate[0])
print(f"Error (TQ1 vs Ref): {error:.6f} (rel: {error/abs(ref_gate[0]):.6f})")

# Pure Python again for comparison
out_py2 = np.array([pure_forward(ref_pn, t0[0])])
print(f"Pure out[0]: {out_py2[0]:.6f}")
error_py = abs(out_py2[0] - ref_gate[0])
print(f"Error (Py vs Ref): {error_py:.6f} (rel: {error_py/abs(ref_gate[0]):.6f})")
