"""Verify: pure Python with correct row ordering vs reference."""
import torch, sys, numpy as np, pickle
from safetensors import safe_open

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python test_row_order.py <atlas.tq1> <model.safetensors> <ref_activations.pkl>")
        sys.exit(1)

safetensors_path = sys.argv[2]
ref_path = sys.argv[3]

with open(ref_path, 'rb') as f:
    ref_data = pickle.load(f)
act = ref_data['activations']
ref_pn = act['L0_post_norm_in'][0]
ref_gate = act['L0_gate_out'][0]

with safe_open(safetensors_path,
               framework='pt', device='cpu') as f:
    w_u8 = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy()
    weight_scale = f.get_tensor("model.layers.0.mlp.gate_proj.weight_scale").item()

max_abs = max(np.max(np.abs(ref_pn)), 1e-5)
input_scale = 127.0 / max_abs
x_q = np.round(ref_pn * input_scale).clip(-128, 127).astype(np.float32)

rows_packed = w_u8.shape[0]  # 5760
out_atlas = np.zeros(rows_packed * 4, dtype=np.float32)
out_ref_order = np.zeros(rows_packed * 4, dtype=np.float32)

for ur in range(rows_packed):
    row = w_u8[ur].astype(np.uint32)
    for q in range(4):
        ternary = ((row >> (2*q)) & 3).astype(np.float32) - 1.0
        raw = np.dot(x_q, ternary)
        out_atlas[ur * 4 + q] = raw
        out_ref_order[q * rows_packed + ur] = raw

# Dequantize
out_atlas *= max_abs / (127.0 * weight_scale)
out_ref_order *= max_abs / (127.0 * weight_scale)

c_bad = np.corrcoef(out_atlas, ref_gate)[0, 1]
c_good = np.corrcoef(out_ref_order, ref_gate)[0, 1]
r_bad = float(np.sqrt(np.mean(out_atlas**2))) / float(np.sqrt(np.mean(ref_gate**2)))
r_good = float(np.sqrt(np.mean(out_ref_order**2))) / float(np.sqrt(np.mean(ref_gate**2)))

print(f"ATLAS order: corr={c_bad:.6f}  RMS ratio={r_bad:.6f}")
print(f"HF    order: corr={c_good:.6f}  RMS ratio={r_good:.6f}")
print(f"First 10 atlas: {out_atlas[:10].round(4)}")
print(f"First 10 ref:   {ref_gate[:10].round(4)}")
print(f"First 10 hf:    {out_ref_order[:10].round(4)}")
