"""Directly compare atlas ternary values vs HF reference (no atlas model load)."""
import struct, sys, numpy as np
from safetensors import safe_open

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_direct_cmp.py <atlas.tq1> <model.safetensors>")
        sys.exit(1)

atlas_path = sys.argv[1]
safetensors_path = sys.argv[2]

# ─── Read atlas directory ───────────────────────────────────────────────
with open(atlas_path, 'rb') as f:
    magic = f.read(5)
    version = struct.unpack('<H', f.read(2))[0]
    n_layers = struct.unpack('<H', f.read(2))[0]
    hidden = struct.unpack('<H', f.read(2))[0]
    inter = struct.unpack('<H', f.read(2))[0]
    n_heads = f.read(1)[0]
    n_kv = f.read(1)[0]
    head_dim = struct.unpack('<H', f.read(2))[0]
    vocab = struct.unpack('<I', f.read(4))[0]
    print(f"Atlas: {n_layers}L {hidden}H {inter}I {vocab}vocab")

    f.seek(60)
    n_tensors = struct.unpack('<I', f.read(4))[0]
    print(f"Tensors: {n_tensors}")

    # Read directory at offset 64
    f.seek(64)
    dir_data = f.read(n_tensors * 12)
    
    # Build name→entry mapping from atlas_infer.py tensor order
    # We need tensor names to match indices
    # Let's first just get the gate_proj entry by looking at all entries
    with safe_open(safetensors_path,
                   framework='pt', device='cpu') as sf:
        names = list(sf.keys())
    
    # Find gate_proj index
    gp_name = "model.layers.0.mlp.gate_proj.weight"
    if gp_name in names:
        idx = names.index(gp_name)
        print(f"gate_proj index: {idx}")
        entry = dir_data[idx*12 : (idx+1)*12]
        ttype = entry[0]
        offset = struct.unpack('<I', entry[1:5])[0]
        row_dim = struct.unpack('<I', entry[5:9])[0]
        ppr_bytes = bytes([entry[9], entry[10], entry[11], 0])
        packed_per_row = struct.unpack('<I', ppr_bytes)[0]
        print(f"  ttype={ttype} offset={offset} rows={row_dim} ppr={packed_per_row}")
        
        # Read the tensor data
        f.seek(offset)
        scale_raw = struct.unpack('<H', f.read(2))[0]
        scale = np.frombuffer(struct.pack('<H', scale_raw), dtype=np.float16)[0].item()
        print(f"  scale={scale:.4f}")
        
        # Read first TQ1 byte of row 0
        packed = np.frombuffer(f.read(packed_per_row), dtype=np.uint8)
        b0 = int(packed[0])
        b1 = int(packed[1])
        print(f"  TQ1 byte 0: {b0} ({bin(b0)})")
        print(f"  TQ1 byte 1: {b1} ({bin(b1)})")

# ─── Read HF reference ─────────────────────────────────────────────────
with safe_open(safetensors_path,
               framework='pt', device='cpu') as f:
    w_u8 = f.get_tensor(gp_name).numpy()

# HF manual unpack (avoid torch)
packed = w_u8  # (5760, 3072)
out_rows = packed.shape[0] * 4
unpacked = np.zeros((out_rows, packed.shape[1]), dtype=np.float32)
for i in range(4):
    start = i * packed.shape[0]
    end = (i + 1) * packed.shape[0]
    unpacked[start:end] = ((packed >> (2*i)) & 3).astype(np.float32) - 1.0

# Row 0, first 10 values
hf_row0_first10 = unpacked[0, :10].astype(int).tolist()
print(f"\nHF row 0[:10]: {hf_row0_first10}")

# Decode TQ1 bytes
def decode_tq1(byte_val):
    vals = []
    tmp = int(byte_val)
    for _ in range(5):
        m = tmp % 3
        vals.append(m - 1)  # 0→-1, 1→0, 2→+1
        tmp //= 3
    return vals

tq1_b0_vals = decode_tq1(b0)
tq1_b1_vals = decode_tq1(b1)
print(f"TQ1 byte 0: {tq1_b0_vals} (cols 0-4)")
print(f"TQ1 byte 1: {tq1_b1_vals} (cols 5-9)")
print(f"HF row0[0:5]:  {hf_row0_first10[:5]}")
print(f"HF row0[5:10]: {hf_row0_first10[5:]}")
print(f"Byte 0 match: {tq1_b0_vals == hf_row0_first10[:5]}")
print(f"Byte 1 match: {tq1_b1_vals == hf_row0_first10[5:]}")

# Also check row 1 (next 4 actual rows)
print(f"\nHF row 1[:10]: {unpacked[1, :10].astype(int).tolist()}")
print(f"HF row 2[:10]: {unpacked[2, :10].astype(int).tolist()}")
print(f"HF row 3[:10]: {unpacked[3, :10].astype(int).tolist()}")
