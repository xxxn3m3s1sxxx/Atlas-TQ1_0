#!/usr/bin/env python3
"""Atlas Packer v2 - Streaming TQ1.0 packer for Falcon3 BitNet"""
import struct, torch, numpy as np, json, os, sys
from safetensors import safe_open

BITNET_MUL = np.array([1, 3, 9, 27], dtype=np.uint32)
TQ1_MUL    = np.array([1, 3, 9, 27, 81], dtype=np.uint32)

def pack_tensor_row_wise(tensor):
    """BitNet uint8 [OUT/4, IN] → TQ1.0 [OUT, IN/4*5] row-major.
    
    Each uint8 byte packs 4 consecutive OUTPUT rows at one INPUT column.
    We de-interleave: uint8_row[ur] → W[4*ur+0,:], W[4*ur+1,:], W[4*ur+2,:], W[4*ur+3,:]
    Then repack each weight row separately: [W[r,0..4]]→1 byte, [W[r,5..9]]→1 byte, ...
    """
    arr = tensor.numpy()
    u8_rows, in_cols = arr.shape        # [OUT/4, IN]
    weight_cols = in_cols                # each col = 1 ternary per output row
    packed_per_row = (weight_cols + 4) // 5
    packed = np.empty(u8_rows * 4 * packed_per_row, dtype=np.uint8)

    for ur in range(u8_rows):
        row = arr[ur].astype(np.uint32)
        # Unpack 4 ternary values per byte (2-bit packing, NOT Base-3!)
        # byte = v0 + v1*4 + v2*16 + v3*64  where v_i ∈ {0,1,2,3}
        # v_i ∈ {0→-1, 1→0, 2→+1}; clamp 3→2 (should not appear for real data)
        t0 = np.minimum(row & 3, 2)
        t1 = np.minimum((row >> 2) & 3, 2)
        t2 = np.minimum((row >> 4) & 3, 2)
        t3 = np.minimum((row >> 6) & 3, 2)

        # De-interleave: each of 4 output rows gets all ternary values for that row
        # row_tern[0] = W[4*ur+0, :], row_tern[1] = W[4*ur+1, :], etc.
        for sub in range(4):
            wt = [t0, t1, t2, t3][sub]  # weight row sub, m ∈ {0,1,2}
            full_len = packed_per_row * 5
            if weight_cols < full_len:
                wt_pad = np.pad(wt, (0, full_len - weight_cols), constant_values=1)
            else:
                wt_pad = wt[:full_len]
            t5 = wt_pad.reshape(packed_per_row, 5)
            v = (t5 * TQ1_MUL).sum(axis=1)
            start = (ur * 4 + sub) * packed_per_row
            packed[start : start + packed_per_row] = np.minimum(v, 242).astype(np.uint8)

    return packed.tobytes(), packed_per_row

def create_atlas_from_config(safetensors_path, output_path):
    print(f"[ATLAS] Opening {safetensors_path}...")
    model_dir = os.path.dirname(safetensors_path)
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)

    hidden = cfg['hidden_size']
    n_layers = cfg['num_hidden_layers']
    n_heads = cfg['num_attention_heads']
    n_kv_heads = cfg['num_key_value_heads']
    inter = cfg['intermediate_size']
    vocab = cfg['vocab_size']
    head_dim = cfg.get('head_dim', hidden // n_heads)
    rope_theta = cfg.get('rope_theta', 10000.0)

    print(f"  Layers:{n_layers} Hidden:{hidden} Heads:{n_heads}/{n_kv_heads}")
    print(f"  Intermediate:{inter} Vocab:{vocab} Head_dim:{head_dim}")

    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        names = list(f.keys())

    print(f"  Tensors: {len(names)}")

    # Preload all weight_scale tensors (they're tiny: shape [1] bfloat16 each)
    scales = {}
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for n in names:
            if n.endswith('weight_scale'):
                scales[n] = f.get_tensor(n).item()

    print(f"  Scales loaded: {len(scales)}")

    with open(output_path, 'wb') as out:
        n_tensors = len(names)
        header = bytearray(64)
        header[0:5] = b'ATLAS'
        struct.pack_into('<H', header, 5, 4)  # v4: added tensor names
        struct.pack_into('<H', header, 7, n_layers)
        struct.pack_into('<H', header, 9, hidden)
        struct.pack_into('<H', header, 11, inter)
        struct.pack_into('<B', header, 13, n_heads)
        struct.pack_into('<B', header, 14, n_kv_heads)
        struct.pack_into('<H', header, 15, head_dim)
        struct.pack_into('<I', header, 17, vocab)
        struct.pack_into('<d', header, 21, rope_theta)
        struct.pack_into('<I', header, 60, n_tensors)

        # Build name block: [name_block_size:4] [name_0\0] [name_1\0] ...
        name_bytes = b''.join(n.encode() + b'\0' for n in names)
        name_block = struct.pack('<I', 4 + len(name_bytes)) + name_bytes
        struct.pack_into('<I', header, 56, len(name_block))

        out.write(header)

        data_start = 64 + len(names) * 12 + len(name_block)
        directory = bytearray(len(names) * 12)
        current_offset = data_start

        # Write name block after directory
        out.seek(64 + len(names) * 12)
        out.write(name_block)

        out.seek(data_start)

        with safe_open(safetensors_path, framework='pt', device='cpu') as f:
            for idx, name in enumerate(names):
                tensor = f.get_tensor(name)

                if tensor.dtype == torch.uint8:
                    raw_bytes, packed_per_row = pack_tensor_row_wise(tensor)
                    sname = name.replace('.weight', '.weight_scale')
                    scale_val = scales.get(sname, 1.0)
                    scale_bytes = struct.pack('<e', float(scale_val))
                    data_bytes = scale_bytes + raw_bytes
                else:
                    packed_per_row = 0
                    data_bytes = tensor.to(torch.float16).numpy().tobytes()

                if current_offset % 32 != 0:
                    pad = 32 - (current_offset % 32)
                    current_offset += pad
                    out.write(b'\x00' * pad)

                ttype = 0 if tensor.dtype == torch.uint8 else (1 if 'norm' in name or 'embed' in name else 2)
                row_dim = tensor.shape[0] * 4 if tensor.dtype == torch.uint8 else tensor.shape[0]

                # Entry: [ttype:1][offset:4][row_dim:4][packed_per_row:3] = 12 bytes
                directory[idx*12] = ttype
                struct.pack_into('<I', directory, idx*12 + 1, current_offset)
                struct.pack_into('<I', directory, idx*12 + 5, row_dim)
                ppr = packed_per_row & 0xFFFFFF
                directory[idx*12 + 9] = ppr & 0xFF
                directory[idx*12 + 10] = (ppr >> 8) & 0xFF
                directory[idx*12 + 11] = (ppr >> 16) & 0xFF

                out.write(data_bytes)
                current_offset += len(data_bytes)
                del tensor

                if idx % 16 == 0:
                    print(f"  [{idx}/{len(names)}] {name[:55]:55s} {len(data_bytes)/1024:7.1f}KB")

        out.seek(64)
        out.write(directory)

    total_gb = current_offset / 1024**3
    print(f"[ATLAS] Done! {total_gb:.2f} GB")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python atlas_packer.py <model.safetensors> <output.atlas>")
        sys.exit(1)
    create_atlas_from_config(sys.argv[1], sys.argv[2])
