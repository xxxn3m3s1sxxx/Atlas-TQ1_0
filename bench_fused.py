#!/usr/bin/env python3
"""Benchmark fused vs per-layer 10B decode speed."""
import ctypes, os, time, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_infer import AtlasModel, dll

def run_decode(model, input_ids, eos_id, n_gen, mode="fused"):
    seq_len = len(input_ids)
    full_logits = model.forward(input_ids[None])
    next_token = int(np.argmax(full_logits[0, -1, :]))
    output = [next_token]
    n_steps = 0
    t0 = time.perf_counter()
    for step in range(n_gen):
        if next_token == eos_id: break
        h = model._get_embedding(next_token)
        if mode == "fused":
            pos = np.array([seq_len + step], dtype=np.int32)
            dll.atlas_forward(
                model.model_ptr,
                h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                1, pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                model.k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                model.v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                model.max_seq_len, seq_len + step,
                model._layer_idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                model.n_layers)
        else:
            for layer in range(model.n_layers):
                h = model.forward_layer(h, layer, [seq_len + step])
        h_norm = model._rmsnorm(h.flatten(), "model.norm.weight")
        logits = model._matmul_f16("lm_head.weight", h_norm.reshape(1, -1)).flatten()
        next_token = int(np.argmax(logits))
        output.append(next_token)
        n_steps += 1
    dt = time.perf_counter() - t0
    return n_steps, dt

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python bench_fused.py <atlas.tq1> <model_dir>")
        sys.exit(1)
    from transformers import AutoTokenizer
    atlas_path, model_dir = sys.argv[1], sys.argv[2]
    safe_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading {atlas_path}...")
    m = AtlasModel(atlas_path, safe_path)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    eos = tok.eos_token_id
    prompt = "What is the capital of France?"
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text, return_tensors='np')[0]
    seq_len = len(input_ids)
    print(f"Prompt: {seq_len} tok + 10 gen\n")

    for label, mode in [("FUSED  (all layers, single C call)", "fused"),
                         ("PER-LAYER  (forward_layer each)", "per-layer")]:
        print(f"  {label}")
        m.k_cache[:] = 0; m.v_cache[:] = 0
        runs = []
        for _ in range(3):
            n, t = run_decode(m, input_ids, eos, 10, mode)
            runs.append(t / n * 1000)
        avg = np.mean(runs)
        print(f"    {'  '.join(f'{r:.0f}' for r in runs)}  ms/tok  (mean {avg:.0f} ms, {1000/avg:.1f} tok/s)\n")
