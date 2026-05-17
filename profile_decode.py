#!/usr/bin/env python3
"""Profile a single decode step: C++ layers, Python RMSNorm, lm_head."""
import ctypes, os, time, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas_infer import AtlasModel, dll

def profile_decode_step(model, seq_len):
    B = 1
    n_layers = model.n_layers
    times = {}

    # Prewarm: page in all int8 data
    warmup = model._get_embedding(0)
    for l in range(n_layers):
        warmup = model.forward_layer(warmup, l, [seq_len])
    for _ in range(2):
        model.forward_layer(model._get_embedding(0), 0, [seq_len])

    n_runs = 3
    for run in range(n_runs):
        h = model._get_embedding(42)
        x_copy = h.copy()
        positions_arr = np.array([seq_len], dtype=np.int32)

        t0 = time.perf_counter()
        for layer in range(n_layers):
            idx_slice = model._layer_idx_arr[layer * 9 : (layer + 1) * 9].copy()
            kc = model.k_cache[layer].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            vc = model.v_cache[layer].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            dll.atlas_forward(
                model.model_ptr,
                x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                1, positions_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                kc, vc, model.max_seq_len, seq_len,
                idx_slice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 1)
        t_cpp = time.perf_counter() - t0
        times.setdefault("cpp_layers", []).append(t_cpp)

        t0 = time.perf_counter()
        h_norm = model._rmsnorm(x_copy.flatten(), "model.norm.weight")
        t_rms = time.perf_counter() - t0
        times.setdefault("rmsnorm", []).append(t_rms)

        t0 = time.perf_counter()
        logits = model._matmul_f16("lm_head.weight", h_norm.reshape(1, -1)).flatten()
        t_lm = time.perf_counter() - t0
        times.setdefault("lm_head", []).append(t_lm)

    for k in times:
        times[k] = np.mean(times[k])
    return times

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python profile_decode.py <atlas.tq1> <model_dir>")
        sys.exit(1)
    atlas_path = sys.argv[1]
    model_dir = sys.argv[2]
    safe_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading {atlas_path}...")
    t0 = time.perf_counter()
    model = AtlasModel(atlas_path, safe_path)
    print(f"Load: {time.perf_counter()-t0:.1f}s")

    print("\nProfiling decode step (seq_len=10)...")
    times = profile_decode_step(model, 10)

    cpp = times["cpp_layers"]
    rms = times["rmsnorm"]
    lm = times["lm_head"]
    total = cpp + rms + lm

    print(f"\n{'Component':<25} {'Time (ms)':<12} {'Share':<10}")
    print("-" * 47)
    print(f"{'C++ layers (all)':<25} {cpp*1000:<12.1f} {cpp/total*100:<9.1f}%")
    print(f"{'  per-layer mean':<25} {cpp*1000/model.n_layers:<12.1f}")
    print(f"{'Python RMSNorm':<25} {rms*1000:<12.1f} {rms/total*100:<9.1f}%")
    print(f"{'Python lm_head':<25} {lm*1000:<12.1f} {lm/total*100:<9.1f}%")
    print(f"{'Total decode step':<25} {total*1000:<12.1f} 100.0%")
    print(f"\nDecode rate: {1/total:.1f} tok/s")
