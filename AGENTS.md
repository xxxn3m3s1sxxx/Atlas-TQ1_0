# ATLAS — Falcon3 TQ1.0 Inference Engine

## Goal
- TQ1.0 (Base-3, 5 ternary trits/byte) inference engine for Falcon3-10B on i5 laptop, end-to-end text generation via C++ DLL + Python.

## Constraints & Preferences
- 16 GB RAM only (no GPU); Falcon3-10B atlas file = 3.35 GB.
- Target CPU: i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA).
- Compiler: Clang 22.1.5 (LLVM MinGW x86_64-w64-windows-gnu) from WinGet.
- OpenMP: dynamic link via `libomp.dll` (shipped with LLVM-MinGW). Set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid conflict with numpy MKL's `libiomp5md.dll`.
- Base-3 encoding and matmul kernels as before.
- Same model architecture (1000042 theta, GQA 12/4 heads, etc).
- **10B model only** on current machine (40 layers). No 7B safetensors available — only a GGUF variant exists.

## Progress
### Done
- Packed Falcon3-10B into `falcon3-10b-tq1.atlas` (3.35 GB, 643 tensors).
- Core C++ DLL: load, matmul, rmsnorm, rope. Scalar LUT kernel, 4-way unrolled.
- Python inference: full GQA attention (einsum + causal mask), RoPE, KV cache, SiLU FFN, autoregressive loop.
- **fseek 32-bit overflow fix** (`_fseeki64` via FSEEK macro).
- **2-bit packing fix** (was decoding as Base-3 instead of `& 3`, `>> 2`).
- **Row ordering fix** (interleaved `ur*4+q` → stride `q*rows_packed+ur`). This was the root cause of corr≈0.
- All L0 TQ1 projections verified corr > 0.999 with HF reference.
- **OpenMP enabled**: `-fopenmp` in compile.bat, rebuilt `atlas.dll`.
  - `down_proj` matmul: 31ms → 5.8ms (5.4× speedup)
  - Decode: 0.3 → 1.0 tok/s on Falcon3-10B (3.3×)
  - "What is 2+2?": 97s → 23s
- `KMP_DUPLICATE_LIB_OK=TRUE` set in `atlas_infer.py` for numpy MKL compat.
- README paths fixed to relative/portable.
- `atlas_*.dll` temp builds gitignored.

### Open
- **7B model**: safetensors not present on this machine (`Falcon3-7B-Instruct-1.58bit-GGUF` is GGUF only). Would need download/re-pack for 7B testing.
- **Performance ceiling**: Python overhead (attention einsum, per-layer loop) adds ~30% to per-token time vs pure C++.
- **Prefill is slow**: 0.3 tok/s (same as decode) — no batching optimization.

### Blocked
- (none)

## Key Decisions
- **Scalar over AVX2 gather**: Gather was 4× slower (40 vs 162 tok/s) on Alder Lake.
- **Dynamic OpenMP**: `libomp.dll` at runtime (no static libomp.a in LLVM-MinGW).
- **5 separate LUT arrays** for register-based per-position access.
- **Hybrid Python+C++**: Python handles norm, attention, RoPE; C++ DLL runs heavy TQ1 matmuls.
- **OMP_NUM_THREADS**: Defaults to all cores. On i5-1235U (2P+8E), 8 threads gives best throughput.

## Potential Next Steps
1. **Pack 7B model**: Download Falcon3-7B safetensors from HF, repack for 7B benchmarks.
2. **Fuse operations**: Move attention (RoPE + score + softmax + weighted sum) into C++ DLL to reduce Python overhead.
3. **Batch prefill**: Process prompt tokens in micro-batches for faster prefill.
4. **Top-k/top-p sampling**: Add to `generate()` method.
5. **AVX2 kernel**: Alternative gather-free AVX2 path for the big matmuls (down_proj, gate_proj, up_proj).
6. **Profile guided optimization**: Use VTune or perf to find remaining bottlenecks.

## Relevant Files
- `atlas_packer.py`: Packer — streams safetensors → TQ1 ATLAS file.
- `atlas_infer.py`: Python inference engine — full forward pass + generate.
- `atlas_api.cpp`: C-exported DLL — load, matmul, rmsnorm, rope.
- `atlas_kernel.hpp`: Header-only TQ1 matmul kernels (AVX2 gather + scalar).
- `atlas.dll`: Prebuilt DLL with OpenMP (80 KB) + `libomp.dll` (968 KB).
- `compile.bat`: DLL build script (`clang++ -fopenmp -O2 -mavx2 -mfma`).
- `test_direct_cmp.py`: Atlas file vs HF ternary value comparison.
- `test_kernel_row.py`: Single-row kernel vs pure Python.
- `test_row_order.py`: Row reordering demonstration.
- `C:\models\Falcon3-10B-Instruct-1.58bit\`: Safetensors + config + tokenizer.
