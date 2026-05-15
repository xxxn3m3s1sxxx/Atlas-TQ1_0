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
- **Fused attention** (`atlas_attention_f32`): RoPE + GQA + softmax + weighted sum in one C call.
- **RMSNorm cache**: raw fp16 bytes cached to avoid repeated DLL roundtrips.
- **Top-k/Top-p sampling**: added to `generate()` (default: top_k=40, top_p=0.9).
- **Int8 matmul kernel**: `_mm256_maddubs_epi16` AVX2 dot product with offset trick.
  - All 280 TQ1 tensors decompressed to int8 at load time (once), packed data freed.
  - gate_proj 1tok: 8.9ms → 3.2ms (2.8× speedup)
  - Layer 0 (hot): ~19ms (vs ~29ms before int8)
  - Decode: 1.0 → ~1.4 tok/s (40% gain)
- `KMP_DUPLICATE_LIB_OK=TRUE` set in `atlas_infer.py` for numpy MKL compat.
- README paths fixed to relative/portable.
- `atlas_*.dll` temp builds gitignored.

### Open
- **Prefill is slow**: 15s for 18-token prompt (includes load-time decompression of all layers). Prefill matmul batching (seq=6 → 33ms vs seq=1 → 19ms) limits improvement.
- **Memory**: 16GB RAM tight. Int8 weights ~9.5GB + KVCache 0.34GB + Python 1GB + Windows 3-4GB ≈ 14-15GB total.

### Blocked
- **7B model**: safetensors not present on this machine. GGUF variant exists at `C:\models\Falcon3-7B-Instruct-1.58bit-GGUF` but packer reads safetensors only.

## Key Decisions
- **TQ1 decompressed to int8 at C++ load time**: 280 tensors decoded once, packed data freed. Fast `maddubs` SIMD replaces LUT lookups.
- **Per-tensor int8 cache in C++** (via `atlas_get_int8`): Python creates zero-copy numpy views into DLL memory.
- **Scalar over AVX2 gather**: Gather was 4× slower (40 vs 162 tok/s) on Alder Lake.
- **Dynamic OpenMP**: `libomp.dll` at runtime (no static libomp.a in LLVM-MinGW).
- **Hybrid Python+C++**: Python handles norm, attention, RoPE; C++ DLL runs heavy int8 matmuls.
- **OMP_NUM_THREADS**: Defaults to all cores. On i5-1235U (2P+8E), 8 threads gives best throughput.

## Potential Next Steps
1. **Prefill batching in C++**: Fuse all 7 matmuls for a layer into one call, reducing activation quantization overhead.
2. **Batch FFT/strassen**: Not applicable.
3. **Profile-guided optimization**: Compile with PGO for 10-15% further gain.
4. **Full C++ forward pass**: Rewrite entire per-layer forward in C++ (no Python loop), aim for 2×.
5. **7B model**: Download safetensors from HF, repack for 7B benchmarks.
6. **Mmap-based int8 loading**: Avoid load-time decompression from packed format.

## Relevant Files
- `atlas_packer.py`: Packer — streams safetensors → TQ1 ATLAS file.
- `atlas_infer.py`: Python inference engine — full forward pass + generate.
- `atlas_api.cpp`: C-exported DLL — load, matmul, rmsnorm, rope, int8 matmul, decompress.
- `atlas.dll`: Prebuilt DLL with OpenMP (96 KB) + `libomp.dll` (968 KB).
- `compile.bat`: DLL build script (`clang++ -fopenmp -O2 -mavx2 -mfma`).
- `C:\models\Falcon3-10B-Instruct-1.58bit\`: Safetensors + config + tokenizer.
