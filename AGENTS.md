# ATLAS — Falcon3 TQ1.0 Inference Engine

## Goal
- TQ1.0 (Base-3, 5 ternary trits/byte) inference engine for Falcon3 models on i5 laptop, end-to-end text generation via C++ DLL/SO + Python (Windows + Linux x86-64).

## Constraints & Preferences
- 16 GB RAM only (no GPU); atlas files: 1.21 GB (1B) to 3.27 GB (10B).
- Target CPU: i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA).
- Windows: Clang LLVM-MinGW x86_64-w64-windows-gnu + `libomp.dll`.
- Linux: GCC or Clang + `libgomp` or `libomp`.
- OpenMP: `-fopenmp`. Set `KMP_DUPLICATE_LIB_OK=TRUE` for numpy MKL compat (Windows).
- **8 GB RAM**: 1B (~1.9 GB) and 3B (~4.7 GB) models run on 8 GB systems.

## Progress
### Done
- **All four Falcon3 models** packed, tested, verified:

| Model | Layers | Hidden / FFN | Heads (Q/KV) | Archive Size | Prefill Speed | Decode Speed | Per-Layer C++ | Runtime RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Falcon3-1B** | 18 | 2048 × 8192 | 8 / 4 | 1.21 GB | 26.7 tok/s | 8.9 tok/s | 3.5 ms | **1.9 GB** |
| **Falcon3-3B** | 22 | 3072 × 9216 | 12 / 4 | 1.95 GB | 10.7 tok/s | 4.6 tok/s | 5.4 ms | **4.7 GB** |
| **Falcon3-7B** | 28 | 3072 × 23040 | 12 / 4 | 2.74 GB | - | - | 13.7 ms* | **8.9 GB** |
| **Falcon3-10B** | 40 | 3072 × 23040 | 12 / 4 | 3.27 GB | **7.8 tok/s** | **1.9 tok/s** | 16.0 ms | **10.8 GB** |

> 💡 **Benchmark-Notiz (v1.0.4):** Die Werte wurden nach der Integration des spezialisierten Int8-GEMV-Kernels für den `lm_head` neu vermessen. Der Prefill-Speed des 10B-Modells klettert von 4.5 auf **7.8 tok/s** (+73%), während der Decode-Schnitt im echten Fused-Flugmodus die **1.9 tok/s** Marke durchbricht. Modelle 1B und 7B waren für den Re-Benchmark temporär nicht verfügbar.

- **Int8 mmap cache (Bug 8: cache corruption)**: Five root causes fixed:
  1. `atlas_save_cache`: seek-back pattern (`FSEEK`→`fwrite` offset→`FSEEK` data) produced duplicate offsets. Fixed: precompute all offsets via `std::vector<int64_t>`, write once.
  2. `atlas_load`: ttype=2 GQA scales got `data_size = row_dim * hidden_dim * 2` (e.g., 24576 for shape [4]), over-reading 24KB+ into next tensor. Fixed: use actual file offset gaps (`file_data_size`) — if file gap < formula, use gap; else use formula (lm_head).
  3. `atlas_save_cache`: cached ALL tensors (including inflated GQA scales). Fixed: only cache ttype==3 (int8-decoded) tensors.
  4. `atlas_infer.py`: prefetch was skipped for cached loads. Fixed: always call `atlas_prefetch_int8` regardless of cache source.
  5. `atlas_save_cache`: single `fwrite` of 70 MB+ tensors returned short writes on Windows. Fixed: write in 64 KB chunks.
- **Cache performance**: 3B load 11.1s→3.5s (3×), 10B load 35.9s→15.3s (2.3×). 10B cache: 8.86 GB.
- **fp16 lm_head RAM fix**: lm_head weight kept as fp16 (768 MB), lazily converted to fp32 on first access. `matmul_f16` uses fast full-size matmul (not chunked) for lm_head; saves 768 MB persistent RAM vs always-fp32. Fixes 10B OOM during warmup.

All with `head_dim=256`, `rope_theta=1000042`, `vocab_size=131072`, GQA architecture.

- **C++ layer loop fusion** (`atlas_forward`): all N layers run in one C call, eliminating Python loop. Ping-pong buffers (`hidden_states` ↔ `buf_out`) avoid per-layer copies. Dedicated `buf_out` prevents conflict with `buf_hidden` scratch. 10B prefill: 4.5→6.2 tok/s (+38%).
- **All 7 critical bugs fixed** (fseek overflow, Base-3 decode, row stride, K/V swap, RMSNorm truncation, snap buffer overflow, TQ1 padding overflow). All four models verified corr > 0.99 end-to-end.
- **VirtualAlloc fix**: CRT `new`/`delete` returns freed pages only on memory pressure, not to OS. VirtualAlloc/VirtualFree releases 3.35 GB packed data immediately after decompress. Prefill: 10.8→2.97s.
- **fp16 embed cache**: embed weight kept as fp16 (1.6 GB), converted on lookup. Saves 1.6 GB RAM.
- **fp16 lm_head**: kept as fp16, lazily converted to fp32 on first access. Saves 768 MB RAM vs always-fp32. Fixes 10B OOM during warmup.
- **Int8 matmul kernel**: `_mm256_maddubs_epi16` AVX2 dot product with offset trick. Replaced scalar LUT at load time.
- **v1.0.0 released** on GitHub (xxxn3m3s1sxxx/Atlas-TQ1_0) with prebuilt DLLs. Repo public.
- **v1.0.1 sterilization**: snap_* debug buffers removed, `atlas_forward_layer` (dead since fusion) removed, `atlas_find_tensor` removed. Clean `AtlasModelConfig` struct + `atlas_get_config()` for FFI consumers.
- **v1.0.1 new in cache**: int8 mmap cache (`.i8` companion file) — saves decompression on subsequent loads. 3B load 11.1s→3.5s, 10B load 35.9s→15.3s. 64KB chunked fwrite avoids short writes on large tensors.
- **v1.0.1 new in memory**: lazy fp32 lm_head conversion (chunked → full matmul for 10B). RAM peak during warmup ~14.8 GB (10B) includes temp fp32 copy of lm_head (1.5 GB transient). After warmup, persistent RAM ~11.9 GB. 3B peak ~6.5 GB, persistent ~4.7 GB.

### v1.0.2
- **Bug 9: ping-pong buffer off-by-one** (`atlas_forward`): `forward_layer` (internal per-layer helper with `n_layers=1`, odd) returned the input unchanged. Fused `forward` with all layers (even count) was always correct. Root cause: after the buf_a↔buf_b swap in the loop, the last `forward_layer_internal` output was in `buf_a` for odd counts, but code copied `buf_b` (the input). Fixed: copy from `buf_a` for odd `n_layers`.
- **Bug 10: KV cache pointer in forward_layer**: `forward_layer` passed full K/V cache pointers but `n_layers=1`, so C++ always used layer 0's cache. Masked by Bug 9 (no-op never wrote cache). Fixed: offset cache pointers per-layer in Python.
- **Cross-platform port** (`atlas_api.cpp`): all Windows-specific APIs (`VirtualAlloc`/`VirtualFree`, `CreateFileMapping`/`MapViewOfFile`, `CreateFileA`, `__declspec`) wrapped in `#ifdef _WIN32` / `#else` with POSIX equivalents (`mmap`/`munmap`, `open`, `__attribute__((visibility))`). Same file compiles on both platforms.
- **`compile-linux.sh`**: builds `libatlas.so` with g++ or clang++ (`-fopenmp -mavx2 -mfma -fPIC`).
- **PyTorch dependency removed**: `atlas_infer.py` uses `framework='np'` instead of `'pt'` for `safe_open` — no more PyTorch dependency on Linux.
- **Platform-aware DLL name**: `atlas_infer.py` loads `atlas.dll` on Windows, `libatlas.so` on Linux.
- **`atlas_forward` reverted** to pre-lm_head-fusion signature (9 params: no `idx_norm_final`, `idx_lm_head`, `logits_out`). C++ lm_head matmul was 2× slower than numpy/MKL; keeping lm_head in Python.
- **`atlas_ffi.h` updated**: signature matches reverted `atlas_forward`.
- **Benchmarks re-done**: 3B prefill 10.7 tok/s, decode 4.6 tok/s, per-layer 5.4ms. 10B prefill 2.1 tok/s, decode 1.4 tok/s, per-layer 16ms. 1B/7B unavailable.

### v1.0.3
- **lm_head fp32 pre-converted at load time**: eliminates 1240ms first-token warmup spike. Adds 1.5 GB persistent RAM.
- **10B decode breakdown**: C++ 40 layers ~504ms (85%), Python lm_head ~88ms (15%), RMSNorm ~1ms.
- **Fused vs per-layer benchmark**: fused `generate()` 1.7 tok/s vs per-layer 1.4 tok/s (19% faster).

### v1.0.4 — Die lm_head Int8-Transformation
Der `lm_head` Matmul-Block (Vokabular-Projektion) wurde vollständig aus der Python-Ebene (NumPy/FP32) verbannt und als nativer AVX2-Kernel in den C++ Core integriert.

* **Mathematik:** Symmetrische Per-Row-Quantisierung ($W_q \in \text{int8}$) mit vorberechneten statischen Zeilensummen ($\sum_i Wq_{j,i}$) zur Kompensation des vorzeichenlosen `_mm256_maddubs_epi16` $+128$ Shifts.
* **Speicher-Layout:** Der persistente RAM-Bedarf des `lm_head` sank drastisch von **1.5 GB (FP32) auf ~403 MB (Int8)**, wodurch wertvolle **1.1 GB System-Arbeitsspeicher freigegeben** wurden.
* **Inferenz-Gewinne (10B):**
  * `lm_head` Decode (B=1): **88.4 ms → 18.9 ms (5.0× Speedup)**
  * `lm_head` Prefill (B=18): **395.9 ms → 41.8 ms (9.5× Speedup)**
  * **Full decode step**: ~516ms/tok (**1.9 tok/s**, 19% faster)
* **`_matmul_f16` fallback**: auto-detects null tensor data and redirects to GEMV — all existing bench/tools/scripts continue to work.
* **`atlas_ffi.h` updated**: `atlas_quantize_lmhead(model, idx)` + `atlas_lmhead_gemv(model, act, out, B)`.

### v1.0.5 — Fused Gate/Up FFN Pipeline
* Fused sequential `gate_proj` and `up_proj` execution into a single OMP parallel region over 4-row packed groups.
* Eliminated raw matmul scratch buffer (`buf_hidden`) — inline reorder+dequant writes directly to `buf_gate`/`buf_up`.
* Shared activation quantization stays hot in L1 cache (no eviction between gate+up).
* **Benchmark (10B):**

| Metrik | v1.0.4 (separat) | v1.0.5 (fused) | Δ |
|--------|:-:|:-:|:-:|
| C++ layers total | 612.0 ms | **554.9 ms** | **−57 ms (9.3%)** |
| Per-layer mean | 15.3 ms | **13.9 ms** | **−1.4 ms** |
| Full decode step | ~632 ms | **~576 ms** | **−56 ms (8.8%)** |
| GEN (9 tok) | 4.8 s (1.9 tok/s) | **4.4 s (2.0 tok/s)** | **+0.1 tok/s** |

* **Coherence verified**: `"The capital of France is Paris."` — same exact output.
* No file format change — fully backward compatible with v1.0.4 `.atlas` files.

### v1.0.6 (current) — Fused SiLU+MUL+Quantize Pipeline
* Eliminated four separate RAM passes (silu_inplace → mul → copy → quantize) into a single two-pass-per-token kernel.
* `silu_inplace()` removed — SiLU computed inline during fused pass.
* buf_hidden removed from the critical path — result written directly to buf_act.
* **Benchmark (10B):**

| Metrik | v1.0.5 (separat) | v1.0.6 (fused) | Δ |
|--------|:-:|:-:|:-:|
| C++ layers total | 554.9 ms | **454.8 ms** | **−100 ms (18.0%)** |
| Per-layer mean | 13.9 ms | **11.4 ms** | **−2.5 ms** |
| Full decode step | ~576 ms | **~473 ms** | **−103 ms (17.9%)** |

**Decode Rate: 2.1 tok/s**

* **Coherence verified**: 3B `"Hello! How can I assist you today?"`, 10B `'Paris'`.
* No file format change — fully backward compatible.

### Blocked
- **7B model**: safetensors not present on this machine. Only GGUF variant exists.

## Key Decisions
- **TQ1 decompressed to int8 at C++ load time**: 126-280 tensors decoded once, packed data freed. Fast `maddubs` SIMD.
- **VirtualAlloc/VirtualFree**: CRT heap pages not returned to OS → VirtualFree with MEM_RELEASE frees both VA space and RAM immediately.
- **fp16 embed cache**: Embed kept as fp16 to save 1.6 GB RAM.
- **Cache raw ctypes pointer** instead of `create_string_buffer` for fp16 weight passing to avoid NULL-byte truncation.
- **C++ layer fusion** (`atlas_forward`): all layers in one C call. Dedicated `buf_out` buffer for ping-pong (not `buf_hidden`, which is scratch).
- **8GB RAM support**: explicitly documented in README for 1B and 3B models.
- **File naming**: `falcon3-{size}b-tq1_0.atlas` convention (.atlas = engine, tq1_0 = format+version).
- **Int8 lm_head**: per-row symmetric quantization with float dequant scales. Stored separately from TensorInfo (not in cache). Frees fp16 data on quantize. Fallback path in `_matmul_f16` for backward compat.

## Next Steps
1. **7B enablement (re-evaluate)**: with int8 lm_head + fused SiLU, RAM ~8.4 GB. 7B safetensors needed.
2. **Test on native Linux hardware** for reliable cross-platform numbers.
3. **Simplify installation**: PyPI package, single-command install, auto-download prebuilt binaries.
4. **Expand int8 mmap cache** to cover fp16 tensors (embed, norms) → sub-second load.
5. **Port to Mojo / Rust / Zig** — after C++ kernel is confirmed as pure source of truth on both Windows + Linux.

## Relevant Files
- `atlas_packer.py`: Packer — streams safetensors → TQ1 ATLAS file.
- `atlas_infer.py`: Python inference engine — forward + generate.
- `atlas_api.cpp`: C-exported DLL — load, decompress, `atlas_forward` (fused layers), int8 matmul, fused attention, norms, RoPE.
- `atlas.dll`: Prebuilt DLL with OpenMP (107 KB) + `libomp.dll` (968 KB).
- `libatlas.so`: Prebuilt shared library for Linux (39 KB).
- `compile.bat`: DLL build script (`clang++ -fopenmp -O2 -mavx2 -mfma`).
- `compile-linux.sh`: SO build script (`g++ -fopenmp -O2 -mavx2 -mfma -fPIC`).
- `bench_atlas.py`: Benchmark — load time, per-layer profile, prefill + decode.
- `C:\atlas\falcon3-{1b,3b,7b,10b}-tq1*.atlas`: Packed model files.
- `C:\models\Falcon3-{3B,7B,10B}-Instruct-1.58bit\`: Safetensors + config + tokenizer.
