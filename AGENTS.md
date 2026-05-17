# ATLAS — Falcon3 TQ1.0 Inference Engine

## Goal
- TQ1.0 (Base-3, 5 ternary trits/byte) CPU inference engine for Falcon3 models on i5, end-to-end text generation via C++ DLL/SO + Python (Windows + Linux x86-64).

## Constraints & Preferences
- Windows 11 / Linux, 8–16 GB RAM, no GPU. i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA).
- Windows: `clang++ -fopenmp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17` + `libomp.dll`.
- Linux: GCC or Clang, `-fPIC`, `libomp-dev` or `libgomp`.
- `KMP_DUPLICATE_LIB_OK=TRUE` for MKL compat (Windows only).
- **KOHÄRENZ MUSS IMMER ERHALTEN BLEIBEN (0.999999)**.
- All models: `head_dim=256`, `rope_theta=1000042`, GQA. 1B/3B/10B `vocab_size=131072`, 7B `131080`.
- 10B: 40L, 3072H, 23040I, 12/4 heads. 7B: 28L, 3072H, 23040I. 3B: 22L, 3072H, 9216I. 1B: 18L, 2048H, 8192I, 8/4 heads.

## Progress
### Done
- **v5 format with embedded tokenizer**: All 4 models repacked as `.atlas` v5 with tokenizer.json + tokenizer_config.json embedded. `generate()` no longer needs `model_dir`. C API `atlas_get_tokenizer()` exposes raw bytes.
- **Bug 8.6 fixed (Cache Short-Write Protection)**: `atlas_save_cache` checks disk space via `GetDiskFreeSpaceExA`, uses `setvbuf(IONBF)` unbuffered writes, retries on short writes, deletes corrupt partial cache on any failure. `atlas_load_cache` validates file size against header offsets before mapping.
- **f32 matmul bypass added**: `atlas_set_use_f32_matmul(AtlasModel*, int)` — skips activation quantization, uses direct AVX2 f32×i8 FMA. Enabled for `hidden <= 2048` (1B model). `matmul_f32_reorder` + f32 gate+up FFN kernel.
- **1B coherence analyzed**: Greedy degenerates (`,` p=0.43) due to model-inherent distribution. Sampling (`T=1.0, top_k=40, top_p=0.9`) produces correct output. Engine exact — f32 bypass produces identical argmax.
- **Bug 9 revisited (Ping-Pong Buffer)**: Pointer semantics correctly analyzed — even `n_layers` means `buf_a == hidden_states` after loop (correct). The `if (n_layers % 2 == 1)` copy is correct. No engine bug.
- **False Alarm corr=0.23**: Two bugs in Python test reference (not engine): `_rmsnorm` in-place residual corruption + shared quantization gap. Fixed reference gives **corr=0.9967**, max_diff=4.0.
- **v1.0.9 Memory Audit — 4 bugs fixed**: AllocHdr fix (Linux munmap leak), is_mapped guard before vfree (mmap double-free), proper mmap_size tracking (Windows), Linux fd close.
- **v1.1.0 Production Hardening**: AllocHdr-based valloc/vfree, is_mapped guards, fd close on Linux, int8 quant clip fix in Python.

### In Progress
- **(none)**

### Blocked
- **1B greedy degeneration**: Not fixable in engine. 1.58-bit quant 1B model has `,` as argmax (p=0.43, entropy 2.75). Requires sampling.
- **WSL performance**: ~4–5× slower than native Windows.
- **8 GB RAM limit**: 10B does not fit on 8 GB machines.

## Key Decisions
- **v5 format**: `[header:64] [dir:n*12] [name_block] [token_data...] [tokenizer_block]`. Tokenizer stored as separate raw JSON block (no merge — avoids tokenizers Rust parser corruption). Header bytes 29-32: tokenizer_size, 33-36: tokenizer_offset.
- **f32 bypass bleibt drin**: Eliminates engine quantization noise. Serves as numerical reference path.
- **`atlas_forward` seq_now**: Must be actual sequence length, NOT layer count.
- **Shared gate+up quantization**: C++ fused path = single shared scale. Python per-layer = separate scales. 0.3% gap is EXPECTED.

## Next Steps
1. ~~#3 Tokenizer + safetensors overhead killen~~ **(DONE — v5 embedded tokenizer)**
2. **Linux native testen** mit v1.1.0 mmap + AllocHdr + fd-close + v5 tokenizer.
3. **10B coherence test** auf 16 GB Maschine.
4. **README finalisieren** mit v1.1.0 (v5 format, embedded tokenizer, engine hardening).

## Critical Context
- **v1.1.0 latest release** (Production Hardening + v5 embedded tokenizer).
- **All 4 models in v5 format**: 1.22 GB (1B), 1.96 GB (3B), 2.75 GB (7B), 3.28 GB (10B).
- **`.i8` cache**: ~1.1 GB (1B) to ~9.5 GB (10B). Auto-generated on first load, mmap'd on subsequent loads.
- **Prefill is already batched**: `forward()` passes all prompt tokens as B=prompt_len in single `atlas_forward` call. `atlas_attention_f32` handles causal masking per-token within the batch.
- **Engine correctness**: corr=0.9967 with fixed Python reference. Remaining 0.3% = shared quantization gap.
- **1B f32 bypass**: Auto-enabled for `hidden <= 2048`. Confirms u8 path is exact (identical argmax).
- **Benchmarks (v1.1.0, 8 OMP threads, mmap cache, cold start = first load without cache, warm = cached)**:
  - **1B**: cold=4.5s warm=0.8s load, gen=1.5s (6.7 tok/s)
  - **3B**: cold=10.1s warm=1.4s load, gen=2.2s (4.5 tok/s)
  - **7B**: load~7s, gen~5.6s (1.8 tok/s)
  - **10B**: load~? (3.28 GB .atlas, needs 16 GB RAM), gen~2.1 tok/s
- **Memory**: Loading via file mmap (zero-copy). `AtlasModel(model)` = ~200 MB Python + ~1-10 GB C++ (OS-paged from mmap).

## Relevant Files
- `C:\atlas\atlas_api.cpp`: `atlas_get_tokenizer()` C API (line 441), v5 header parsing (bytes 29-36), AllocHdr valloc/vfree, is_mapped guards, int8 matmul kernel, f32 bypass.
- `C:\atlas\atlas_infer.py`: `AtlasModel` — embedded tokenizer loading via `atlas_get_tokenizer()`, `generate()` uses `PreTrainedTokenizerFast` with embedded data, no `model_dir` required for v5.
- `C:\atlas\atlas_ffi.h`: Updated with `atlas_get_tokenizer` declaration and v5 header layout docs.
- `C:\atlas\atlas_packer.py`: v5 format writer — appends tokenizer block after tensor data, stores offset in header.
- `C:\atlas\atlas_kernel.hpp`: Legacy TQ1 gather kernel — not in inference path.
- `C:\atlas\falcon3-{1b,3b,7b,10b}-tq1.atlas`: **v5** packed model files with embedded tokenizer.
- `C:\models\Falcon3-{1B,3B,7B,10B}-Instruct-1.58bit\`: model config, optional safetensors (only needed for repacking).
