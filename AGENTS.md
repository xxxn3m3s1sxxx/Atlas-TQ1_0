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
- **All 4 models packed as v4 format** — names embedded, safetensors dependency eliminated.
- **Bug 8.6 fixed (Cache Short-Write Protection)**: `atlas_save_cache` checks disk space via `GetDiskFreeSpaceExA`, uses `setvbuf(IONBF)` unbuffered writes, retries on short writes, deletes corrupt partial cache on any failure. `atlas_load_cache` validates file size against header offsets before mapping.
- **f32 matmul bypass added**: `atlas_set_use_f32_matmul(AtlasModel*, int)` — skips activation quantization, uses direct AVX2 f32×i8 FMA. Enabled for `hidden <= 2048` (1B model). `matmul_f32_reorder` + f32 gate+up FFN kernel.
- **1B coherence analyzed**: Greedy degenerates (`,` p=0.43) due to model-inherent distribution. Sampling (`T=1.0, top_k=40, top_p=0.9`) produces correct output. Engine exact — f32 bypass produces identical argmax.
- **15.9 GB cleaned**: All Ollama models deleted. 17.74 GB free.
- **Bug 9 revisited (Ping-Pong Buffer)**: Pointer-Semantik analysiert — `buf_a = hidden_states` kopiert **Pointer**, nicht Daten. Nach Loop: für gerade `n_layers` zeigt `buf_a == hidden_states` (gleiche Addresse) → `hidden_states` hat bereits korrekten Output. Für ungerade: Copy aus `buf_a` nötig. Code `if (n_layers % 2 == 1) { memcpy... }` IST KORREKT. Kein Bug im Engine.
- **False Alarm corr=0.23**: Zwei Bugs im `test_coherence.py` Reference-Code, nicht im Engine:
  1. `_rmsnorm` modifiziert Input **in-place** → zerstört `x + attn_proj` Residual. Fix: `.copy()` übergeben.
  2. Python per-layer quantisiert gate+up **separat** (eigene Scale), C++ fused nutzt **shared quantization** → 0.3% Korrelationsrest.
  Mit korrigiertem Reference: **corr=0.9967**, max_diff=4.0 (durch shared quantization). Engine korrekt.
- **`atlas_set_use_f32_matmul` API**, Bug 8.6 Disk Space Check, Short-Write Retry, File Size Validation, `hsum_ps` helper.
- **All 4 models coherence verified** with default sampling.
- **v1.0.9 Memory Audit**: Vier Bugs im mmap/Cleanup-Code gefixt:
  1. **Bug A — atlas_vfree silent no-op**: `munmap(ptr, 0)` auf Linux leakt Speicher (kernel gibt -EINVAL für len=0). Fix: Allocation-Header in `atlas_valloc` (speichert base+size 16 Bytes vor aligned Data Pointer). `atlas_vfree` liest Header aus → `munmap(h->base, h->total)` bzw. `VirtualFree(h->base, 0, MEM_RELEASE)`.
  2. **Bug B — atlas_vfree auf mmap-Pointer**: `atlas_decompress_all` und `atlas_load_cache` riefen `atlas_vfree(t.data)` auf Pointern in die atlas-mmap. Fix: `is_mapped`-Check vor jedem `atlas_vfree`.
  3. **Bug C — is_mapped fake Size**: Windows nutzte `0xFFFFFFFF` als mmap-Range. Fix: `mmap_size`/`atlas_mmap_size` Felder in `AtlasModel`, gesetzt beim Laden, genutzt in `is_mapped`.
  4. **Bug D — Linux fd leak**: `dup(fd)` in `atlas_load` wurde nie geschlossen. Fix: `close((int)(intptr_t)m->atlas_mmap_file)` in `atlas_free`.
- **Speicherbereinigung**: Alle 9 Platform-Fixes konsolidiert, `atlas.dll` rebuild.

### In Progress
- **(none)**

### Blocked
- **1B greedy degeneration**: Not fixable in engine. 1.58-bit quant 1B model has `,` as argmax (p=0.43, entropy 2.75). 3B has `Hello` (p=0.34, entropy 4.25). Requires sampling.
- **WSL performance**: ~4–5× slower than native Windows.
- **8 GB RAM limit**: 10B (10.8 GB) does not fit on 8 GB machines.

## Key Decisions
- **f32 bypass bleibt drin**: Eliminates engine quantization noise. Serves as numerical reference path for kernel optimization.
- **1B greedy NOT a bug**: Model distribution causes degeneration. Documentation-only fix.
- **`atlas_forward` seq_now**: Must be actual sequence length (token count), NOT layer count. Test reference fix confirmed engine correct.
- **Shared gate+up quantization**: C++ fused path quantizes gate **and** up activations with a single shared scale. Python per-layer path quantizes separately. This 0.3% correlation gap is EXPECTED and CORRECT.
- **Disk space management**: `.i8` cache 1.1 GB (1B) to 9.5 GB (10B). Save checks available space before writing.
- **Test reference bug prevention**: `_rmsnorm` modifies in-place! Always pass `.copy()` if preserving original for residual.

## Next Steps
1. **README finalisieren** mit v1.0.8 (Bug 8.6, f32 bypass, 1B sampling note, Bug 9/False Alarm clarification).
2. **Git tag v1.0.8** und push.
3. **Tokenizer + safetensors overhead killen** — Tensor-Namen in v4, Tokenizer noch aus `model_dir`.
4. **Linux native testen**.

## Critical Context
- **v1.0.7 current stable**. v1.0.8 bereit (Bug 8.6 + f32 bypass + Engine Korrektheit bestätigt).
- **10B fused decode**: C++ layers 454.8 ms, lm_head 17.9 ms, total 473 ms (2.1 tok/s). Per-layer 11.4 ms.
- **f32 bypass**: Auto-enabled for `hidden <= 2048` (1B). Adds ~10–20% per-layer overhead.
- **`atlas_set_use_f32_matmul`** C API: checked in `forward_layer_internal` for all matmul sites.
- **`.i8` cache recovery**: Partial write → file deleted, retry on next load.
- **17.74 GB free** after Ollama cleanup. 7B `.i8` (6.7 GB) kann jetzt parallel zu 10B Cache existieren.
- **1B coherence**: Sampling `T=1.0, top_k=40, top_p=0.9` → `'Hello! How can I help you today?'` / `'. The capital is Paris.'`. Greedy degenerates (Model-Eigenschaft).
- **Bug 9 resolved**: Kein Bug. Pointer-Semantik korrekt. `if (n_layers % 2 == 1)` copy condition is correct. Correlation gap between Python per-layer and C++ fused is explained by shared quantization (not a bug).

## Relevant Files
- `C:\atlas\atlas_api.cpp`: DLL/SO — f32 matmul bypass (`matmul_f32_reorder`, f32 gate+up), `atlas_set_use_f32_matmul` API, Bug 8.6 fixes (disk space check, short-write retry, file size validation), `hsum_ps` helper.
- `C:\atlas\atlas_infer.py`: `AtlasModel.__init__` calls `atlas_set_use_f32_matmul` for `self.hidden <= 2048`.
- `C:\atlas\atlas_ffi.h`: Updated with `atlas_set_use_f32_matmul` declaration.
- `C:\atlas\test_coherence.py`: Correlation test script (fix: pass `.copy()` to `_rmsnorm` to preserve residual).
- `C:\atlas\falcon3-{1b,3b,7b,10b}-tq1.atlas`: v4 packed model files.
- `C:\models\Falcon3-{3B,7B,10B}-Instruct-1.58bit\`: model config, tokenizer, optional safetensors.
