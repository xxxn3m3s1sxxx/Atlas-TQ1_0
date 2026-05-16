# ATLAS — TQ1.0 Ternary Inference Engine

ATLAS is a CPU-based inference engine for BitNet b1.58 ternary-quantized language models. It repacks HuggingFace safetensors into the **TQ1.0** format (5 ternary trits packed per byte, Base-3 encoded) and runs fast inference through a hybrid C++ DLL + Python pipeline. No GPU required.

## Motivation

BitNet b1.58 models replace full-precision weights with ternary values (-1, 0, +1), reducing memory by 16-32x versus FP16. However, existing inference frameworks target CUDA GPUs, leaving CPU users unable to run these models. ATLAS fills this gap: it packs ternary weights into a compact TQ1 format and uses lookup-table (LUT) matmuls on CPU to achieve usable decode speeds on commodity hardware.

## Supported Models

| Model | Parameters | Atlas Size | Layers | Architecture |
|-------|-----------|-----------|--------|--------------|
| Falcon3-7B-Instruct-1.58bit | 7B | 2.74 GB | 28 | LlamaForCausalLM + BitLinear |
| Falcon3-10B-Instruct-1.58bit | 10B | 3.27 GB | 40 | LlamaForCausalLM + BitLinear |
Both use `hidden_size=3072`, `intermediate_size=23040`, GQA 12/4 heads, `head_dim=256` (so Q output = 12×256 = 3072 = hidden_dim). The 10B packs more layers (40 vs 28).

## Quick Start

### Requirements

- Python 3.10+ with `numpy`, `safetensors`, `transformers`, `torch`
- 16 GB RAM (no GPU)
- Clang 22.1.5+ (LLVM MinGW) to rebuild the DLL (optional)
- Windows (originally developed on i5-1235U, portable to Linux with `fseek`/`fseeko` change)

```
pip install numpy safetensors transformers torch
```

### 1. Pack safetensors into ATLAS format

```bash
python atlas_packer.py path/to/model.safetensors model.atlas
```

The packer reads the safetensors file, de-interleaves BitNet's 4-row-packed uint8 format, repacks each weight row into TQ1 Base-3 bytes, and writes an ATLAS file with a 64-byte header, 12-byte-per-tensor directory, and tensor data blobs. The model directory (containing `config.json` and tokenizer) is inferred from the safetensors path.

### 2. Run inference

```bash
python atlas_infer.py model.atlas path/to/model_dir "What is the capital of France?"
```

The inference script loads the atlas file into the C++ DLL, initializes the tokenizer from the model directory, and runs autoregressive generation with GQA attention, RoPE, KV cache, SiLU FFN, and temperature sampling.

### 3. Compile the DLL (if modifying C++ code)

```bash
compile.bat
```

Requires `clang++` (LLVM MinGW) in PATH. Builds `atlas.dll` with AVX2+FMA and OpenMP.

OpenMP is enabled by default. `libomp.dll` must be discoverable at runtime — copy it from `llvm-mingw\x86_64-w64-mingw32\bin\libomp.dll` next to `atlas.dll`, or add that directory to `PATH`.

**Windows-specific**: The MKL backend that numpy may use loads `libiomp5md.dll`, a different OpenMP runtime. This causes `OMP: Error #15` at import time. `atlas_infer.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically to suppress this. If you see the error despite this, set `KMP_DUPLICATE_LIB_OK=TRUE` in your environment before running.

## Architecture

```
safetensors ──► atlas_packer.py ──► .atlas file ──► atlas_infer.py ──► atlas.dll (C++)
                                                         │
                                                    [virtual] forward_layer
                                                         │
                                              ┌──────────┼──────────┐
                                          RMSNorm  Attention  FFN(SiLU)
                                         (C++ inline) (C++)  (7× int8)
```

### Pipeline stages

1. **Packer** (`atlas_packer.py`): Reads HuggingFace safetensors with pre-quantized 2-bit packed weights (`byte = v0 + v1*4 + v2*16 + v3*64`). De-interleaves the BitNet row-aware format (4 weight rows per uint8 row). Repacks each row independently into TQ1.0: 5 ternary trits mapped to Base-3 values 0-242, padded with ternary-0 (mapped from value 1) to fill the last byte. FP16 per-tensor scales are stored as 2-byte prefixes.

2. **ATLAS file format**: Binary file with a 64-byte header (magic "ATLAS", layer count, hidden dim, intermediate dim, head counts, vocab size, tensor count). Followed by a directory of 12-byte entries (ttype, file offset, row dim, packed cols). Tensor data follows: TQ1 weights (ttype=0) have a 2-byte FP16 scale prefix followed by packed rows; embeddings/norms (ttype=1) and LM head (ttype=2) are raw FP16.

3. **DLL** (`atlas_api.cpp`): Loads the atlas file into memory. On load, TQ1 tensors are decompressed from Base-3 packed format to int8 (one VirtualAlloc per tensor, packed data freed). Then `atlas_forward_layer` runs one complete transformer layer — RMSNorm + 7× int8 matmul (Q/K/V/O/gate/up/down) + fused GQA attention (RoPE + cache + causal softmax + weighted sum) + SiLU FFN — in a single C call. No Python round-trips between sub-operations.

4. **Int8 matmul kernel** (`atlas_matmul_i8_f32`): Uses `_mm256_maddubs_epi16` AVX2 dot-product with a +128 offset trick. Weights are stored as signed int8 (decompressed from TQ1 at load). Activations are quantized per-token to uint8 with max-abs scaling. Output rows are deinterleaved from the BitNet 4-row-packed layout back to natural order. OpenMP parallelizes over rows.

5. **Python inference** (`atlas_infer.py`): Coordinates loading, tokenization, and the autoregressive loop. Each layer calls `atlas_forward_layer` in C++. Only the final RMSNorm, LM head matmul, and sampling remain in Python.

All working buffers use `VirtualAlloc`/`VirtualFree` (not CRT `new`/`delete`) so freed memory is returned to the OS immediately — critical for staying within 16 GB RAM on a 3.35 GB model.

## Performance

Measured on i5-1235U (Alder Lake, 2P+8E, 8 OMP threads, AVX2+FMA, 16 GB DDR4).

| Model | Prefill (12 tok) | Decode (per token) | Per-Layer C++ |
|-------|------------------|-------------------|---------------|
| Falcon3-10B (40L, 3072x23040) | 2.7 s (4.5 tok/s) | 0.64 s (1.6 tok/s) | 15 ms mean |

### Breakdown (Falcon3-10B)

- **Per-layer C++ forward**: ~15 ms (RMSNorm + 7× int8 matmul + fused attention + SiLU FFN)
- **Python overhead**: ~41 ms (cache indexing, ctypes marshalling, sampling)
- **Total decode**: ~641 ms/token → 1.6 tok/s

The Int8 AVX2 kernel (`_mm256_maddubs_epi16` dot-product, deinterleaved output) replaced the earlier scalar LUT-based TQ1 matmul. This gave ~2.8× speedup on large projections (e.g., gate_proj: 8.9 → 3.2 ms). OpenMP adds another ~3-5× on 8 cores versus single-threaded. The scalar LUT kernel was chosen over AVX2 gather (4× slower on Alder Lake due to gather latency), then superseded by the int8 approach.

### Memory

| Component | Size |
|-----------|------|
| Int8 weight cache (decompressed TQ1, 280 tensors) | 9.5 GB |
| FP16 embedding cache | 1.6 GB |
| KV cache (fp16, seq_len=4096, 40 layers) | 0.34 GB |
| Python + overhead | ~0.5 GB |
| **Total** | **~11.9 GB** |

Fits comfortably in 16 GB. Using `VirtualAlloc`/`VirtualFree` for tensor data ensures the 3.35 GB of packed TQ1 data is freed and returned to the OS after decompression, avoiding page thrashing that occurred with CRT `new`/`delete`.

## Bugfix Chronology

Six critical bugs were discovered and fixed during development. Any one of them would cause the model to produce garbage output (correlation near zero with reference activations).

### Bug 1: `fseek` 32-bit overflow

The ATLAS file for Falcon3-7B is 2.74 GB. Tensors beyond offset ~2 GB were being read from the wrong file position because `fseek` (32-bit) truncated the offset. Fixed by replacing with `_fseeki64` (Windows) / `fseeko` (POSIX) via a `FSEEK` macro.

**Symptoms**: Layer-0 projections correct, deeper layers produce NaN or garbage.

### Bug 2: 2-bit packing vs Base-3 unpacking

HuggingFace BitNet safetensors store 2-bit packed ternary values: `byte = v0 + v1*4 + v2*16 + v3*64`. The original packer decoded them with `%3` and `//3` (Base-3), producing incorrect ternary values. Fixed by using `& 3`, `>> 2`, etc.

**Symptoms**: Weight values off by ~5% per element, correlation still measurable (~0.5) but never reaching 1.0.

### Bug 3: Row ordering (interleaved vs stride)

BitNet stores weights in a row-aware interleaved format: uint8 row `ur` contains columns for output rows `4*ur+0` through `4*ur+3`. The C++ matmul output is in this interleaved order (`ur*4+q`). But the reference HuggingFace `unpack_weights` produces stride-order output (`q*rows_packed+ur`). Without reordering, every projected tensor had correlation near 0 despite correct ternary values.

**Fix**: `out.reshape(batch, rows_packed, 4).transpose(0, 2, 1).reshape(batch, rows)`.

**Symptoms**: Correlation ~0 with reference, but RMS magnitude was ~1. This was the most deceptive bug — it looked like a scale or encoding issue but was purely a layout mismatch.

### Bug 4: K/V cache swap

In `atlas_forward_layer`, K was written to `buf_hidden` but the attention copy read from `buf_up`; V was written to `buf_up` but read from `buf_hidden`. Fixed by swapping the copy destinations so K→buf_up and V→buf_hidden.

**Symptoms**: Attention output garbage, correlation ~0 despite correct QKV projections.

### Bug 5: `_rmsnorm` weight truncation (create_string_buffer)

`ctypes.create_string_buffer()` treats the input as a C-string and truncates at the first NULL byte (`\x00`). FP16 value `1.0` = bytes `\x00\x3C` (little-endian), so RMSNorm weights containing many values ≈1.0 got truncated at the first such value, zeroing most norm outputs. This was the root cause of layer 0 having corr=0.94 (not 1.0) in the C++ forward layer.

**Fix**: Cache the DLL's raw `ctypes.POINTER(c_uint8)` directly instead of converting to `bytes` → `create_string_buffer`. Also fixed `len(x)` (returns 1 for `(1, 3072)`) → `x.shape[-1]`.

**Symptoms**: RMSNorm output had only the first element non-zero; layer output near zero for most elements. This was a double bug — `create_string_buffer` truncation AND `len(x)` returning 1 instead of 3072.

### Bug 6: Snap buffer overflow (batch resize)

Debug snapshot buffers (`snap_q/k/v/o/norm1`) were allocated once with the initial batch size and never resized. Prefill (B=12) after decode warmup (B=1) wrote past the end, causing `OSError: exception: access violation writing 0x...` .

**Fix**: Resize snap buffers in `ensure_buffers()` alongside the working buffers, using the new batch size.

**Symptoms**: Immediate access violation crash when switching from single-token decode to batched prefill.

### Verification

After all six fixes, the full 40-layer C++ forward pass matches the Python reference with correlation > 0.99 end-to-end. The model produces correct text: "What is 2+2?" → "2+2=4", "Hello" → "Hello! How can I assist you today?".

## File Reference

| File | Purpose |
|------|---------|
| `atlas_packer.py` | Streams safetensors → TQ1 ATLAS file |
| `atlas_infer.py` | End-to-end inference engine (Python) |
| `atlas_api.cpp` | C-exported DLL (load, decompress TQ1→int8, forward_layer, int8 matmul, fused attention, norms) |
| `atlas.dll` | Prebuilt DLL (Clang LLVM-MinGW, 107 KB) |
| `libomp.dll` | OpenMP runtime (968 KB, ships with LLVM-MinGW) |
| `compile.bat` | DLL build script (`clang++ -fopenmp -O2 -mavx2 -mfma`) |
| `test_layer0.py` | Full 40-layer C++ forward vs Python reference verification |
| `bench_atlas.py` | Benchmark: load time, per-layer profiling, prefill + decode |

## License

This project's code (TQ1.0 format, packer, inference engine, kernels) is Apache 2.0. BitNet b1.58 is from Microsoft Research. Falcon3 models are from Technology Innovation Institute (TII) and subject to the [TII Falcon License 1.0](https://falconllm.tii.ae/). Users must accept TII's license to download and use Falcon3 weights.

## Citation

If you use this work in research:

```bibtex
@misc{atlas-tq1,
  title = {ATLAS: A TQ1.0 Ternary Inference Engine for BitNet b1.58 on CPU},
  year = {2026}
}
```
