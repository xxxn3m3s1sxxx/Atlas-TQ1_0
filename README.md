# ATLAS — TQ1.0 Ternary Inference Engine

ATLAS is a CPU-based inference engine for BitNet b1.58 ternary-quantized language models. It repacks HuggingFace safetensors into the **TQ1.0** format (5 ternary trits packed per byte, Base-3 encoded) and runs fast inference through a hybrid C++ DLL + Python pipeline. No GPU required.

## Motivation

BitNet b1.58 models replace full-precision weights with ternary values (-1, 0, +1), reducing memory by 16-32x versus FP16. However, existing inference frameworks target CUDA GPUs, leaving CPU users unable to run these models. ATLAS fills this gap: it packs ternary weights into a compact TQ1 format and uses lookup-table (LUT) matmuls on CPU to achieve usable decode speeds on commodity hardware.

## Supported Models

| Model | Parameters | Atlas Size | Layers | Architecture |
|-------|-----------|-----------|--------|--------------|
| Falcon3-7B-Instruct-1.58bit | 7B | 2.74 GB | 28 | LlamaForCausalLM + BitLinear |
| Falcon3-10B-Instruct-1.58bit | 10B | 3.27 GB | 40 | LlamaForCausalLM + BitLinear |
Both use `hidden_size=3072`, `intermediate_size=23040`, `head_dim=256`. The 10B packs more layers (40 vs 28).

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
safetensors ──► atlas_packer.py ──► .atlas file ──► atlas_infer.py (Python) ──► atlas.dll (C++)
```

### Pipeline stages

1. **Packer** (`atlas_packer.py`): Reads HuggingFace safetensors with pre-quantized 2-bit packed weights (`byte = v0 + v1*4 + v2*16 + v3*64`). De-interleaves the BitNet row-aware format (4 weight rows per uint8 row). Repacks each row independently into TQ1.0: 5 ternary trits mapped to Base-3 values 0-242, padded with ternary-0 (mapped from value 1) to fill the last byte. FP16 per-tensor scales are stored as 2-byte prefixes.

2. **ATLAS file format**: Binary file with a 64-byte header (magic "ATLAS", layer count, hidden dim, intermediate dim, head counts, vocab size, tensor count). Followed by a directory of 12-byte entries (ttype, file offset, row dim, packed cols). Tensor data follows: TQ1 weights (ttype=0) have a 2-byte FP16 scale prefix followed by packed rows; embeddings/norms (ttype=1) and LM head (ttype=2) are raw FP16.

3. **DLL** (`atlas_api.cpp` / `atlas_kernel.hpp`): Loads the atlas file into memory. Provides `atlas_matmul_f32` — a scalar LUT-based matrix multiply. For each output row, it iterates over packed bytes, uses 5 precomputed lookup tables (one per trit position in the byte) to decode ternary values inline, and accumulates into float32. OpenMP parallelizes over rows. Also provides RMSNorm and RoPE kernels.

4. **Python inference** (`atlas_infer.py`): Coordinates the full LlamaForCausalLM forward pass. Activation quantization (per-token int8 round-trip via max-abs scaling). GQA attention with KV cache. SiLU-gated FFN. Autoregressive loop with top-1 or temperature sampling. All heavy matmuls call into the DLL; norms, attention, and RoPE run in NumPy.

The matmul kernel uses OpenMP row-level parallelism. The prebuilt DLL is compiled with `-fopenmp` and achieves ~5× speedup on 8 cores versus single-threaded. The `libomp.dll` runtime must be available at load time.

## Performance

Measured on i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA, 16 GB DDR4).

| Model | Single Projection (down_proj) | Full Decode |
|-------|------------------------------|-------------|
| Falcon3-7B (28L, 3072x23040) | ~120 tok/s (OMP, 8 threads) | ~0.7 s/tok (~1.4 tok/s) |
| Falcon3-10B (40L, 3072x23040) | ~175 tok/s (OMP, 8 threads) | ~1.0 s/tok (~1.0 tok/s) |

Decode throughput improves ~3× with OpenMP (measured `down_proj`: 31ms → 5.8ms on 23040×614 packed matmul). Per-token time includes 7 TQ1 matmuls per layer × 40 layers plus Python overhead (attention einsum, RMSNorm, KV cache). Prefill (prompt processing) is batch-parallel and achieves similar per-token throughput. The scalar LUT kernel was chosen over AVX2 gather (4× slower on Alder Lake due to gather latency).

## Bugfix Chronology

Three critical bugs were discovered and fixed during development. Any one of them would cause the model to produce garbage output (correlation near zero with reference activations).

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

### Verification

After all three fixes, all layer-0 TQ1 projections match the HuggingFace reference with correlation > 0.999 (gate=0.999932, up=0.999928, down=0.999831). The small remaining difference is from float16 accumulation in the reference vs float32 in ATLAS.

## File Reference

| File | Purpose |
|------|---------|
| `atlas_packer.py` | Streams safetensors → TQ1 ATLAS file |
| `atlas_infer.py` | End-to-end inference engine (Python) |
| `atlas_api.cpp` | C-exported DLL (load, matmul, norm, rope) |
| `atlas_kernel.hpp` | Header-only LUT + matmul kernels |
| `atlas.dll` | Prebuilt DLL (Clang LLVM-MinGW, 80 KB) |
| `compile.bat` | DLL build script |
| `test_direct_cmp.py` | Atlas file vs HF ternary values (`python test_direct_cmp.py model.atlas model.safetensors`) |
| `test_kernel_row.py` | Single-row kernel vs pure Python |
| `test_row_order.py` | Row reordering demonstration (corr 0.006 vs 0.999) |

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
