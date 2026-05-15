# ATLAS — Falcon3-7B TQ1.0 Inference Engine

## Goal
- Complete TQ1.0 (Base-3, 5 ternary trits/byte) inference engine for Falcon3-7B on i5 laptop, end-to-end text generation via C++ DLL + Python.

## Constraints & Preferences
- 16 GB RAM only (no GPU); Falcon3-7B safetensors = 3.05 GB, atlas file = 2.74 GB.
- Target CPU: i5-1235U (Alder Lake, 8 OMP threads, AVX2+FMA).
- Compiler: Clang 22.1.5 (LLVM MinGW x86_64-w64-windows-gnu).
- Base-3 encoding: byte V = m0 + m1·3 + m2·9 + m3·27 + m4·81 (m_i∈{0,1,2}→ternary{-1,0,1}), V∈[0,242].
- Row-aware packing: each BitNet uint8 row holds 4 interleaved weight rows; de-interleave before TQ1 repack.
- Per-tensor FP16 scale from `weight_scale` safetensors, stored as 2-byte prefix.
- 12-byte tensor directory: [ttype:1][offset:4][row_dim:4][packed_per_row:3]; n_tensors at byte 60.
- 5 separate LUTs (float[256], 32-byte aligned).
- Matmul outputs row-major: `output[t * rows + r]`.
- Model: `LlamaForCausalLM` with `BitLinear` wrappers, `hidden_size=3072`, `intermediate_size=23040`, `num_attention_heads=12`, `num_key_value_heads=4`, `head_dim=256`, `rope_theta=1000042`, `rms_norm_eps=1e-6`.
- Safetensors weights are pre-quantized: **2-bit packing** (`byte = v0 + v1·4 + v2·16 + v3·64`, v_i∈{0,1,2,3}) for `nn.Linear` layers; `bfloat16` for `lm_head.weight`, `embed_tokens.weight`, norms.

## Progress
### Done
- Packed Falcon3-7B into `C:\atlas\falcon3-7b-tq1.atlas` (2.74 GB, 451 tensors, 196 TQ1 weight tensors).
- Wrote `atlas_kernel.hpp` (LUT, TQ1Mat, scalar unrolled-4x matmul).
- Wrote `atlas_api.cpp` (C-exported DLL: load, matmul, rmsnorm, rope).
- Compiled `atlas_bench.exe` (scalar kernel); benchmarked down_proj [3072×23040]: 165 tok/s peak.
- Wrote `atlas_infer.py` (AtlasModel class with GQA attention, RoPE, KV cache, SiLU FFN).
- Fixed DLL runtime by copying `libc++.dll`, `libunwind.dll`, `libomp.dll` from MinGW bin/ into C:\atlas.
- Fixed `file_offset` overflow (int→uint32_t in TensorInfo).
- Fixed `_load_tq1` ctypes pointer arithmetic (`addressof(ptr.contents)+2` instead of `ptr+2`).
- Fixed matmul activation padding (pad to `packed_cols * 5` when input dim < padded dim).
- Fixed GQA attention — working einsum shapes and KV cache indexing.
- Fixed RMSNorm weight pointer (`_load_weight_f16` returned float32; DLL expected raw float16 bytes).
- Added activation quantization (`_matmul_tq1` BitNet round-trip: quantize→matmul→dequantize with per-token `max_abs`).
- **Fixed `fseek` 32-bit overflow**: replaced with `_fseeki64(f, (int64_t)offset, SEEK_SET)` via `FSEEK` macro.
- **Recompiled atlas.dll** with fseek fix; all 28 layers now complete without NaN.
- **Added `_matmul_f16` method** for non-TQ1 float16 tensors (LM head `ttype=2`).
- **HF reference model runs** via `torch._dynamo.config.suppress_errors=True` (TorchInductor Windows bug suppressed). Reference top-1 for BOS: token 12 (`'\n'`); L0 RMS=0.34; L28 RMS=1.63; logits RMS=3.45.
- **Fixed packer 2-bit packing bug**: Safetensors use 2-bit packing (`byte = v0 + v1·4 + v2·16 + v3·64`), but packer was using Base-3 decoding (`%3, //3`). Fixed in `atlas_packer.py`.
- **Re-packed atlas** with 2-bit fix; ternary values now match HF reference exactly.
- **ROOT CAUSE FIX: Row ordering bug**. Atlas stores TQ1 rows in interleaved order (`ur*4+q`), but HF `unpack_weights` produces output in stride order (`q*5760+ur`). After C++ matmul, output must be reordered. Without this fix, corr≈0 despite correct ternary values. Fix: `out.reshape(batch, rows_packed, 4).transpose(0, 2, 1).reshape(batch, rows)`.
- **Verified**: All L0 TQ1 projections match reference with corr > 0.999 (gate=0.999932, up=0.999928, down=0.999831). Small remaining diff from float16 accumulation vs float32.
- **`save_ref_activations.py`**: Saves reference model activations via forward hooks (256 key+value tensors, 8.6 MB pickle).
- **`test_direct_cmp.py`**: Direct atlas file vs HF ternary value comparison without loading atlas model (avoids pagefile crash).
- **`test_pure_python.py`**: Pure NumPy matmul comparison (may be incomplete due to memory).
- **`test_kernel_row.py`**: Single-row kernel vs pure Python comparison (confirmed row 0 matches).
- **`test_row_order.py`**: Demonstrates corr≈0 vs corr≈1 with vs without row reordering.

### In Progress
- **Full 28-layer forward loop**: Need to implement GQA attention (RoPE, scaled_dot_product, KV cache), residual connections, and SiLU FFN in Python to complete end-to-end inference. Individual TQ1 projections verified corr > 0.999 with correct inputs.

### Blocked
- **(none)**

## Key Decisions
- **Row ordering fix is CRITICAL**: `atlas_matmul_f32` outputs in atlas order (`ur*4+q`). Must reorder to HF order (`q*rows_packed+ur`). This was the root cause of all "corr≈0 but RMS≈1" bugs.
- **Divide by scale is correct**: `out = raw_sum * max_abs / (127 * scale)` matches HF `post_quant_process(y, weight_scale, input_scale)`.
- **2-bit NOT Base-3 in safetensors**: Original packer incorrectly used Base-3 decoding. Fixed to 2-bit unpack.
- **Scalar over AVX2 gather**: Gather was 4× slower (40 vs 162 tok/s) on Alder Lake.
- **5 separate LUT arrays** (tq1_lut0..tq1_lut4) for register-based per-position access.
- **No reverse-scaling in LUT**: Packer writes raw Base-3 (0..242), LUT reads `temp = b` directly.
- **Hybrid Python+C++**: Python handles norm, attention, RoPE; C++ DLL runs heavy TQ1 matmuls.
- **KV cache**: 4096 tokens × float16 (≈0.94 GB for K+V).

## Next Steps
1. **Complete full inference loop**: Implement GQA attention (RoPE, causal mask, KV cache), SiLU FFN, and residual connections in `atlas_infer.py` using fixed `_matmul_tq1`.
2. **Test first token**: Run BOS → one generated token, compare token ID with reference (expected: 12).
3. **End-to-end generation**: Implement temperature sampling, top-k, autoregressive loop with KV cache.
4. **Benchmark**: Measure tok/s for generation.
5. **Optionally optimize**: Profile C++ kernel, add AVX2 path if needed.

## Critical Context
- Atlas weight scales are **inverse BitNet α** (27–67), not `mean(|W_full|)` (0.01–0.3). Divide is correct; multiply is wrong.
- **`fseek` 32-bit overflow** was the critical hidden bug corrupting all tensors >2 GB offset. Fixed by replacing with `_fseeki64`.
- **Row ordering** was the SECOND critical bug: `out[batch, q*rows_packed+ur] = out_kernel[batch, ur*4+q]`.
- With both fixes: all L0 TQ1 projections match reference with corr > 0.999.
- The reference model wraps weights in `BitLinear` module. Forward: `activation_quant` (int8) → `F.linear(input_quant.to(dtype), unpacked_weights)` → `post_quant_process(y, weight_scale, input_scale)`. The `post_quant_process` is `@torch.compile` decorated (source of TorchInductor crash).
- HF reference output was obtained via `torch._dynamo.config.suppress_errors = True` (skips TorchInductor-induced crash on Windows).
- Pagefile limit prevents loading both atlas model + safetensors simultaneously (~6 GB total). Memory-efficient scripts (`test_direct_cmp.py`) avoid atlas model load by reading atlas file directly at byte offsets.

## Relevant Files
- `C:\atlas\atlas_packer.py`: Packer — 2-bit unpacking (lines 25–30). Run to re-pack atlas.
- `C:\atlas\atlas_infer.py`: Python inference engine — `_matmul_f16` for LM head, `_matmul_tq1` with activation quant + row reordering.
- `C:\atlas\atlas_api.cpp`: C-exported DLL — fseek fix applied (`FSEEK` macro).
- `C:\atlas\atlas_kernel.hpp`: Header-only TQ1 matmul kernels (unused if using atlas_api.cpp directly).
- `C:\atlas\falcon3-7b-tq1.atlas`: Packed model (2.74 GB, re-packed with 2-bit fix).
- `C:\atlas\ref_activations.pkl`: Saved reference model activations (256 tensors).
- `C:\atlas\diagnose_calibrate.py`: Layer-0 kernel injection + per-row scale comparison.
- `C:\atlas\test_direct_cmp.py`: Direct atlas file vs HF ternary value comparison (minimal memory).
- `C:\atlas\test_kernel_row.py`: Single-row kernel vs pure Python comparison.
- `C:\atlas\test_row_order.py`: Demonstrates row ordering fix (corr 0.006 → 0.999).
- `C:\models\Falcon3-7B-Instruct-1.58bit\`: Original safetensors + config + tokenizer.
