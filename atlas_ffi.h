// atlas_ffi.h — Pure C API contract for ATLAS TQ1.0 inference engine
// v1.0.2 — Single source of truth for FFI consumers (Mojo, Rust, Zig, Go, C)
//
// File format: ATLAS TQ1.0 (.atlas)
//   [0:5]   "ATLAS" magic
//   [5:2]   uint16 version
//   [7:2]   uint16 n_layers
//   [9:2]   uint16 hidden_dim
//   [11:2]  uint16 inter_dim
//   [13:1]  uint8  n_heads
//   [14:1]  uint8  n_kv_heads
//   [15:2]  uint16 head_dim
//   [17:4]  uint32 vocab_size (int32)
//   [21:8]  float64 rope_theta (version>=3), else 10000.0
//   [29:4]  int32  tokenizer_size (v5+), 0 if no embedded tokenizer
//   [33:4]  uint32 tokenizer_offset (v5+), absolute file offset
//   [56:4]  int32  name_block_size (v4+)
//   [60:4]  int32  n_tensors
//   [64:]   tensor directory: n × [ttype:1][file_offset:4][row_dim:4][packed_cols:3]
//   then name block, then tensor data at each directory file_offset
//   (v5+: tokenizer data appended after tensor data, at tokenizer_offset)
//
// Int8 cache format (.i8 companion file)
//   [0:4]     int32 n_tensors (must match atlas n_tensors)
//   [4:]      n × [ttype:1][row_dim:4][pc:4][data_size:4][offset:8]
//   [4+n*21:] concatenated tensor data (only ttype==3 int8-decoded tensors)
//
// Tensor types (ttype):
//   0 = TQ1 packed (Base-3, 5 trits/byte), 2-byte scale prefix + packed bytes
//   1 = float16 vector (norm) or matrix (embed)— 2 bytes per element
//   2 = float16 matrix (lm_head) or vector (GQA scales)
//   3 = int8-decoded (after decompress or mmap cache load). [2:scale_fp16][rows×dim:i8][rows:row_sums_i32]
//
// All float16 values use IEEE 754 binary16.

#ifndef ATLAS_FFI_H
#define ATLAS_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #ifdef ATLAS_BUILD_DLL
    #define ATLAS_API __declspec(dllexport)
  #else
    #define ATLAS_API __declspec(dllimport)
  #endif
#else
  #define ATLAS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config ─────────────────────────────────────────────────────
typedef struct {
    int n_layers;
    int hidden_dim;
    int inter_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
} AtlasModelConfig;

// ─── Lifecycle ────────────────────────────────────────────────────────
// Load model from .atlas file. Returns opaque pointer or NULL on error.
// Memory: allocates all tensor data via VirtualAlloc (~2-3 GB for 10B).
ATLAS_API void* atlas_load(const char* path);

// Free all model resources. Must be called exactly once per atlas_load.
// Frees tensor data (unless mmap'd), unmaps cache, deletes model struct.
ATLAS_API void atlas_free(void* model);

// ─── Config query ─────────────────────────────────────────────────────
// Get all model parameters in one call. No allocation needed.
ATLAS_API AtlasModelConfig atlas_get_config(void* model);

// Legacy multi-param getter (deprecated — use atlas_get_config instead).
ATLAS_API void atlas_get_info(void* model, int* n_layers, int* hidden_dim,
                              int* inter_dim, int* n_heads, int* n_kv_heads,
                              int* head_dim, int* vocab_size);

// ─── Tensor name API (v4+, no safetensors dependency) ─────────────────
// Get number of tensors in the model. Returns 0 if names not loaded (v3).
ATLAS_API int atlas_get_tensor_count(void* model);

// Get tensor name at index. Returns chars written (excluding \0). 0 if error.
// buf_size should be at least 256.
ATLAS_API int atlas_get_tensor_name(void* model, int idx, char* buf, int buf_size);

// Find tensor index by exact name match. Returns -1 if not found.
ATLAS_API int atlas_get_tensor_index(void* model, const char* name);

// ─── Embedded tokenizer (v5+) ──────────────────────────────────────────
// Get embedded tokenizer data (JSON). Returns pointer to raw bytes or NULL.
// size: set to tokenizer length in bytes, or 0 if not available.
// Pointer is valid until atlas_free. Data is mmap'd from atlas file.
ATLAS_API const uint8_t* atlas_get_tokenizer(void* model, int* size);

// ─── Decompression + cache ────────────────────────────────────────────
// Decompress all TQ1 tensors (ttype==0) to int8 (ttype==3) in-place.
// Frees packed TQ1 data after decompression. Call once after safetensors loaded.
ATLAS_API void atlas_decompress_all(void* model);

// Save decompressed int8 tensors to .i8 companion file. Safe to call
// multiple times (overwrites). Writes in 64KB chunks for Windows compat.
ATLAS_API void atlas_save_cache(void* model, const char* atlas_path);

// Load int8 cache file via mmap. Returns 1 on success, 0 if not found or corrupt.
// Replaces matching ttype==0 tensors with mmap'd ttype==3 pointers.
ATLAS_API int atlas_load_cache(void* model, const char* atlas_path);

// Prefetch all int8 data into physical RAM. Touches one byte per 4KB page.
// Call after decompress or cache load to prevent pagefault stalls.
ATLAS_API void atlas_prefetch_int8(void* model);

// Set full-precision matmul mode (no activation quantization).
// Enable for small models (1B) where u8 quantization degrades coherence.
ATLAS_API void atlas_set_use_f32_matmul(void* model, int val);

// ─── Tensor access ────────────────────────────────────────────────────
// Get tensor metadata: type, row_dim, col_dim (=packed_cols*5 for TQ1, 0 otherwise).
ATLAS_API void atlas_tensor_info(void* model, int idx, int* ttype,
                                  int* row_dim, int* col_dim);

// Get raw tensor data pointer and size in bytes. View is valid until atlas_free.
ATLAS_API const uint8_t* atlas_tensor_data(void* model, int idx, int* size);

// Get int8-decoded tensor for direct access. Returns NULL if not ttype==3.
// scale: float32 dequant multiplier   row_sums: [rows] int32 Σ w_i per row
ATLAS_API const int8_t* atlas_get_int8(void* model, int idx, int* rows,
                                        int* input_dim, float* scale,
                                        const int32_t** row_sums);

// ─── Inference ────────────────────────────────────────────────────────
// Forward ALL transformer layers (RMSNorm + QKV + attention + FFN), fused.
// hidden_states: [B × hidden_dim] float32 — read from, overwritten with final layer output
// positions: [B] int32 — absolute position indices for RoPE
// layer_idx: [n_layers × 9] int32 — flat tensor indices per layer:
//   (ln1, q, k, v, o, ln2, gate, up, down) repeated for each layer
// k_cache, v_cache: [n_layers × n_kv_heads × max_seq_len × head_dim] uint16 (fp16)
// seq_now: current sequence length (positions will be < seq_now for decode)
//
// Final RMSNorm + LM head matmul should be applied in Python (numpy/MKL is faster).
// Memory: buffers allocated internally via valloc (mmap/VirtualAlloc).
//   ~4 × B × max(inter_dim, hidden_dim, n_heads*head_dim) × sizeof(float)
ATLAS_API void atlas_forward(void* model,
    float* hidden_states, int B,
    const int* positions,
    uint16_t* k_cache, uint16_t* v_cache,
    int max_seq_len, int seq_now,
    const int* layer_idx, int n_layers);

// ─── v1.2.0: Sampling + generation ────────────────────────────────────
// Seed the internal Xoshiro256** PRNG. Call once before generation.
ATLAS_API void atlas_set_seed(uint64_t seed);

// Sample one token from logits using Gumbel-max with top-k/top-p.
// logits: [vocab_size] float32 — modified in-place (used as scratch).
// output: [1] int32 — receives the sampled token ID.
ATLAS_API void atlas_sample(void* model, float* logits, int* output,
                             float temperature, int top_k, float top_p);

// End-to-end autoregressive generation. Single C call for the entire decode loop.
// Allocates internal scratch buffers (embedding, norm, logits). KV cache must be
// pre-allocated by the caller.
//
// input_ids:  [n_input] int32 — tokenized prompt IDs
// k/v_cache:  [n_layers × n_kv_heads × max_seq_len × head_dim] uint16 (zero-filled or reused)
// output_ids: [max_new_tokens] int32 — receives generated token IDs
//
// Returns: number of tokens actually generated ( ≤ max_new_tokens ), or -1 on error.
// Stops when EOS token (id=0) is produced or max_new_tokens is reached.
ATLAS_API int atlas_generate(void* model,
    const int* input_ids, int n_input,
    uint16_t* k_cache, uint16_t* v_cache,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    int* output_ids);

// ─── Int8 lm_head ─────────────────────────────────────────────────────
// Quantize lm_head from fp16 to per-row symmetric int8 (~403 MB vs 1.5 GB fp32).
// idx: tensor index of lm_head in the model's tensor list.
// Frees the original fp16 data. Call after Python has created its fp32 copy.
ATLAS_API void atlas_quantize_lmhead(void* model, int idx);

// GEMV: B tokens [B × hidden_dim] → logits [B × vocab_size] via int8 lm_head.
// AVX2 maddubs + offset trick + per-row dequant. OpenMP parallel over vocab.
ATLAS_API void atlas_lmhead_gemv(void* model, const float* act,
                                  float* output, int B);

// ─── Int8 matmul (AVX2 maddubs) ──────────────────────────────────────
// output[t][r] = sum_k act_u8[t][k] * w[r][k] - 128 * row_sums[r]
// act_u8: uint8 (= int8 activation + 128 offset)
// output: raw int32 result (needs dequant: out * max_abs[t] / (127 * scale))
ATLAS_API void atlas_matmul_i8_f32(int rows, int input_dim,
    const int8_t* weights, const uint8_t* act_u8,
    const int32_t* row_sums, float* output,
    int n_tokens);

// ─── Norms + positional encoding ─────────────────────────────────────
// RMSNorm: output[i] = x[i] * w[i] * rms(mean(x²) + eps)^{-1}
ATLAS_API void atlas_rmsnorm_f32(const float* x, const uint8_t* weight_f16,
                                  float* output, int n, float eps);

// RoPE: apply rotary embeddings to Q and K in-place for a single position
ATLAS_API void atlas_rope_f32(float* q, float* k, int n_heads, int n_kv_heads,
                               int head_dim, int position, float rope_theta);

// ─── Fused attention (RoPE + GQA + softmax + weighted sum) ───────────
// q, k, v: [B × n_heads*head_dim / n_kv_heads*head_dim] float32
//   q: RoPE applied in-place;  k, v: read only (cache used for attn)
// k_cache, v_cache: [n_kv_heads × max_seq × head_dim] uint16 (fp16)
// output: [B × n_heads × head_dim] float32
//
// Updates k_cache, v_cache at given positions with fp16 values.
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    const uint16_t* k_cache, const uint16_t* v_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float* output);

#ifdef __cplusplus
}
#endif

#endif // ATLAS_FFI_H
