// atlas_api.cpp — C-exported DLL for TQ1.0 inference acceleration
#define ATLAS_API __declspec(dllexport)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>
#include <io.h>
#include <immintrin.h>
#include <windows.h>

// VirtualAlloc-based allocator for large tensor data — memory is returned to OS on free
static uint8_t* valloc(size_t size) {
    return (uint8_t*)VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
static void vfree(uint8_t* ptr) {
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
}

#ifdef _WIN32
  #define FSEEK _fseeki64
  #define FTELL _ftelli64
#else
  #define FSEEK fseeko
  #define FTELL ftello
#endif

extern "C" {

// ─── LUT ───────────────────────────────────────────────────────────────
static bool lut_initialized = false;
alignas(32) static float tq1_lut0[256];
alignas(32) static float tq1_lut1[256];
alignas(32) static float tq1_lut2[256];
alignas(32) static float tq1_lut3[256];
alignas(32) static float tq1_lut4[256];

static void ensure_lut() {
    if (lut_initialized) return;
    for (int b = 0; b < 256; ++b) {
        int t = b;
        tq1_lut0[b] = (float)((t % 3) - 1); t /= 3;
        tq1_lut1[b] = (float)((t % 3) - 1); t /= 3;
        tq1_lut2[b] = (float)((t % 3) - 1); t /= 3;
        tq1_lut3[b] = (float)((t % 3) - 1); t /= 3;
        tq1_lut4[b] = (float)((t % 3) - 1);
    }
    lut_initialized = true;
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t f32 = (e == 0) ? ((s << 31) | (0 << 23) | (m << 13))
                : (e == 31) ? ((s << 31) | (0xFF << 23) | (m << 13))
                : ((s << 31) | ((e + 112) << 23) | (m << 13));
    float r; memcpy(&r, &f32, 4); return r;
}

// ─── Tensor info ────────────────────────────────────────────────────────
struct TensorInfo {
    int ttype;          // 0=TQ1, 1=norm/embed, 2=other
    int row_dim;        // output rows (weight) or flat size (norm/embed)
    int packed_cols;    // packed bytes per row (0 for non-TQ1)
    uint32_t file_offset;
    uint8_t* data;      // loaded data (scale prefix + packed for TQ1, raw fp16 for others)
    int data_size;
};

struct AtlasModel {
    int n_layers;
    int hidden_dim;
    int inter_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
    std::vector<TensorInfo> tensors;

    // Pre-allocated scratch buffers for forward_layer (lazy init)
    float* buf_gate = nullptr;      // [max_batch * inter_dim]
    float* buf_up = nullptr;        // [max_batch * inter_dim]
    float* buf_hidden = nullptr;    // [max_batch * inter_dim]
    float* buf_act = nullptr;       // [max_batch * max_dim] quantized f32→i8 scratch
    uint8_t* buf_i8 = nullptr;      // [max_batch * max_dim] uint8 quantized activations
    int max_batch = 0;
    // Debug snapshots
    float* snap_q = nullptr;
    float* snap_k = nullptr;
    float* snap_v = nullptr;
    float* snap_o = nullptr;
    float* snap_norm1 = nullptr;

    ~AtlasModel() {
        if (buf_gate) VirtualFree(buf_gate, 0, MEM_RELEASE);
        if (buf_up) VirtualFree(buf_up, 0, MEM_RELEASE);
        if (buf_hidden) VirtualFree(buf_hidden, 0, MEM_RELEASE);
        if (buf_act) VirtualFree(buf_act, 0, MEM_RELEASE);
        if (buf_i8) VirtualFree(buf_i8, 0, MEM_RELEASE);
        if (snap_q) VirtualFree(snap_q, 0, MEM_RELEASE);
        if (snap_k) VirtualFree(snap_k, 0, MEM_RELEASE);
        if (snap_v) VirtualFree(snap_v, 0, MEM_RELEASE);
        if (snap_o) VirtualFree(snap_o, 0, MEM_RELEASE);
        if (snap_norm1) VirtualFree(snap_norm1, 0, MEM_RELEASE);
    }

    void ensure_buffers(int B) {
        if (B <= max_batch) return;
        if (buf_gate) VirtualFree(buf_gate, 0, MEM_RELEASE);
        if (buf_up) VirtualFree(buf_up, 0, MEM_RELEASE);
        if (buf_hidden) VirtualFree(buf_hidden, 0, MEM_RELEASE);
        if (buf_act) VirtualFree(buf_act, 0, MEM_RELEASE);
        if (buf_i8) VirtualFree(buf_i8, 0, MEM_RELEASE);

        int max_dim = inter_dim > hidden_dim ? inter_dim : hidden_dim;
        if (n_heads * head_dim > max_dim) max_dim = n_heads * head_dim;
        int max_aligned = ((max_dim + 7) + 31) & ~31;  // +7 for TQ1 padding (packed_cols*5 up to dim+4)

        buf_gate = (float*)VirtualAlloc(NULL, (size_t)B * inter_dim * sizeof(float), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        buf_up = (float*)VirtualAlloc(NULL, (size_t)B * inter_dim * sizeof(float), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        buf_hidden = (float*)VirtualAlloc(NULL, (size_t)B * inter_dim * sizeof(float), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        buf_act = (float*)VirtualAlloc(NULL, (size_t)B * max_aligned * sizeof(float), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        buf_i8 = (uint8_t*)VirtualAlloc(NULL, (size_t)B * max_aligned * sizeof(uint8_t), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        auto alloc_snap = [&](float*& ptr) {
            if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
            ptr = (float*)VirtualAlloc(NULL, (size_t)B * hidden_dim * sizeof(float), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        };
        alloc_snap(snap_q); alloc_snap(snap_k); alloc_snap(snap_v);
        alloc_snap(snap_o); alloc_snap(snap_norm1);
        max_batch = B;
    }
};

// ─── Load model ─────────────────────────────────────────────────────────
ATLAS_API AtlasModel* atlas_load(const char* path) {
    ensure_lut();
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ATLAS] Cannot open %s\n", path); return nullptr; }

    uint8_t hdr[64];
    if (fread(hdr, 1, 64, f) != 64) { fclose(f); return nullptr; }
    if (memcmp(hdr, "ATLAS", 5) != 0) { fclose(f); return nullptr; }

    AtlasModel* m = new AtlasModel();
    uint16_t tmp; memcpy(&tmp, hdr+7, 2); m->n_layers = tmp;
    memcpy(&tmp, hdr+9, 2); m->hidden_dim = tmp;
    memcpy(&tmp, hdr+11, 2); m->inter_dim = tmp;
    m->n_heads = hdr[13]; m->n_kv_heads = hdr[14];
    memcpy(&tmp, hdr+15, 2); m->head_dim = tmp;
    uint32_t tmp32; memcpy(&tmp32, hdr+17, 4); m->vocab_size = (int)tmp32;
    int n_tensors; memcpy(&n_tensors, hdr+60, 4);
    uint16_t version; memcpy(&version, hdr+5, 2);
    if (version >= 3) {
        double rt; memcpy(&rt, hdr+21, 8); m->rope_theta = (float)rt;
    } else {
        m->rope_theta = 10000.0f;  // default for old files
    }

    printf("[ATLAS] v%d model: %dL %dH %dI %d/%d heads %d vocab %.0f theta | %d tensors\n",
           version,
           m->n_layers, m->hidden_dim, m->inter_dim, m->n_heads, m->n_kv_heads,
           m->vocab_size, m->rope_theta, n_tensors);

    // Read directory
    m->tensors.resize(n_tensors);
    FSEEK(f, 64, SEEK_SET);
    for (int i = 0; i < n_tensors; i++) {
        uint8_t e[12]; fread(e, 1, 12, f);
        m->tensors[i].ttype = e[0];
        uint32_t off; memcpy(&off, e+1, 4); m->tensors[i].file_offset = off;
        memcpy(&m->tensors[i].row_dim, e+5, 4);
        m->tensors[i].packed_cols = e[9] | (e[10]<<8) | (e[11]<<16);
    }

    // Load all tensor data
    for (int i = 0; i < n_tensors; i++) {
        auto& t = m->tensors[i];
        FSEEK(f, (int64_t)t.file_offset, SEEK_SET);

        if (t.ttype == 0) {  // TQ1: 2-byte scale + packed data
            t.data_size = 2 + t.row_dim * t.packed_cols;
        } else if (t.ttype == 1) {  // norm/embed: raw float16
            // embed: [vocab, hidden]; norm: [hidden]
            if (t.row_dim == m->vocab_size) {
                t.data_size = t.row_dim * m->hidden_dim * 2;  // fp16
            } else {
                t.data_size = t.row_dim * 2;  // fp16 1D
            }
            t.packed_cols = 0;
        } else {  // other (lm_head)
            t.data_size = t.row_dim * m->hidden_dim * 2;
            t.packed_cols = 0;
        }

        t.data = valloc(t.data_size);
        fread(t.data, 1, t.data_size, f);
    }

    fclose(f);
    auto& last = m->tensors.back();
    int64_t total = last.file_offset + last.data_size;
    printf("[ATLAS] Loaded %d tensors (%.2f MB)\n", n_tensors, total / 1048576.0);
    return m;
}

// ─── Free model ─────────────────────────────────────────────────────────
ATLAS_API void atlas_free(AtlasModel* m) {
    if (!m) return;
    for (auto& t : m->tensors) vfree(t.data);
    delete m;
}

// ─── Get model info ────────────────────────────────────────────────────
ATLAS_API void atlas_get_info(AtlasModel* m, int* n_layers, int* hidden_dim,
                               int* inter_dim, int* n_heads, int* n_kv_heads,
                               int* head_dim, int* vocab_size) {
    if (n_layers) *n_layers = m->n_layers;
    if (hidden_dim) *hidden_dim = m->hidden_dim;
    if (inter_dim) *inter_dim = m->inter_dim;
    if (n_heads) *n_heads = m->n_heads;
    if (n_kv_heads) *n_kv_heads = m->n_kv_heads;
    if (head_dim) *head_dim = m->head_dim;
    if (vocab_size) *vocab_size = m->vocab_size;
}

// ─── Decompress all TQ1 tensors to int8 in-place ──────────────────────
// Frees packed data after decompression. Call after Python closes safetensors.
ATLAS_API void atlas_decompress_all(AtlasModel* m) {
    int total = 0;
    for (auto& t : m->tensors) {
        if (t.ttype != 0) continue;
        total++;

        int input_dim = t.packed_cols * 5;
        int n_vals = t.row_dim * input_dim;

        // Allocate: [scale_fp16(2)] [i8_data(rows × input_dim)] [row_sums(rows × 4)]
        uint8_t* new_data = valloc(2 + n_vals + t.row_dim * 4);
        new_data[0] = t.data[0];
        new_data[1] = t.data[1];

        int8_t* i8 = (int8_t*)(new_data + 2);
        int32_t* rs = (int32_t*)(i8 + n_vals);
        const uint8_t* packed = t.data + 2;

        for (int r = 0; r < t.row_dim; r++) {
            int sum = 0;
            int pos = 0;
            for (int c = 0; c < t.packed_cols; c++) {
                uint8_t b = packed[r * t.packed_cols + c];
                int v = (b % 3) - 1; b /= 3; i8[r * input_dim + pos++] = (int8_t)v; sum += v;
                v = (b % 3) - 1; b /= 3; i8[r * input_dim + pos++] = (int8_t)v; sum += v;
                v = (b % 3) - 1; b /= 3; i8[r * input_dim + pos++] = (int8_t)v; sum += v;
                v = (b % 3) - 1; b /= 3; i8[r * input_dim + pos++] = (int8_t)v; sum += v;
                v = (b % 3) - 1;        i8[r * input_dim + pos++] = (int8_t)v; sum += v;
            }
            rs[r] = sum;
        }

        // Free packed data, replace with int8
        vfree(t.data);
        t.data = new_data;
        t.data_size = 2 + n_vals + t.row_dim * 4;
        t.ttype = 3;  // int8-decoded
    }
    printf("[ATLAS] Decompressed %d TQ1 tensors to int8\n", total);
}

// ─── Prefetch int8 data into physical RAM ─────────────────────────────
// Touch one byte per 4KB page to force page-in / prevent working-set trim lag.
ATLAS_API void atlas_prefetch_int8(AtlasModel* m) {
    int64_t total = 0;
    for (auto& t : m->tensors) {
        if (t.ttype != 3) continue;
        int n_vals = t.row_dim * t.packed_cols * 5;
        int8_t* data = (int8_t*)(t.data + 2);
        volatile int acc = 0;
        for (int64_t i = 0; i < n_vals; i += 4096 / sizeof(int8_t)) {
            acc += data[i];
        }
        total += n_vals;
        (void)acc;
    }
    printf("[ATLAS] Prefetched %lld int8 values\n", (long long)total);
}

// ─── Get int8-decoded tensor from C++ side ─────────────────────────────
// Returns i8 data pointer or nullptr if not decoded
ATLAS_API const int8_t* atlas_get_int8(AtlasModel* m, int idx, int* rows,
                                        int* input_dim, float* scale,
                                        const int32_t** row_sums) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return nullptr;
    auto& t = m->tensors[idx];
    if (t.ttype != 3) return nullptr;

    uint16_t scale_raw;
    memcpy(&scale_raw, t.data, 2);
    if (scale) *scale = fp16_to_fp32(scale_raw);
    if (rows) *rows = t.row_dim;
    if (input_dim) *input_dim = t.packed_cols * 5;
    if (row_sums) {
        int n_vals = t.row_dim * t.packed_cols * 5;
        *row_sums = (const int32_t*)(t.data + 2 + n_vals);
    }
    return (const int8_t*)(t.data + 2);
}

// ─── Get tensor info ────────────────────────────────────────────────────
ATLAS_API void atlas_tensor_info(AtlasModel* m, int idx, int* ttype,
                                  int* row_dim, int* col_dim) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return;
    auto& t = m->tensors[idx];
    if (ttype) *ttype = t.ttype;
    if (row_dim) *row_dim = t.row_dim;
    if (col_dim) *col_dim = t.ttype == 0 ? t.packed_cols * 5 : 0;
}

// ─── Access tensor data ─────────────────────────────────────────────────
ATLAS_API const uint8_t* atlas_tensor_data(AtlasModel* m, int idx, int* size) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return nullptr;
    if (size) *size = m->tensors[idx].data_size;
    return m->tensors[idx].data;
}

// ─── Find tensor by name pattern ────────────────────────────────────────
// layer_pattern: e.g. "layers.0.mlp.down_proj.weight"
// Returns tensor index or -1
ATLAS_API int atlas_find_tensor(AtlasModel* m, const char* pattern, int layer) {
    // Build the expected name prefix
    char search[128];
    snprintf(search, sizeof(search), "model.layers.%d.%s", layer, pattern);

    // Read tensor names — we don't have them stored.
    // Instead, we infer from the directory order and patterns:
    // The atlas stores tensors in alphabetical order of their safetensors names.
    // We can search by iterating and checking offsets.

    // Since names are not in the atlas, we rely on the Python side
    // to know the exact tensor indices. This function is a placeholder.
    (void)m;
    fprintf(stderr, "atlas_find_tensor: tensor names not stored in atlas\n");
    return -1;
}

// ─── Matmul: TQ1 packed → float32 output ──────────────────────────────
// Performs: output[t][r] = scale * sum_{k} unpack(weight[r][k]) · activation[t][k*5:(k+1)*5]
//
// activations: [n_tokens × packed_cols*5] float32
// output:      [n_tokens × rows] float32
// data:        packed TQ1 bytes (without scale prefix), [rows × packed_cols]
ATLAS_API void atlas_matmul_f32(int rows, int packed_cols, const uint8_t* data,
                                 const float* activations, float* output,
                                 int n_tokens, float scale) {
    const int as = packed_cols * 5;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
        const uint8_t* rd = data + r * packed_cols;
        for (int t = 0; t < n_tokens; t++) {
            const float* ab = activations + t * as;
            float sum = 0.0f;
            int k = 0;
            for (; k + 4 <= packed_cols; k += 4) {
                int base = k * 5;
                uint8_t b0 = rd[k], b1 = rd[k+1], b2 = rd[k+2], b3 = rd[k+3];
                sum += tq1_lut0[b0]*ab[base+0]+tq1_lut1[b0]*ab[base+1]
                     + tq1_lut2[b0]*ab[base+2]+tq1_lut3[b0]*ab[base+3]+tq1_lut4[b0]*ab[base+4];
                base += 5;
                sum += tq1_lut0[b1]*ab[base+0]+tq1_lut1[b1]*ab[base+1]
                     + tq1_lut2[b1]*ab[base+2]+tq1_lut3[b1]*ab[base+3]+tq1_lut4[b1]*ab[base+4];
                base += 5;
                sum += tq1_lut0[b2]*ab[base+0]+tq1_lut1[b2]*ab[base+1]
                     + tq1_lut2[b2]*ab[base+2]+tq1_lut3[b2]*ab[base+3]+tq1_lut4[b2]*ab[base+4];
                base += 5;
                sum += tq1_lut0[b3]*ab[base+0]+tq1_lut1[b3]*ab[base+1]
                     + tq1_lut2[b3]*ab[base+2]+tq1_lut3[b3]*ab[base+3]+tq1_lut4[b3]*ab[base+4];
            }
            for (; k < packed_cols; k++) {
                uint8_t b = rd[k];
                sum += tq1_lut0[b]*ab[k*5+0]+tq1_lut1[b]*ab[k*5+1]
                     + tq1_lut2[b]*ab[k*5+2]+tq1_lut3[b]*ab[k*5+3]+tq1_lut4[b]*ab[k*5+4];
            }
            output[t * rows + r] = sum / scale;
        }
    }
}

// ─── Convenience: matmul for a tensor index ────────────────────────────
// Loads scale from first 2 bytes of tensor data
ATLAS_API void atlas_tensor_matmul(AtlasModel* m, int idx,
                                    const float* activations, float* output,
                                    int n_tokens) {
    if (idx < 0 || idx >= (int)m->tensors.size()) return;
    auto& t = m->tensors[idx];
    if (t.ttype != 0) return;  // not a TQ1 tensor

    uint16_t scale_raw;
    memcpy(&scale_raw, t.data, 2);
    float scale = fp16_to_fp32(scale_raw);

    atlas_matmul_f32(t.row_dim, t.packed_cols, t.data + 2,
                     activations, output, n_tokens, scale);
}

// ─── Matmul: int8 weights × int8 activations → float32 output ──────────
// Uses _mm256_maddubs_epi16 with offset trick:
//   act_int8 ∈ [-127, 127],  act_u8 = act_int8 + 128 ∈ [1, 255]
//   w_int8 ∈ {-1, 0, 1}
//   sum(act_i * w_i) = sum((act_u8 - 128) * w_i)
//                    = sum(act_u8 * w_i) - 128 * row_sum
//   where _mm256_maddubs_epi16(act_u8, w_i8) computes sum(act_u8 * w_i)
//
// activations: [n_tokens × input_dim] int8 (IN THE RANGE [-127, 127])
// output:      [n_tokens × rows] float32
// row_sums:    [rows] int32 — precomputed Σ w_i per output row
ATLAS_API void atlas_matmul_i8_f32(int rows, int input_dim,
                                    const int8_t* weights, const uint8_t* act_u8,
                                    const int32_t* row_sums, float* output,
                                    int n_tokens) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < rows; r++) {
        const int8_t* w = weights + r * input_dim;
        int sum_w = row_sums[r];

        for (int t = 0; t < n_tokens; t++) {
            const uint8_t* a = act_u8 + t * input_dim;

            int c = 0;
            int dot = 0;

            // AVX2: 32 bytes per iteration → 16 pairs via maddubs → 8 int32 via madd
            __m256i acc = _mm256_setzero_si256();

            for (; c + 32 <= input_dim; c += 32) {
                __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                __m256i wi = _mm256_loadu_si256((const __m256i*)(w + c));
                __m256i prod16 = _mm256_maddubs_epi16(au, wi);
                __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, prod32);
            }

            // Horizontal sum of acc
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            dot = _mm_cvtsi128_si32(sum128);

            // Tail (less than 32 elements)
            for (; c < input_dim; c++) {
                dot += (int)a[c] * (int)w[c];
            }

            // Undo 128 offset: dot' = dot - 128 * Σ w_i
            int result = dot - 128 * sum_w;
            output[t * rows + r] = (float)result;
        }
    }
}

// ─── Norm: float16 tensor → RMSNorm ────────────────────────────────────
// Performs: output[i] = x[i] * weight[i] * rms(mean(x^2) + eps)
// Where weight is loaded from atlas tensor (float16)
ATLAS_API void atlas_rmsnorm_f32(const float* x, const uint8_t* weight_f16,
                                  float* output, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);

    for (int i = 0; i < n; i++) {
        uint16_t w;
        memcpy(&w, weight_f16 + i * 2, 2);
        output[i] = x[i] * rms * fp16_to_fp32(w);
    }
}

// ─── RoPE: apply rotary embeddings ──────────────────────────────────────
// Modifies q and k in-place for a single position
// q/k: [n_heads × head_dim] float32 (interleaved format)
ATLAS_API void atlas_rope_f32(float* q, float* k, int n_heads, int n_kv_heads,
                               int head_dim, int position, float rope_theta) {
    float theta_base = rope_theta;
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            float a = qh[2*i], b = qh[2*i+1];
            qh[2*i]   = a * cos_v - b * sin_v;
            qh[2*i+1] = a * sin_v + b * cos_v;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta_base, 2.0f * i / head_dim);
            float cos_v = cosf(position * freq);
            float sin_v = sinf(position * freq);
            float a = kh[2*i], b = kh[2*i+1];
            kh[2*i]   = a * cos_v - b * sin_v;
            kh[2*i+1] = a * sin_v + b * cos_v;
        }
    }
}

// ─── Fused attention: RoPE + GQA + softmax + weighted sum ────────────
// q: [B, n_heads * head_dim] float32 — RoPE applied in-place, modified
// k: [B, n_kv_heads * head_dim] float32 — RoPE applied in-place, modified
// v: [B, n_kv_heads * head_dim] float32
// positions: [B] int32
// k_cache: [n_kv_heads, max_seq, head_dim] uint16 (fp16) — updated + read
// v_cache: [n_kv_heads, max_seq, head_dim] uint16 (fp16) — updated + read
// output: [B, n_heads * head_dim] float32
ATLAS_API void atlas_attention_f32(
    float* q, float* k, float* v, const int* positions,
    const uint16_t* k_cache, const uint16_t* v_cache,
    int max_seq_len, int seq_now, int B,
    int n_heads, int n_kv_heads, int head_dim,
    float rope_theta, float* output) {

    int n_rep = n_heads / n_kv_heads;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < B; b++) {
        int pos = positions[b];
        float* qb = q + b * n_heads * head_dim;
        float* kb = k + b * n_kv_heads * head_dim;
        float* vb = v + b * n_kv_heads * head_dim;

        // RoPE on Q
        for (int h = 0; h < n_heads; h++) {
            float* qh = qb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(rope_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                float a = qh[2*i], b0 = qh[2*i+1];
                qh[2*i]   = a * c - b0 * s;
                qh[2*i+1] = a * s + b0 * c;
            }
        }
        // RoPE on K
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = kb + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(rope_theta, 2.0f * i / head_dim);
                float c = cosf(pos * freq), s = sinf(pos * freq);
                float a = kh[2*i], b0 = kh[2*i+1];
                kh[2*i]   = a * c - b0 * s;
                kh[2*i+1] = a * s + b0 * c;
            }
        }

        // Store K, V into cache (fp32 -> fp16 on the fly)
        for (int h = 0; h < n_kv_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                uint16_t* kc = (uint16_t*)k_cache + h * max_seq_len * head_dim + pos * head_dim + d;
                uint16_t* vc = (uint16_t*)v_cache + h * max_seq_len * head_dim + pos * head_dim + d;
                auto f32_to_f16 = [](float val) -> uint16_t {
                    uint32_t b; memcpy(&b, &val, 4);
                    uint16_t s = (b >> 16) & 0x8000;
                    int exp_f32 = (b >> 23) & 0xFF;
                    int exp_f16 = exp_f32 - 127 + 15;
                    if (exp_f16 <= 0) return s;
                    if (exp_f16 >= 31) return (uint16_t)(s | 0x7C00 | ((b >> 13) & 0x3FF));
                    return (uint16_t)(s | ((uint16_t)exp_f16 << 10) | ((b >> 13) & 0x3FF));
                };
                *kc = f32_to_f16(kb[h * head_dim + d]);
                *vc = f32_to_f16(vb[h * head_dim + d]);
            }
        }

        // Attention scores [n_heads, seq_now] — stack-allocate (max 12*4096=196KB, fine on 1MB stack)
        int max_seq = seq_now;
        float* scores = (float*)alloca(n_heads * max_seq * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            for (int s = 0; s < max_seq; s++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float qv = qb[h * head_dim + d];
                    uint16_t kv_raw = ((uint16_t*)k_cache)[kh * max_seq_len * head_dim + s * head_dim + d];
                    float kv = fp16_to_fp32(kv_raw);
                    sum += qv * kv;
                }
                scores[h * max_seq + s] = sum * inv_sqrt_d;
            }
        }

        // Causal mask + softmax per head
        for (int h = 0; h < n_heads; h++) {
            float* sh = scores + h * max_seq;
            float max_val = -1e9f;
            for (int s = 0; s < max_seq; s++) {
                float val = (s > pos) ? -1e9f : sh[s];
                sh[s] = val;
                if (val > max_val) max_val = val;
            }
            float sum = 0.0f;
            for (int s = 0; s < max_seq; s++) {
                float e = expf(sh[s] - max_val);
                sh[s] = e;
                sum += e;
            }
            float inv_sum = 1.0f / fmaxf(sum, 1e-10f);
            for (int s = 0; s < max_seq; s++) sh[s] *= inv_sum;
        }

        // Weighted sum: output[h, d] = sum_s scores[h, s] * v_cache[kh, s, d]
        for (int h = 0; h < n_heads; h++) {
            int kh = h / n_rep;
            float* sh = scores + h * max_seq;
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int s = 0; s < max_seq; s++) {
                    uint16_t vv_raw = ((uint16_t*)v_cache)[kh * max_seq_len * head_dim + s * head_dim + d];
                    float vv = fp16_to_fp32(vv_raw);
                    sum += sh[s] * vv;
                }
                output[b * n_heads * head_dim + h * head_dim + d] = sum;
            }
        }
    }
}

// ─── Helper: quantize float32 activations → uint8 (+128 offset) ────────
// act: [B, D] float32 → act_u8: [B, D] uint8, max_abs: [B] float32
static void quantize_f32_to_u8(const float* act, int B, int D,
                                float* max_abs_out, uint8_t* act_u8_out) {
    for (int t = 0; t < B; t++) {
        float max_val = 1e-5f;
        for (int i = 0; i < D; i++) {
            float v = fabsf(act[t * D + i]);
            if (v > max_val) max_val = v;
        }
        max_abs_out[t] = max_val;
        float inv = 127.0f / max_val;
        for (int i = 0; i < D; i++) {
            int q = (int)(act[t * D + i] * inv + 128.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            act_u8_out[t * D + i] = (uint8_t)q;
        }
    }
}

// ─── Helper: int8 matmul + reorder + dequant ──────────────────────────
// act_u8: [B, input_dim] quantized activations
// max_abs: [B] per-token max
// scratch: [B, rows] raw matmul scratch
// output: [B, rows] dequantized + reordered
static void matmul_reorder_deq(int rows, int input_dim,
    const int8_t* weights, const int32_t* row_sums,
    const uint8_t* act_u8, const float* max_abs,
    float scale, float* scratch, float* output, int B) {
    atlas_matmul_i8_f32(rows, input_dim, weights, act_u8, row_sums, scratch, B);
    int rows_packed = rows / 4;
    float deq_scale = 1.0f / (127.0f * scale);
    for (int t = 0; t < B; t++) {
        float mabs = max_abs[t];
        float* dst = output + t * rows;
        float* src = scratch + t * rows;
        for (int ur = 0; ur < rows_packed; ur++) {
            int a0 = ur * 4 + 0, a1 = ur * 4 + 1, a2 = ur * 4 + 2, a3 = ur * 4 + 3;
            int h0 = 0 * rows_packed + ur;
            int h1 = 1 * rows_packed + ur;
            int h2 = 2 * rows_packed + ur;
            int h3 = 3 * rows_packed + ur;
            dst[h0] = src[a0] * mabs * deq_scale;
            dst[h1] = src[a1] * mabs * deq_scale;
            dst[h2] = src[a2] * mabs * deq_scale;
            dst[h3] = src[a3] * mabs * deq_scale;
        }
    }
}

// ─── SiLU (Sigmoid Linear Unit) in-place ─────────────────────────────
static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// ─── Forward one transformer layer (all 7 matmuls + attention in C) ──
// input/output: [B, hidden_dim] float32
// positions: [B] int32 position indices
ATLAS_API void atlas_forward_layer(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    uint16_t* k_cache, uint16_t* v_cache,
    int max_seq_len, int seq_now,
    int idx_ln1, int idx_q, int idx_k, int idx_v, int idx_o,
    int idx_ln2, int idx_gate, int idx_up, int idx_down) {

    int H = m->hidden_dim;
    int nH = m->n_heads, nKV = m->n_kv_heads, hd = m->head_dim;
    int qd = nH * hd, kvd = nKV * hd;
    int inter = m->inter_dim;
    float theta = m->rope_theta;

    m->ensure_buffers(B);

    // ─── 1. Pre-attention RMSNorm ───
    {
        auto& t = m->tensors[idx_ln1];
        const uint8_t* w = t.data;
        for (int b = 0; b < B; b++) {
            const float* xb = input + b * H;
            float* nb = output + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
    }
    float* x_norm = output;
    memcpy(m->snap_norm1, x_norm, B * H * sizeof(float));

    // ─── 2. QKV projections (int8) ───
    auto get_i8 = [](const TensorInfo& t, int8_t*& w, int32_t*& rs,
                     int& rows, int& dim, float& scale) {
        if (t.ttype != 3) { rows = 0; dim = 0; w = nullptr; rs = nullptr; return; }
        uint16_t sr; memcpy(&sr, t.data, 2);
        scale = fp16_to_fp32(sr);
        rows = t.row_dim;
        dim = t.packed_cols * 5;
        int nv = rows * dim;
        w = (int8_t*)(t.data + 2);
        rs = (int32_t*)(w + nv);
    };

    auto& tq = m->tensors[idx_q];
    auto& tk = m->tensors[idx_k];
    int i8_q_dim = tq.packed_cols * 5;
    int i8_k_dim = tk.packed_cols * 5;
    int max_qkv_dim = i8_q_dim > i8_k_dim ? i8_q_dim : i8_k_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * max_qkv_dim, x_norm + b * H, H * sizeof(float));
        memset(m->buf_act + b * max_qkv_dim + H, 0,
               (max_qkv_dim - H) * sizeof(float));
    }

    float* max_abs = (float*)alloca(B * sizeof(float));
    quantize_f32_to_u8(m->buf_act, B, max_qkv_dim, max_abs, m->buf_i8);

    // Q projection: scratch=buf_act, output=buf_gate
    {
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(tq, w, rs, rows, dim, scale);
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_act, m->buf_gate, B);
        memcpy(m->snap_q, m->buf_gate, B * H * sizeof(float));
    }
    auto& tv = m->tensors[idx_v];
    int i8_v_dim = tv.packed_cols * 5;

    // K projection: scratch=buf_act (reused), output=buf_hidden
    {
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(tk, w, rs, rows, dim, scale);
        if (i8_k_dim != max_qkv_dim) {
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * i8_k_dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * i8_k_dim + H, 0,
                       (i8_k_dim - H) * sizeof(float));
            }
            quantize_f32_to_u8(m->buf_act, B, i8_k_dim, max_abs, m->buf_i8);
        }
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_act, m->buf_hidden, B);
        memcpy(m->snap_k, m->buf_hidden, B * kvd * sizeof(float));
    }
    // V projection: scratch=buf_act (reused again, free after K over), output=buf_up
    {
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(tv, w, rs, rows, dim, scale);
        if (i8_v_dim != i8_k_dim) {
            for (int b = 0; b < B; b++) {
                memcpy(m->buf_act + b * i8_v_dim, x_norm + b * H, H * sizeof(float));
                memset(m->buf_act + b * i8_v_dim + H, 0,
                       (i8_v_dim - H) * sizeof(float));
            }
            quantize_f32_to_u8(m->buf_act, B, i8_v_dim, max_abs, m->buf_i8);
        }
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_act, m->buf_up, B);
        memcpy(m->snap_v, m->buf_up, B * kvd * sizeof(float));
    }

    // ─── 3. Fused attention (reuse atlas_attention_f32) ───
    float* attn_out = (float*)alloca(B * qd * sizeof(float));
    // Q in buf_gate, K in buf_hidden, V in buf_up
    float* q_f32 = (float*)alloca(B * qd * sizeof(float));
    float* k_f32 = (float*)alloca(B * kvd * sizeof(float));
    float* v_f32 = (float*)alloca(B * kvd * sizeof(float));
    for (int b = 0; b < B; b++) {
        memcpy(q_f32 + b * qd, m->buf_gate + b * tq.row_dim, qd * sizeof(float));
        memcpy(k_f32 + b * kvd, m->buf_hidden + b * tk.row_dim, kvd * sizeof(float));
        memcpy(v_f32 + b * kvd, m->buf_up + b * tv.row_dim, kvd * sizeof(float));
    }
    atlas_attention_f32(q_f32, k_f32, v_f32, positions,
        k_cache, v_cache, max_seq_len, seq_now, B,
        nH, nKV, hd, theta, attn_out);

    // ─── 4. O projection (int8) ───
    {
        auto& to = m->tensors[idx_o];
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(to, w, rs, rows, dim, scale);
        for (int b = 0; b < B; b++) {
            memcpy(m->buf_act + b * dim, attn_out + b * qd, qd * sizeof(float));
            memset(m->buf_act + b * dim + qd, 0, (dim - qd) * sizeof(float));
        }
        quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_hidden, m->buf_gate, B);
        memcpy(m->snap_o, m->buf_gate, B * H * sizeof(float));
    }

    // ─── 5. Residual: output = input + attn_out_proj ───
    for (int i = 0; i < B * H; i++) {
        output[i] = input[i] + m->buf_gate[i];
    }
    // ─── 6. Post-attention RMSNorm ───
    {
        auto& t = m->tensors[idx_ln2];
        const uint8_t* w = t.data;
        for (int b = 0; b < B; b++) {
            const float* xb = output + b * H;
            float* nb = m->buf_act + b * H;
            float ss = 0.0f;
            for (int i = 0; i < H; i++) ss += xb[i] * xb[i];
            float rms = 1.0f / sqrtf(ss / H + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint16_t w16; memcpy(&w16, w + i * 2, 2);
                nb[i] = xb[i] * rms * fp16_to_fp32(w16);
            }
        }
    }
    float* x_norm2 = m->buf_act;

    // ─── 7. FFN: gate + up projections (int8) ───
    auto& tg = m->tensors[idx_gate];
    auto& tu = m->tensors[idx_up];
    int g_dim = tg.packed_cols * 5;
    int u_dim = tu.packed_cols * 5;
    int ffn_dim = g_dim > u_dim ? g_dim : u_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
        memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
    }
    quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);

    // Gate
    {
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(tg, w, rs, rows, dim, scale);
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_hidden, m->buf_gate, B);
    }
    // Up
    {
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(tu, w, rs, rows, dim, scale);
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_hidden, m->buf_up, B);
    }

    // ─── 8. SiLU(gate) * up ───
    silu_inplace(m->buf_gate, B * inter);
    for (int i = 0; i < B * inter; i++) {
        m->buf_hidden[i] = m->buf_gate[i] * m->buf_up[i];
    }

    // ─── 9. Down projection (int8) ───
    {
        auto& td = m->tensors[idx_down];
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(td, w, rs, rows, dim, scale);
        for (int b = 0; b < B; b++) {
            memcpy(m->buf_act + b * dim, m->buf_hidden + b * inter, inter * sizeof(float));
            memset(m->buf_act + b * dim + inter, 0, (dim - inter) * sizeof(float));
        }
        quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
        matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                          scale, m->buf_hidden, m->buf_gate, B);
    }

    // ─── 10. Residual: output += down_proj ───
    for (int i = 0; i < B * H; i++) {
        output[i] += m->buf_gate[i];
    }
}

// ─── Debug: get internal buffer pointers ──────────────────────────────
ATLAS_API float* atlas_get_snap_q(AtlasModel* m) { return m->snap_q; }
ATLAS_API float* atlas_get_snap_k(AtlasModel* m) { return m->snap_k; }
ATLAS_API float* atlas_get_snap_v(AtlasModel* m) { return m->snap_v; }
ATLAS_API float* atlas_get_snap_o(AtlasModel* m) { return m->snap_o; }
ATLAS_API float* atlas_get_snap_norm1(AtlasModel* m) { return m->snap_norm1; }

}  // extern "C"
