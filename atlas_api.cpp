// atlas_api.cpp — C-exported DLL for TQ1.0 inference acceleration
#define ATLAS_API __declspec(dllexport)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <malloc.h>
#include <io.h>

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

        t.data = new uint8_t[t.data_size];
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
    for (auto& t : m->tensors) delete[] t.data;
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

}  // extern "C"
