// atlas_api.cpp — C-exported DLL for TQ1.0 inference acceleration
// atlas_ffi.h is the pure C API contract for FFI consumers (standalone reference)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <string>
#include <immintrin.h>

#ifdef _WIN32
  #define ATLAS_API __declspec(dllexport)
  #include <malloc.h>
  #include <io.h>
  #include <windows.h>
  // VirtualAlloc returns memory to OS on free — no fragmentation
  struct AllocHdr { void* base; size_t total; };
  static uint8_t* atlas_valloc(size_t size) {
      size_t hdr = sizeof(AllocHdr), align = 32;
      size_t total = hdr + size + (align - 1);
      uint8_t* base = (uint8_t*)VirtualAlloc(NULL, total,
          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!base) return nullptr;
      uint8_t* data = (uint8_t*)(((uintptr_t)(base + hdr + align - 1)) & ~(align - 1));
      AllocHdr* h = (AllocHdr*)(data - hdr);
      h->base = base; h->total = total;
      return data;
  }
  static void atlas_vfree(uint8_t* ptr) {
      if (!ptr) return;
      AllocHdr* h = (AllocHdr*)(ptr - sizeof(AllocHdr));
      VirtualFree(h->base, 0, MEM_RELEASE);
  }
  #define FSEEK _fseeki64
  #define FTELL _ftelli64
  #define STRICMP _stricmp
#else
  #define ATLAS_API __attribute__((visibility("default")))
  #include <cstdlib>
  #include <unistd.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  // mmap-based allocator — MAP_POPULATE hints pages into RAM
  struct AllocHdr { void* base; size_t total; };
  static uint8_t* atlas_valloc(size_t size) {
      size_t hdr = sizeof(AllocHdr), align = 32;
      size_t total = hdr + size + (align - 1);
      size_t pages = (total + 4095) & ~4095;
      void* p = mmap(NULL, pages, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
      if (p == MAP_FAILED) return nullptr;
      uint8_t* base = (uint8_t*)p;
      uint8_t* data = (uint8_t*)(((uintptr_t)(base + hdr + align - 1)) & ~(align - 1));
      AllocHdr* h = (AllocHdr*)(data - hdr);
      h->base = base; h->total = pages;
      return data;
  }
  static void atlas_vfree(uint8_t* ptr) {
      if (!ptr) return;
      AllocHdr* h = (AllocHdr*)(ptr - sizeof(AllocHdr));
      munmap(h->base, h->total);
  }
  #define FSEEK fseeko
  #define FTELL ftello
  #define STRICMP strcasecmp
#endif

// ─── Xoshiro256** PRNG (thread-safe, 64-bit) ──────────────────────────
static uint64_t xoshiro_state[4] = {0};

static void xoshiro_seed(uint64_t seed) {
    auto sm64 = [](uint64_t& s) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    xoshiro_state[0] = sm64(seed);
    xoshiro_state[1] = sm64(seed);
    xoshiro_state[2] = sm64(seed);
    xoshiro_state[3] = sm64(seed);
}

static uint64_t xoshiro_next() {
    uint64_t t = xoshiro_state[1] << 17;
    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[3] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[0] ^= xoshiro_state[3];
    xoshiro_state[2] ^= t;
    t = xoshiro_state[3];
    t = (t ^ (t >> 16)) * 0x45D9F3BEB5AADB97ull;
    t = (t ^ (t >> 16)) * 0x45D9F3BEB5AADB97ull;
    return t ^ (t >> 16);
}

static float xoshiro_float() {
    return (float)((xoshiro_next() >> 11) * 0x1.0p-53);
}

// ─── Gumbel-max sample internal (used by both atlas_sample and atlas_generate) ──
// Modifies logits in-place as scratch. Returns sampled token ID.
// When temperature <= 0: deterministic argmax (greedy).
static int gumbel_sample(float* logits, int V,
                          float temperature, int top_k, float top_p) {
    // Temperature ≤ 0: deterministic greedy argmax
    if (temperature <= 0.0f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < V; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        return best;
    }

    if (temperature != 1.0f) {
        float invT = 1.0f / temperature;
        for (int i = 0; i < V; i++) logits[i] *= invT;
    }

    // Top-k: find kth largest, zero out below it
    if (top_k > 0 && top_k < V) {
        std::vector<float> copy(logits, logits + V);
        std::nth_element(copy.begin(), copy.begin() + top_k - 1, copy.end(),
                         [](float a, float b) { return a > b; });
        float threshold = copy[top_k - 1];
        for (int i = 0; i < V; i++)
            if (logits[i] < threshold) logits[i] = -FLT_MAX / 4;
    }

    // Softmax (max-subtraction for numerical stability) into probs
    float max_val = -FLT_MAX / 4;
    for (int i = 0; i < V; i++) if (logits[i] > max_val) max_val = logits[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < V; i++) sum_exp += expf(logits[i] - max_val);
    float inv_sum = 1.0f / (sum_exp + 1e-38f);
    for (int i = 0; i < V; i++) logits[i] = expf(logits[i] - max_val) * inv_sum;

    // Top-p: find nucleus via descending sort, mask tail
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return logits[a] > logits[b]; });
        float cum = 0.0f;
        for (int i = 0; i < V; i++) {
            cum += logits[idx[i]];
            if (cum > top_p) {
                for (int j = i + 1; j < V; j++) logits[idx[j]] = 0.0f;
                break;
            }
        }
    }

    // Gumbel-max: add noise, take argmax
    int best = 0;
    float best_val = -FLT_MAX / 4;
    for (int i = 0; i < V; i++) {
        if (logits[i] <= 0.0f) continue;
        float u = xoshiro_float();
        if (u <= 0.0f) u = 1e-38f;
        float noise = -logf(-logf(u));
        float val = logf(logits[i]) + noise;
        if (val > best_val) { best_val = val; best = i; }
    }
    return best;
}

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
    bool use_f32_matmul = false; // skip activation quantization (1B model needs full precision)
    std::vector<TensorInfo> tensors;
    // Tensor names (loaded from v4+ atlas files, eliminates safetensors dependency)
    std::vector<std::string> tensor_names;

    // Cached layer index array for atlas_generate (v1.2.0)
    std::vector<int> layer_idx_cache;
    bool has_layer_idx = false;

    // Pre-allocated scratch buffers for forward_layer (lazy init)
    float* buf_gate = nullptr;      // [max_batch * inter_dim]
    float* buf_up = nullptr;        // [max_batch * inter_dim]
    float* buf_hidden = nullptr;    // [max_batch * inter_dim]
    float* buf_act = nullptr;       // [max_batch * max_dim] quantized f32→i8 scratch
    uint8_t* buf_i8 = nullptr;      // [max_batch * max_dim] uint8 quantized activations
    float* buf_out = nullptr;       // [max_batch * hidden_dim] layer output ping-pong for atlas_forward
    int max_batch = 0;
    // mmap cache handles (for int8 data loaded from .i8 file)
    void* mmap_base = nullptr;      // MapViewOfFile base (.i8 cache)
    void* mmap_handle = nullptr;    // CreateFileMapping handle (Win) / file size (Lin)
    void* mmap_file = nullptr;      // CreateFile handle (Win) / fd (Lin)
    size_t mmap_size = 0;           // actual file size for range checks
    // mmap atlas file handles (for fp16 tensor data — demand-paged by OS)
    void* atlas_mmap_base = nullptr;
    void* atlas_mmap_handle = nullptr;  // CreateFileMapping handle (Win) / file size (Lin)
    void* atlas_mmap_file = nullptr;    // duplicated fd (Win: HANDLE, Lin: fd)
    size_t atlas_mmap_size = 0;        // actual file size for range checks
    // Embedded tokenizer data (v5+)
    int tokenizer_size = 0;
    uint32_t tokenizer_offset = 0;
    // Int8 quantized lm_head (per-row symmetric, ~403 MB instead of 1.5 GB fp32)
    int8_t* lm_head_i8 = nullptr;
    int32_t* lm_head_offsets = nullptr;  // precomputed 128 * sum(w) per row
    float* lm_head_scales = nullptr;
    bool lm_head_quantized = false;
    ~AtlasModel() {
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);
        if (lm_head_i8) atlas_vfree((uint8_t*)lm_head_i8);
        if (lm_head_offsets) atlas_vfree((uint8_t*)lm_head_offsets);
        if (lm_head_scales) atlas_vfree((uint8_t*)lm_head_scales);
    }

    bool is_mapped(const uint8_t* ptr) const {
        if (mmap_base && mmap_size > 0) {
            if (ptr >= (const uint8_t*)mmap_base &&
                ptr < (const uint8_t*)mmap_base + mmap_size) return true;
        }
        if (atlas_mmap_base && atlas_mmap_size > 0) {
            return ptr >= (const uint8_t*)atlas_mmap_base &&
                   ptr < (const uint8_t*)atlas_mmap_base + atlas_mmap_size;
        }
        return false;
    }

    void ensure_buffers(int B) {
        if (B <= max_batch) return;
        if (buf_gate) atlas_vfree((uint8_t*)buf_gate);
        if (buf_up) atlas_vfree((uint8_t*)buf_up);
        if (buf_hidden) atlas_vfree((uint8_t*)buf_hidden);
        if (buf_act) atlas_vfree((uint8_t*)buf_act);
        if (buf_i8) atlas_vfree((uint8_t*)buf_i8);
        if (buf_out) atlas_vfree((uint8_t*)buf_out);

        int max_dim = inter_dim > hidden_dim ? inter_dim : hidden_dim;
        if (n_heads * head_dim > max_dim) max_dim = n_heads * head_dim;
        int max_aligned = ((max_dim + 7) + 31) & ~31;  // +7 for TQ1 padding (packed_cols*5 up to dim+4)

        buf_gate = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
        buf_up = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
        buf_hidden = (float*)atlas_valloc((size_t)B * inter_dim * sizeof(float));
        buf_act = (float*)atlas_valloc((size_t)B * max_aligned * sizeof(float));
        buf_i8 = (uint8_t*)atlas_valloc((size_t)B * max_aligned * sizeof(uint8_t));
        buf_out = (float*)atlas_valloc((size_t)B * hidden_dim * sizeof(float));
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

    // v5+ tokenizer header fields
    memcpy(&tmp32, hdr+29, 4); m->tokenizer_size = (int)tmp32;
    uint32_t tok_off_u32; memcpy(&tok_off_u32, hdr+33, 4); m->tokenizer_offset = tok_off_u32;

    printf("[ATLAS] v%d model: %dL %dH %dI %d/%d heads %d vocab %.0f theta | %d tensors %s\n",
           version,
           m->n_layers, m->hidden_dim, m->inter_dim, m->n_heads, m->n_kv_heads,
           m->vocab_size, m->rope_theta, n_tensors,
           m->tokenizer_size > 0 ? "(embedded tokenizer)" : "");

    // Read directory
    m->tensors.resize(n_tensors);
    std::vector<uint32_t> file_offsets(n_tensors);
    FSEEK(f, 64, SEEK_SET);
    for (int i = 0; i < n_tensors; i++) {
        uint8_t e[12]; fread(e, 1, 12, f);
        m->tensors[i].ttype = e[0];
        memcpy(&file_offsets[i], e+1, 4); m->tensors[i].file_offset = file_offsets[i];
        memcpy(&m->tensors[i].row_dim, e+5, 4);
        m->tensors[i].packed_cols = e[9] | (e[10]<<8) | (e[11]<<16);
    }

    // Load tensor names (v4+)
    m->tensor_names.clear();
    if (version >= 4) {
        int nb_size; memcpy(&nb_size, hdr+56, 4);
        if (nb_size > 0) {
            // Names stored right after directory: [name_block_size:4] [name_0\0]... 
            FSEEK(f, 64 + n_tensors * 12, SEEK_SET);
            uint8_t* nb = new uint8_t[nb_size];
            fread(nb, 1, nb_size, f);
            int pos = 4;  // skip size field
            for (int i = 0; i < n_tensors && pos < nb_size; i++) {
                const char* s = (const char*)(nb + pos);
                int len = (int)strnlen(s, nb_size - pos);
                m->tensor_names.push_back(std::string(s, len));
                pos += len + 1;
            }
            delete[] nb;
        }
    }

    // Compute file size for last-tensor calculation
    FSEEK(f, 0, SEEK_END);
    int64_t file_size = FTELL(f);
    FSEEK(f, 64, SEEK_SET);

    // Load all tensor data from mmap'd atlas file
    // mmap the entire atlas file for zero-copy access to tensor data
#ifdef _WIN32
    HANDLE hFileW = (HANDLE)_get_osfhandle(_fileno(f));
    HANDLE hDup = INVALID_HANDLE_VALUE;
    DuplicateHandle(GetCurrentProcess(), hFileW, GetCurrentProcess(),
                    &hDup, FILE_READ_DATA, FALSE, 0);
    HANDLE hMapW = CreateFileMappingW(hDup, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapW) {
        uint8_t* map_base = (uint8_t*)MapViewOfFile(hMapW, FILE_MAP_READ, 0, 0, 0);
        if (map_base) {
            m->atlas_mmap_base = map_base;
            m->atlas_mmap_handle = hMapW;
            m->atlas_mmap_file = (void*)hDup;
            m->atlas_mmap_size = (size_t)file_size;
        } else {
            CloseHandle(hMapW); CloseHandle(hDup);
        }
    } else {
        CloseHandle(hDup);
    }
#else
    int fd = fileno(f);
    // dup fd so mmap outlives fclose
    int map_fd = dup(fd);
    if (file_size > 0 && map_fd >= 0) {
        uint8_t* map_base = (uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, map_fd, 0);
        if (map_base != MAP_FAILED) {
            m->atlas_mmap_base = map_base;
            m->atlas_mmap_handle = (void*)(intptr_t)file_size;
            m->atlas_mmap_file = (void*)(intptr_t)map_fd;
            m->atlas_mmap_size = (size_t)file_size;
        } else {
            close(map_fd);
        }
    }
#endif

    for (int i = 0; i < n_tensors; i++) {
        auto& t = m->tensors[i];

        if (t.ttype == 0) {  // TQ1: 2-byte scale + packed data
            t.data_size = 2 + t.row_dim * t.packed_cols;
        } else if (t.ttype == 1) {  // norm/embed: raw float16
            if (t.row_dim == m->vocab_size) {
                t.data_size = t.row_dim * m->hidden_dim * 2;
            } else {
                t.data_size = t.row_dim * 2;
            }
            t.packed_cols = 0;
        } else {  // lm_head / scales
            int actual_bytes = (i + 1 < n_tensors)
                ? (int)(file_offsets[i + 1] - file_offsets[i]) - ((int)(file_offsets[i + 1] - file_offsets[i]) % 32)
                : (int)(file_size - (int64_t)file_offsets[i]);
            int expected = t.row_dim * m->hidden_dim * 2;
            if (actual_bytes < expected && actual_bytes > 0) {
                t.data_size = actual_bytes;
            } else {
                t.data_size = expected;
            }
            t.packed_cols = 0;
        }

        // Point into mmap'd atlas file instead of fread
        if (m->atlas_mmap_base) {
            t.data = (uint8_t*)m->atlas_mmap_base + file_offsets[i];
        } else {
            // Fallback: fread into valloc'd buffer (no mmap available)
            t.data = atlas_valloc(t.data_size);
            FSEEK(f, (int64_t)file_offsets[i], SEEK_SET);
            fread(t.data, 1, t.data_size, f);
        }
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
    // Free valloc'd tensors (not mmap'd ones)
    for (auto& t : m->tensors) {
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
    }
    // Unmap cache if loaded
    if (m->mmap_base) {
#ifdef _WIN32
        UnmapViewOfFile(m->mmap_base);
        CloseHandle(m->mmap_handle);
        CloseHandle((HANDLE)m->mmap_file);
#else
        munmap(m->mmap_base, m->mmap_size);
        close((int)(intptr_t)m->mmap_file);
#endif
    }
    // Unmap atlas file if loaded
    if (m->atlas_mmap_base) {
#ifdef _WIN32
        UnmapViewOfFile(m->atlas_mmap_base);
        CloseHandle(m->atlas_mmap_handle);
        CloseHandle((HANDLE)m->atlas_mmap_file);
#else
        munmap(m->atlas_mmap_base, m->atlas_mmap_size);
        close((int)(intptr_t)m->atlas_mmap_file);
#endif
    }
    delete m;
}

// ─── Model config struct ───────────────────────────────────────────────
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

// ─── Get model info ────────────────────────────────────────────────────
ATLAS_API AtlasModelConfig atlas_get_config(AtlasModel* m) {
    AtlasModelConfig cfg;
    cfg.n_layers = m->n_layers;
    cfg.hidden_dim = m->hidden_dim;
    cfg.inter_dim = m->inter_dim;
    cfg.n_heads = m->n_heads;
    cfg.n_kv_heads = m->n_kv_heads;
    cfg.head_dim = m->head_dim;
    cfg.vocab_size = m->vocab_size;
    cfg.rope_theta = m->rope_theta;
    return cfg;
}

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

// ─── Tensor name API (v4+, no safetensors dependency) ─────────────────
ATLAS_API int atlas_get_tensor_count(AtlasModel* m) {
    return m ? (int)m->tensor_names.size() : 0;
}

ATLAS_API int atlas_get_tensor_name(AtlasModel* m, int idx, char* buf, int buf_size) {
    if (!m || idx < 0 || idx >= (int)m->tensor_names.size() || !buf || buf_size <= 0)
        return 0;
    const std::string& s = m->tensor_names[idx];
    int len = (int)s.size();
    int copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, s.data(), copy);
    buf[copy] = '\0';
    return copy;
}

// ─── Embedded tokenizer API (v5+) ────────────────────────────────────
ATLAS_API const uint8_t* atlas_get_tokenizer(AtlasModel* m, int* size) {
    if (!m || !m->tokenizer_size || !m->atlas_mmap_base) {
        if (size) *size = 0; return nullptr;
    }
    if (size) *size = m->tokenizer_size;
    return (const uint8_t*)m->atlas_mmap_base + (ptrdiff_t)m->tokenizer_offset;
}

ATLAS_API int atlas_get_tensor_index(AtlasModel* m, const char* name) {
    if (!m || !name) return -1;
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        if (m->tensor_names[i] == name) return i;
    }
    return -1;
}

// ─── Cache file ──────────────────────────────────────────────────────
// Save decompressed int8 tensors to a .i8 cache file for instant reload.
// Format: [n_tensors:4] then per-tensor [ttype:1][row_dim:4][pc:4][ds:4][off:8]
//         then all tensor data concatenated.

static void cache_path(const char* atlas_path, char* out, int out_size) {
    snprintf(out, out_size, "%s", atlas_path);
    int len = (int)strlen(out);
    // Replace .atlas suffix with .i8 (or append .i8 if no .atlas)
    const char* dot = strrchr(out, '.');
    if (dot && STRICMP(dot, ".atlas") == 0) {
        int prefix_len = (int)(dot - out);
        out[prefix_len] = '.';
        out[prefix_len+1] = 'i';
        out[prefix_len+2] = '8';
        out[prefix_len+3] = '\0';
    } else {
        strncat(out, ".i8", out_size - len - 1);
    }
}

ATLAS_API void atlas_save_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; cache_path(atlas_path, path, sizeof(path));

    int n = (int)m->tensors.size();
    // Compute total data size needed
    int64_t total_data = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if (t.ttype == 3 && t.data_size > 0 && t.data)
            total_data += t.data_size;
    }

    int64_t header_size = 4 + (int64_t)n * 21;
    int64_t cache_size = header_size + total_data;

    // Check available disk space on Windows
#ifdef _WIN32
    char root[4] = { path[0], ':', '\\', 0 };
    ULARGE_INTEGER free_bytes;
    if (GetDiskFreeSpaceExA(root, &free_bytes, NULL, NULL)) {
        if (cache_size > (int64_t)free_bytes.QuadPart) {
            printf("[CACHE] Skip: need %.1f GB, only %.1f GB free on %s\n",
                   cache_size / 1e9, (double)free_bytes.QuadPart / 1e9, root);
            return;
        }
    }
#endif

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[CACHE] Cannot write %s\n", path); return; }
#ifdef _WIN32
    setvbuf(f, NULL, _IONBF, 0);
#endif

    fwrite(&n, 4, 1, f);

    // Only cache int8-decoded tensors (ttype==3). Non-int8 tensors (norms, embed, GQA scales)
    // have correct data from atlas_load and don't need caching.
    std::vector<int64_t> offsets(n, -1);
    int64_t cur = 0;
    for (int i = 0; i < n; i++) {
        auto& t = m->tensors[i];
        if (t.ttype == 3 && t.data_size > 0 && t.data) {
            offsets[i] = cur;
            cur += t.data_size;
        }
    }

    // Write header with correct offsets
    for (int i = 0; i < n; i++) {
        uint8_t ttype = (uint8_t)m->tensors[i].ttype;
        int row_dim = m->tensors[i].row_dim;
        int pc = m->tensors[i].packed_cols;
        int ds = m->tensors[i].data_size;
        int64_t off = offsets[i] >= 0 ? offsets[i] : 0;
        fwrite(&ttype, 1, 1, f);
        fwrite(&row_dim, 4, 1, f);
        fwrite(&pc, 4, 1, f);
        fwrite(&ds, 4, 1, f);
        fwrite(&off, 8, 1, f);
    }

    // Write data, retrying on short writes
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (offsets[i] < 0) continue;
        const uint8_t* ptr = m->tensors[i].data;
        int remaining = m->tensors[i].data_size;
        while (remaining > 0) {
            size_t written = fwrite(ptr, 1, remaining, f);
            if ((int)written <= 0) {
                fprintf(stderr, "[CACHE] ERROR: tensor %d write failed (%d remaining)\n", i, remaining);
                ok = false; break;
            }
            ptr += written;
            remaining -= (int)written;
        }
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "[CACHE] Write failed, deleting partial cache\n");
        remove(path);
    } else {
        printf("[CACHE] Saved %d tensors (%.1f MB)\n", n, cache_size / 1e6);
    }
}

ATLAS_API int atlas_load_cache(AtlasModel* m, const char* atlas_path) {
    char path[1024]; cache_path(atlas_path, path, sizeof(path));
    uint8_t* base = nullptr;
    void* hFile = nullptr;
    void* hMap = nullptr;
    size_t file_size = 0;

#ifdef _WIN32
    HANDLE hFileW = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFileW == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFileW, &fsize) || fsize.QuadPart < 4) {
        CloseHandle(hFileW); return 0;
    }
    file_size = (size_t)fsize.QuadPart;
    HANDLE hMapW = CreateFileMappingA(hFileW, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapW) { CloseHandle(hFileW); return 0; }
    base = (uint8_t*)MapViewOfFile(hMapW, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(hMapW); CloseHandle(hFileW); return 0; }
    hFile = (void*)hFileW;
    hMap = (void*)hMapW;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t fsize = lseek(fd, 0, SEEK_END);
    if (fsize <= 4) { close(fd); return 0; }
    file_size = (size_t)fsize;
    base = (uint8_t*)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return 0; }
    hFile = (void*)(intptr_t)fd;
    hMap = (void*)(intptr_t)fsize;
#endif

    int n = *(int*)base;
    int64_t min_size = 4 + (int64_t)n * 21;
    if (n != (int)m->tensors.size() || file_size < (size_t)min_size) {
#ifdef _WIN32
        UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
        munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
        return 0;
    }

    uint8_t* hdr = base + 4;
    int64_t data_start = min_size;

    int replaced = 0;
    for (int i = 0; i < n; i++) {
        uint8_t* e = hdr + i * 21;
        int cttype = (int)e[0];
        int row_dim; memcpy(&row_dim, e+1, 4);
        int cpc; memcpy(&cpc, e+5, 4);
        int ds; memcpy(&ds, e+9, 4);
        int64_t off; memcpy(&off, e+13, 8);

        // Validate offset + size fits within file
        if (cttype == 3 && ds > 0 && off >= 0) {
            if ((size_t)(data_start + off + ds) > file_size) {
                // Truncated/partial cache — unsafe to use
                printf("[CACHE] Truncated (tensor %d exceeds file), ignoring cache\n", i);
#ifdef _WIN32
                UnmapViewOfFile(base); CloseHandle((HANDLE)hMap); CloseHandle((HANDLE)hFile);
#else
                munmap(base, (size_t)(intptr_t)hMap); close((int)(intptr_t)hFile);
#endif
                return 0;
            }
        }

        auto& t = m->tensors[i];
        if (t.ttype == 0 && cttype == 3 && ds > 0 && off >= 0) {
            if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
            t.ttype = 3;
            t.row_dim = row_dim;
            t.packed_cols = cpc;
            t.data_size = ds;
            t.data = (uint8_t*)(base + data_start + off);
            replaced++;
        }
    }

    // Store mmap handles + size for cleanup in atlas_free
    m->mmap_base = base;
    m->mmap_handle = hMap;
    m->mmap_file = hFile;
    m->mmap_size = file_size;

    printf("[CACHE] Loaded %d/%d tensors\n", replaced, n);
    return replaced > 0 ? 1 : 0;
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
        uint8_t* new_data = atlas_valloc(2 + n_vals + t.row_dim * 4);
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
        if (t.data && !m->is_mapped(t.data)) atlas_vfree(t.data);
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
    int64_t step = 4096 / sizeof(int8_t);
    int n = (int)m->tensors.size();
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 4)
    for (int ti = 0; ti < n; ti++) {
        auto& t = m->tensors[ti];
        if (t.ttype != 3) continue;
        int n_vals = t.row_dim * t.packed_cols * 5;
        int8_t* data = (int8_t*)(t.data + 2);
        for (int64_t i = 0; i < n_vals; i += step) {
            volatile int sink = data[i]; (void)sink;
        }
        total += n_vals;
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

// ─── Set f32 matmul mode (no activation quantization) ────────────────
ATLAS_API void atlas_set_use_f32_matmul(AtlasModel* m, int val) {
    if (m) m->use_f32_matmul = val ? true : false;
}

// ─── Helper: horizontal sum of __m256 float ──────────────────────────
static inline float hsum_ps(__m256 v) {
    __m128 l = _mm256_castps256_ps128(v);
    __m128 h = _mm256_extractf128_ps(v, 1);
    l = _mm_add_ps(l, h);
    l = _mm_hadd_ps(l, l);
    l = _mm_hadd_ps(l, l);
    return _mm_cvtss_f32(l);
}

// ─── Helper: f32×i8 matmul + reorder (no activation quantization) ───
// act_f32: [B, input_dim] float activations (not quantized)
// weights: [rows, input_dim] int8 weights
// scale: per-tensor dequant scale
// output: [B, rows] reordered float output
static void matmul_f32_reorder(int rows, int input_dim,
    const int8_t* weights, const float* act_f32,
    float scale, float* output, int B) {
    int rows_packed = rows / 4;
    #ifdef _OPENMP
    #pragma omp parallel for if(rows_packed > 4)
    #endif
    for (int ur = 0; ur < rows_packed; ur++) {
        for (int b = 0; b < B; b++) {
            const float* a = act_f32 + b * input_dim;
            float out4[4];
            for (int sub = 0; sub < 4; sub++) {
                const int8_t* w = weights + (ur * 4 + sub) * input_dim;
                __m256 sum = _mm256_setzero_ps();
                int c = 0;
                for (; c + 8 <= input_dim; c += 8) {
                    __m256 af = _mm256_loadu_ps(a + c);
                    __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + c));
                    __m256i w32 = _mm256_cvtepi8_epi32(w8);
                    __m256 wf = _mm256_cvtepi32_ps(w32);
                    sum = _mm256_fmadd_ps(af, wf, sum);
                }
                float s = hsum_ps(sum);
                for (; c < input_dim; c++) s += a[c] * w[c];
                out4[sub] = s / scale;
            }
            float* dst = output + b * rows;
            dst[0 * rows_packed + ur] = out4[0];
            dst[1 * rows_packed + ur] = out4[1];
            dst[2 * rows_packed + ur] = out4[2];
            dst[3 * rows_packed + ur] = out4[3];
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


// ─── Internal: forward one transformer layer ──────────────────────────
// input: [B, H] float32 (read-only, preserved for residual)
// output: [B, H] float32 (must not alias input)
// Per-layer K/V cache offset is computed from k_cache/v_cache + layer * n_kv * max_seq * hd
static void forward_layer_internal(
    AtlasModel* m,
    const float* input, float* output, int B,
    const int* positions,
    uint16_t* k_cache_layer, uint16_t* v_cache_layer,
    int max_seq_len, int seq_now,
    int idx_ln1, int idx_q, int idx_k, int idx_v, int idx_o,
    int idx_ln2, int idx_gate, int idx_up, int idx_down) {

    int H = m->hidden_dim;
    int nH = m->n_heads, nKV = m->n_kv_heads, hd = m->head_dim;
    int qd = nH * hd, kvd = nKV * hd;
    int inter = m->inter_dim;
    float theta = m->rope_theta;

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

    auto& tv = m->tensors[idx_v];
    if (m->use_f32_matmul) {
        // Full-precision path: f32×i8, no activation quantization
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tq, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tk, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_hidden, B);
        }
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tv, w, rs, rows, dim, scale);
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_up, B);
        }
    } else {
        quantize_f32_to_u8(m->buf_act, B, max_qkv_dim, max_abs, m->buf_i8);

        // Q projection: scratch=buf_act, output=buf_gate
        {
            int8_t* w; int32_t* rs; int rows, dim; float scale;
            get_i8(tq, w, rs, rows, dim, scale);
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_act, m->buf_gate, B);
        }
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
        }
    }

    // ─── 3. Fused attention (reuse atlas_attention_f32) ───
    float* attn_out = (float*)alloca(B * qd * sizeof(float));
    float* q_f32 = (float*)alloca(B * qd * sizeof(float));
    float* k_f32 = (float*)alloca(B * kvd * sizeof(float));
    float* v_f32 = (float*)alloca(B * kvd * sizeof(float));
    for (int b = 0; b < B; b++) {
        memcpy(q_f32 + b * qd, m->buf_gate + b * tq.row_dim, qd * sizeof(float));
        memcpy(k_f32 + b * kvd, m->buf_hidden + b * tk.row_dim, kvd * sizeof(float));
        memcpy(v_f32 + b * kvd, m->buf_up + b * tv.row_dim, kvd * sizeof(float));
    }
    atlas_attention_f32(q_f32, k_f32, v_f32, positions,
        k_cache_layer, v_cache_layer, max_seq_len, seq_now, B,
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
        if (m->use_f32_matmul) {
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        } else {
            quantize_f32_to_u8(m->buf_act, B, dim, max_abs, m->buf_i8);
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_hidden, m->buf_gate, B);
        }
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

    // ─── 7. FFN: fused gate + up projections (one OMP region) ───
    // Fused kernel processes both in one pass: shared activation, no scratch buffer,
    // reorder+dequant inline.
    auto& tg = m->tensors[idx_gate];
    auto& tu = m->tensors[idx_up];
    int g_dim = tg.packed_cols * 5;
    int u_dim = tu.packed_cols * 5;
    int ffn_dim = g_dim > u_dim ? g_dim : u_dim;

    for (int b = 0; b < B; b++) {
        memcpy(m->buf_act + b * ffn_dim, x_norm2 + b * H, H * sizeof(float));
        memset(m->buf_act + b * ffn_dim + H, 0, (ffn_dim - H) * sizeof(float));
    }

    if (m->use_f32_matmul) {
        // Full-precision FFN: f32 activations × int8 weights, no quantization
        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
        int rows_packed = rows / 4;

        #ifdef _OPENMP
        #pragma omp parallel for if(rows_packed > 4)
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            const int8_t* gw4 = gw + ur * 4 * dim_w;
            const int8_t* uw4 = uw + ur * 4 * dim_w;
            for (int b = 0; b < B; b++) {
                const float* a = m->buf_act + b * ffn_dim;
                float g_val[4], u_val[4];
                for (int sub = 0; sub < 4; sub++) {
                    const int8_t* wg = gw4 + sub * dim_w;
                    const int8_t* wu = uw4 + sub * dim_w;
                    __m256 gs = _mm256_setzero_ps();
                    __m256 us = _mm256_setzero_ps();
                    int c = 0;
                    for (; c + 8 <= dim_w; c += 8) {
                        __m256 af = _mm256_loadu_ps(a + c);
                        __m128i wg8 = _mm_loadl_epi64((const __m128i*)(wg + c));
                        __m128i wu8 = _mm_loadl_epi64((const __m128i*)(wu + c));
                        __m256 wg_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wg8));
                        __m256 wu_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wu8));
                        gs = _mm256_fmadd_ps(af, wg_f, gs);
                        us = _mm256_fmadd_ps(af, wu_f, us);
                    }
                    float gs_f = hsum_ps(gs);
                    float us_f = hsum_ps(us);
                    for (; c < dim_w; c++) {
                        gs_f += a[c] * wg[c];
                        us_f += a[c] * wu[c];
                    }
                    g_val[sub] = gs_f / g_scale;
                    u_val[sub] = us_f / u_scale;
                }
                float* g_out = m->buf_gate + b * rows;
                float* u_out = m->buf_up + b * rows;
                g_out[0 * rows_packed + ur] = g_val[0];
                g_out[1 * rows_packed + ur] = g_val[1];
                g_out[2 * rows_packed + ur] = g_val[2];
                g_out[3 * rows_packed + ur] = g_val[3];
                u_out[0 * rows_packed + ur] = u_val[0];
                u_out[1 * rows_packed + ur] = u_val[1];
                u_out[2 * rows_packed + ur] = u_val[2];
                u_out[3 * rows_packed + ur] = u_val[3];
            }
        }
    } else {
        quantize_f32_to_u8(m->buf_act, B, ffn_dim, max_abs, m->buf_i8);

        int8_t* gw; int32_t* grs; int g_rows, g_dim_v; float g_scale;
        int8_t* uw; int32_t* urs; int u_rows, u_dim_v; float u_scale;
        get_i8(tg, gw, grs, g_rows, g_dim_v, g_scale);
        get_i8(tu, uw, urs, u_rows, u_dim_v, u_scale);
        int rows = g_rows, dim_w = g_dim_v;
        int rows_packed = rows / 4;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int ur = 0; ur < rows_packed; ur++) {
            const int8_t* gw4 = gw + ur * 4 * dim_w;
            const int8_t* uw4 = uw + ur * 4 * dim_w;
            int32_t g_off[4] = {128 * grs[ur*4+0], 128 * grs[ur*4+1],
                                128 * grs[ur*4+2], 128 * grs[ur*4+3]};
            int32_t u_off[4] = {128 * urs[ur*4+0], 128 * urs[ur*4+1],
                                128 * urs[ur*4+2], 128 * urs[ur*4+3]};

            for (int b = 0; b < B; b++) {
                const uint8_t* a = m->buf_i8 + b * ffn_dim;
                float deq = max_abs[b] / 127.0f;

                float g_val[4], u_val[4];
                for (int sub = 0; sub < 4; sub++) {
                    const int8_t* wg = gw4 + sub * dim_w;
                    const int8_t* wu = uw4 + sub * dim_w;
                    int c = 0, g_dot = 0, u_dot = 0;

                    __m256i g_acc = _mm256_setzero_si256();
                    __m256i u_acc = _mm256_setzero_si256();
                    for (; c + 32 <= dim_w; c += 32) {
                        __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                        __m256i g_wv = _mm256_loadu_si256((const __m256i*)(wg + c));
                        __m256i u_wv = _mm256_loadu_si256((const __m256i*)(wu + c));
                        g_acc = _mm256_add_epi32(g_acc,
                            _mm256_madd_epi16(_mm256_maddubs_epi16(au, g_wv), _mm256_set1_epi16(1)));
                        u_acc = _mm256_add_epi32(u_acc,
                            _mm256_madd_epi16(_mm256_maddubs_epi16(au, u_wv), _mm256_set1_epi16(1)));
                    }

                    {
                        __m128i lo = _mm256_castsi256_si128(g_acc);
                        __m128i hi = _mm256_extracti128_si256(g_acc, 1);
                        __m128i s = _mm_add_epi32(lo, hi);
                        s = _mm_hadd_epi32(s, s);
                        s = _mm_hadd_epi32(s, s);
                        g_dot = _mm_cvtsi128_si32(s);
                    }
                    {
                        __m128i lo = _mm256_castsi256_si128(u_acc);
                        __m128i hi = _mm256_extracti128_si256(u_acc, 1);
                        __m128i s = _mm_add_epi32(lo, hi);
                        s = _mm_hadd_epi32(s, s);
                        s = _mm_hadd_epi32(s, s);
                        u_dot = _mm_cvtsi128_si32(s);
                    }

                    for (; c < dim_w; c++) {
                        g_dot += (int)a[c] * (int)wg[c];
                        u_dot += (int)a[c] * (int)wu[c];
                    }

                    g_dot -= g_off[sub]; u_dot -= u_off[sub];
                    g_val[sub] = (float)g_dot * deq / g_scale;
                    u_val[sub] = (float)u_dot * deq / u_scale;
                }

                // Reorder inline: each packed group scatters to 4 HF rows
                float* g_out = m->buf_gate + b * rows;
                float* u_out = m->buf_up + b * rows;
                g_out[0 * rows_packed + ur] = g_val[0];
                g_out[1 * rows_packed + ur] = g_val[1];
                g_out[2 * rows_packed + ur] = g_val[2];
                g_out[3 * rows_packed + ur] = g_val[3];
                u_out[0 * rows_packed + ur] = u_val[0];
                u_out[1 * rows_packed + ur] = u_val[1];
                u_out[2 * rows_packed + ur] = u_val[2];
                u_out[3 * rows_packed + ur] = u_val[3];
            }
        }
    }

    // ─── 8. Fused SiLU(gate)*up → down matmul ───
    {
        auto& td = m->tensors[idx_down];
        int8_t* w; int32_t* rs; int rows, dim; float scale;
        get_i8(td, w, rs, rows, dim, scale);
        if (m->use_f32_matmul) {
            // Full-precision: compute SiLU(gate)*up directly into buf_act, then f32×i8 matmul
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                for (int i = 0; i < inter; i++)
                    tmp[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
            }
            matmul_f32_reorder(rows, dim, w, m->buf_act, scale, m->buf_gate, B);
        } else {
            // Quantized path: SiLU(gate)*up → quantize to u8 → maddubs × i8 → dequant
            for (int b = 0; b < B; b++) {
                const float* g = m->buf_gate + b * inter;
                const float* u = m->buf_up + b * inter;
                float* tmp = m->buf_act + b * dim;
                float mb = 1e-5f;
                for (int i = 0; i < inter; i++) {
                    float v = (g[i] / (1.0f + expf(-g[i]))) * u[i];
                    tmp[i] = v;
                    float av = fabsf(v);
                    if (av > mb) mb = av;
                }
                for (int i = inter; i < dim; i++) tmp[i] = 0.0f;
                float inv = 127.0f / mb;
                max_abs[b] = mb;
                for (int i = 0; i < dim; i++) {
                    int q = (int)(tmp[i] * inv + 128.5f);
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    m->buf_i8[b * dim + i] = (uint8_t)q;
                }
            }
            matmul_reorder_deq(rows, dim, w, rs, m->buf_i8, max_abs,
                              scale, m->buf_hidden, m->buf_gate, B);
        }
    }

    // ─── 10. Residual: output += down_proj ───
    for (int i = 0; i < B * H; i++) {
        output[i] += m->buf_gate[i];
    }
}

// ─── Forward ALL transformer layers in one C call ────────────────────
// (single-layer atlas_forward_layer removed — fusion is always used)
// hidden_states: [B, H] float32 — overwritten with final layer output
// positions: [B] int32 position indices
// layer_idx: [n_layers * 9] int32 — flat array of tensor indices per layer
//           (ln1, q, k, v, o, ln2, gate, up, down) repeated for each layer
// k_cache, v_cache: flat buffers [n_layers * n_kv_heads * max_seq * head_dim] uint16
ATLAS_API void atlas_forward(
    AtlasModel* m,
    float* hidden_states, int B,
    const int* positions,
    uint16_t* k_cache, uint16_t* v_cache,
    int max_seq_len, int seq_now,
    const int* layer_idx, int n_layers) {

    m->ensure_buffers(B);
    int H = m->hidden_dim;
    int nKV = m->n_kv_heads, hd = m->head_dim;

    // Ping-pong: layer N output goes to separate buf_out (not buf_hidden, which is scratch)
    float* buf_a = hidden_states;
    float* buf_b = m->buf_out;

    for (int L = 0; L < n_layers; L++) {
        const int* idx = layer_idx + L * 9;
        uint16_t* kc = k_cache + L * nKV * max_seq_len * hd;
        uint16_t* vc = v_cache + L * nKV * max_seq_len * hd;
        forward_layer_internal(m, buf_a, buf_b, B, positions,
            kc, vc, max_seq_len, seq_now,
            idx[0], idx[1], idx[2], idx[3], idx[4],
            idx[5], idx[6], idx[7], idx[8]);
        float* tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    // If odd number of layers, final output is in buf_a after the last swap,
    // not in hidden_states. Copy it back.
    if (n_layers % 2 == 1) {
        memcpy(hidden_states, buf_a, (size_t)B * H * sizeof(float));
    }
}

// ─── Quantize lm_head from fp16 to per-row symmetric int8 ─────────────
// Reads the fp16 tensor at idx, quantizes each row to int8, stores in
// AtlasModel fields. Frees the fp16 data (saves 768 MB).
// Call after Python has created its fp32 copy (never before step 4 above).
ATLAS_API void atlas_quantize_lmhead(AtlasModel* m, int idx) {
    if (!m || idx < 0 || idx >= (int)m->tensors.size()) return;
    auto& t = m->tensors[idx];
    if (t.ttype != 2 || t.data == nullptr) return;

    int V = t.row_dim;
    int H = m->hidden_dim;
    int64_t n_vals = (int64_t)V * H;

    int8_t* i8 = (int8_t*)atlas_valloc((size_t)n_vals);
    int32_t* offs = (int32_t*)atlas_valloc((size_t)V * sizeof(int32_t));
    float* scales = (float*)atlas_valloc((size_t)V * sizeof(float));

    uint16_t* fp16 = (uint16_t*)t.data;

    for (int r = 0; r < V; r++) {
        float max_abs = 1e-5f;
        for (int c = 0; c < H; c++) {
            float v = fp16_to_fp32(fp16[r * H + c]);
            float av = fabsf(v);
            if (av > max_abs) max_abs = av;
        }
        float inv = max_abs / 127.0f;
        scales[r] = inv;
        int32_t row_sum = 0;
        for (int c = 0; c < H; c++) {
            float v = fp16_to_fp32(fp16[r * H + c]);
            int q = (int)(v / inv + 0.5f);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            i8[r * H + c] = (int8_t)q;
            row_sum += q;
        }
        offs[r] = 128 * row_sum;
    }

    if (!m->is_mapped(t.data)) atlas_vfree(t.data);
    t.data = nullptr;
    t.data_size = 0;

    m->lm_head_i8 = i8;
    m->lm_head_offsets = offs;
    m->lm_head_scales = scales;
    m->lm_head_quantized = true;

    float mb = (float)(n_vals + (int64_t)V * 6) / (1024.0f * 1024.0f);
    printf("[ATLAS] Quantized lm_head: %d × %d = %.1f MB int8\n", V, H, mb);
}

// ─── GEMV: int8 lm_head × u8 quantized activations ────────────────────
// Quantizes B hidden-state vectors [B, H] to u8, then computes full vocab
// dot products with per-row dequant. AVX2 maddubs + offset trick.
//
//   out[b][v] = (Σ_h act_u8[b][h] * W_i8[v][h] - offset[v])
//               * (max_abs_act[b] / 127) * (weight_scale[v])
//
// act: [B, H] float32    output: [B, V] float32
ATLAS_API void atlas_lmhead_gemv(AtlasModel* m, const float* act,
                                  float* output, int B) {
    if (!m || !m->lm_head_quantized) return;

    int V = m->vocab_size;
    int H = m->hidden_dim;

    // Quantize activations to u8
    uint8_t* act_u8 = (uint8_t*)atlas_valloc((size_t)B * H);
    float* max_abs = (float*)atlas_valloc((size_t)B * sizeof(float));

    for (int b = 0; b < B; b++) {
        float ma = 1e-5f;
        for (int i = 0; i < H; i++) {
            float v = fabsf(act[b * H + i]);
            if (v > ma) ma = v;
        }
        max_abs[b] = ma;
        float inv = 127.0f / ma;
        for (int i = 0; i < H; i++) {
            int q = (int)(act[b * H + i] * inv + 128.5f);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            act_u8[b * H + i] = (uint8_t)q;
        }
    }

    const int8_t* w = m->lm_head_i8;
    const int32_t* offs = m->lm_head_offsets;
    const float* scales = m->lm_head_scales;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int r = 0; r < V; r++) {
        const int8_t* wr = w + r * H;
        int32_t off = offs[r];
        float s = scales[r];

        for (int b = 0; b < B; b++) {
            const uint8_t* a = act_u8 + b * H;
            int c = 0;
            int dot = 0;
            __m256i acc = _mm256_setzero_si256();

            for (; c + 32 <= H; c += 32) {
                __m256i au = _mm256_loadu_si256((const __m256i*)(a + c));
                __m256i wv = _mm256_loadu_si256((const __m256i*)(wr + c));
                __m256i prod16 = _mm256_maddubs_epi16(au, wv);
                __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, prod32);
            }

            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            dot = _mm_cvtsi128_si32(sum128);

            for (; c < H; c++) {
                dot += (int)a[c] * (int)wr[c];
            }

            dot -= off;
            output[b * V + r] = (float)dot * (max_abs[b] / 127.0f) * s;
        }
    }

    atlas_vfree((uint8_t*)act_u8);
    atlas_vfree((uint8_t*)max_abs);
}

// ─── v1.2.0: Seed PRNG ─────────────────────────────────────────────────
ATLAS_API void atlas_set_seed(uint64_t seed) {
    xoshiro_seed(seed ? seed : 0xDEADBEEFCAFEBABEull);
}

// ─── v1.2.0: Sample one token via Gumbel-max ──────────────────────────
ATLAS_API void atlas_sample(AtlasModel* m, float* logits, int* output,
                             float temperature, int top_k, float top_p) {
    (void)m;
    if (!output || !logits) return;
    *output = gumbel_sample(logits, m ? m->vocab_size : 131072,
                             temperature, top_k, top_p);
}

// ─── v1.2.0: End-to-end generation (single C call) ───────────────────
// Builds layer index array from tensor names (cached after first call).
static void ensure_layer_idx(AtlasModel* m) {
    if (m->has_layer_idx) return;
    auto& names = m->tensor_names;
    auto find = [&](const std::string& n) -> int {
        for (int i = 0; i < (int)names.size(); i++)
            if (names[i] == n) return i;
        return -1;
    };
    m->layer_idx_cache.clear();
    m->layer_idx_cache.reserve(m->n_layers * 9);
    for (int L = 0; L < m->n_layers; L++) {
        char buf[128];
        auto push = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", L, suffix);
            m->layer_idx_cache.push_back(find(buf));
        };
        push("input_layernorm.weight");
        push("self_attn.q_proj.weight");
        push("self_attn.k_proj.weight");
        push("self_attn.v_proj.weight");
        push("self_attn.o_proj.weight");
        push("post_attention_layernorm.weight");
        push("mlp.gate_proj.weight");
        push("mlp.up_proj.weight");
        push("mlp.down_proj.weight");
    }
    m->has_layer_idx = true;
}

ATLAS_API int atlas_generate(AtlasModel* m,
    const int* input_ids, int n_input,
    uint16_t* k_cache, uint16_t* v_cache,
    int max_seq_len, int max_new_tokens,
    float temperature, int top_k, float top_p,
    int* output_ids)
{
    if (!m || !input_ids || !output_ids || n_input < 1 || max_new_tokens < 1)
        return -1;

    // Find required tensors by name
    int idx_norm = -1, idx_embed = -1;
    for (int i = 0; i < (int)m->tensor_names.size(); i++) {
        if (m->tensor_names[i] == "model.norm.weight") idx_norm = i;
        if (m->tensor_names[i] == "model.embed_tokens.weight") idx_embed = i;
    }
    if (idx_norm < 0 || idx_embed < 0) {
        fprintf(stderr, "[ATLAS] atlas_generate: missing norm/embed tensors\n");
        return -1;
    }

    int H = m->hidden_dim;
    int V = m->vocab_size;
    uint16_t* embed_w = (uint16_t*)m->tensors[idx_embed].data;
    uint8_t* norm_w = m->tensors[idx_norm].data;

    if (!m->lm_head_quantized) {
        fprintf(stderr, "[ATLAS] atlas_generate: lm_head not quantized\n");
        return -1;
    }

    ensure_layer_idx(m);
    const int* layer_idx = m->layer_idx_cache.data();

    // Allocate scratch: embedding buffer, norm scratch, logits
    float* embed_buf = (float*)atlas_valloc((size_t)n_input * H * sizeof(float));
    float* h_norm = (float*)atlas_valloc((size_t)H * sizeof(float));
    float* logits = (float*)atlas_valloc((size_t)V * sizeof(float));
    if (!embed_buf || !h_norm || !logits) {
        if (embed_buf) atlas_vfree((uint8_t*)embed_buf);
        if (h_norm) atlas_vfree((uint8_t*)h_norm);
        if (logits) atlas_vfree((uint8_t*)logits);
        return -1;
    }

    // ─── Prefill: embed all input tokens ───
    int n_gen = 0;
    for (int i = 0; i < n_input; i++) {
        int tid = input_ids[i];
        if (tid < 0 || tid >= V) tid = 0;
        for (int j = 0; j < H; j++)
            embed_buf[i * H + j] = fp16_to_fp32(embed_w[tid * H + j]);
    }

    // Build position array for prefill
    int* positions = (int*)atlas_valloc((size_t)n_input * sizeof(int));
    if (!positions) {
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); return -1;
    }
    for (int i = 0; i < n_input; i++) positions[i] = i;

    // Fused forward for all prompt tokens at once
    atlas_forward(m, embed_buf, n_input, positions,
                  k_cache, v_cache, max_seq_len, n_input,
                  layer_idx, m->n_layers);

    // Final RMSNorm + LM head — only the last prompt token's logits are needed
    {
        const float* x = embed_buf + (int64_t)(n_input - 1) * H;
        atlas_rmsnorm_f32(x, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);
    }

    // Sample first token from prefill logits
    int next_token = gumbel_sample(logits, V, temperature, top_k, top_p);
    output_ids[n_gen++] = next_token;

    if (next_token == 0) {  // EOS
        atlas_vfree((uint8_t*)embed_buf); atlas_vfree((uint8_t*)h_norm);
        atlas_vfree((uint8_t*)logits); atlas_vfree((uint8_t*)positions);
        return n_gen;
    }

    // ─── Decode loop ───
    atlas_vfree((uint8_t*)positions);
    for (int step = 1; step < max_new_tokens; step++) {
        // Embed last generated token
        int tid = next_token;
        if (tid < 0 || tid >= V) tid = 0;
        float* h = embed_buf;  // reuse embed_buf as single-token buffer
        for (int j = 0; j < H; j++)
            h[j] = fp16_to_fp32(embed_w[tid * H + j]);

        int seq_now = n_input + step;
        int pos = seq_now - 1;
        atlas_forward(m, h, 1, &pos,
                      k_cache, v_cache, max_seq_len, seq_now,
                      layer_idx, m->n_layers);

        atlas_rmsnorm_f32(h, norm_w, h_norm, H, 1e-6f);
        atlas_lmhead_gemv(m, h_norm, logits, 1);

        next_token = gumbel_sample(logits, V, temperature, top_k, top_p);
        output_ids[n_gen++] = next_token;

        if (next_token == 0) break;  // EOS
    }

    atlas_vfree((uint8_t*)embed_buf);
    atlas_vfree((uint8_t*)h_norm);
    atlas_vfree((uint8_t*)logits);

    return n_gen;
}

}  // extern "C"
