#ifndef ATLAS_KERNEL_HPP
#define ATLAS_KERNEL_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

constexpr int   TQ1_TRITS_PER_BYTE = 5;
constexpr float TQ1_SCALE          = 1.0f;

alignas(32) static float tq1_lut0[256];
alignas(32) static float tq1_lut1[256];
alignas(32) static float tq1_lut2[256];
alignas(32) static float tq1_lut3[256];
alignas(32) static float tq1_lut4[256];

static float* tq1_luts[5] = { tq1_lut0, tq1_lut1, tq1_lut2, tq1_lut3, tq1_lut4 };

inline void init_tq1_lut() {
    for (int b = 0; b < 256; ++b) {
        int temp = b;
        tq1_lut0[b] = (float)((temp % 3) - 1); temp /= 3;
        tq1_lut1[b] = (float)((temp % 3) - 1); temp /= 3;
        tq1_lut2[b] = (float)((temp % 3) - 1); temp /= 3;
        tq1_lut3[b] = (float)((temp % 3) - 1); temp /= 3;
        tq1_lut4[b] = (float)((temp % 3) - 1);
    }
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign  = (uint32_t)((h >> 15) & 0x0001);
    uint32_t exp   = (uint32_t)((h >> 10) & 0x001F);
    uint32_t mant  = (uint32_t)( h        & 0x03FF);
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            int e = -1;
            while ((mant & 0x0400) == 0) { mant <<= 1; e--; }
            mant &= 0x03FF;
            f32 = (sign << 31) | ((uint32_t)(exp + 112 + e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

struct TQ1Mat {
    int      rows;
    int      cols;
    int      packed_cols;
    float*   scales;
    uint8_t* data;
};

inline void mv_tq10_avx2_omp(
    TQ1Mat* w,
    float*  activations,
    float*  output,
    int     n_tokens
) {
    const int rows        = w->rows;
    const int packed_cols = w->packed_cols;
    const int act_stride  = packed_cols * TQ1_TRITS_PER_BYTE;

    #pragma omp parallel for if (rows > 16)
    for (int r = 0; r < rows; r++) {
        const float   scale    = w->scales[r];
        const uint8_t* row_data = w->data + r * packed_cols;

        for (int t = 0; t < n_tokens; t++) {
            const float* act_base = activations + t * act_stride;
            __m256 vacc = _mm256_setzero_ps();

            int k;
            for (k = 0; k + 8 <= packed_cols; k += 8) {
                __m128i b8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(row_data + k));
                __m256i b32 = _mm256_cvtepu8_epi32(b8);

                int act_k = k * TQ1_TRITS_PER_BYTE;

                for (int p = 0; p < TQ1_TRITS_PER_BYTE; p++) {
                    __m256 wvals = _mm256_i32gather_ps(
                        tq1_luts[p], b32, sizeof(float));

                    __m256i act_idx = _mm256_set_epi32(
                        act_k + p + 7 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 6 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 5 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 4 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 3 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 2 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 1 * TQ1_TRITS_PER_BYTE,
                        act_k + p + 0 * TQ1_TRITS_PER_BYTE);
                    __m256 avals = _mm256_i32gather_ps(act_base, act_idx, sizeof(float));

                    vacc = _mm256_fmadd_ps(wvals, avals, vacc);
                }
            }

            __m128 lo = _mm256_castps256_ps128(vacc);
            __m128 hi = _mm256_extractf128_ps(vacc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float total = _mm_cvtss_f32(sum);

            for (; k < packed_cols; k++) {
                uint8_t b = row_data[k];
                total += tq1_lut0[b] * act_base[k * TQ1_TRITS_PER_BYTE + 0];
                total += tq1_lut1[b] * act_base[k * TQ1_TRITS_PER_BYTE + 1];
                total += tq1_lut2[b] * act_base[k * TQ1_TRITS_PER_BYTE + 2];
                total += tq1_lut3[b] * act_base[k * TQ1_TRITS_PER_BYTE + 3];
                total += tq1_lut4[b] * act_base[k * TQ1_TRITS_PER_BYTE + 4];
            }

            output[t * rows + r] = total / scale;
        }
    }
}

inline int tq1_load_row(uint8_t* src, float* scale_out, uint8_t* dst, int packed_cols) {
    uint16_t scale_raw;
    memcpy(&scale_raw, src, 2);
    *scale_out = fp16_to_fp32(scale_raw);
    memcpy(dst, src + 2, packed_cols);
    return 2 + packed_cols;
}

#endif // ATLAS_KERNEL_HPP
