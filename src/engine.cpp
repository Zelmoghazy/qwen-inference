#include "engine.hpp"

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline f32 ggml_compute_fp16_to_fp32(uint16_t h) 
{
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline uint16_t ggml_compute_fp32_to_fp16(float f) 
{
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }
    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}


inline f32 f16_to_f32(u16 h)
{
    Profile("f16 to f32");

    u32 sign = (u32)(h & 0x8000u) << 16;
    u32 exp  = (h >> 10) & 0x1Fu;
    u32 mant =  h        & 0x3FFu;
    u32 bits;

    if (exp == 0)
    {
        if (mant == 0)
        {
            bits = sign; // +/- zero
        }
        else
        {
            // subnormal half -> normalize into a normal float
            exp = 1;
            while ((mant & 0x400u) == 0)
            {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    }
    else if (exp == 0x1F)
    {
        bits = sign | 0x7F800000u | (mant << 13); // inf / nan
    }
    else
    {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    f32 f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline f32 dot_q8_0_f32(const block_q8_0 *row, u32 n_blocks, const f32 *x)
{
    f32 acc = 0.0f;
    for (u32 l = 0; l < n_blocks; l++)
    {
        f32 delta = ggml_compute_fp16_to_fp32(row[l].d);

        const int8_t *qs = row[l].qs;
        const f32    *xb = x + (u64)l * 32;

        f32 block_acc = 0.0f;
        for (u32 i = 0; i < 32; i++){
            block_acc += (f32)qs[i] * xb[i];
        }

        acc += block_acc * delta;
    }
    return acc;
}

void matmul_q8_0(const block_q8_0 *weight, u32 n_in, u32 n_out, const f32 *input, f32 *output)
{
    Profile("Matrix Multiplication");
    assert(n_in % 32 == 0 && "q8_0 rows must be a multiple of the 32-element block size");
    u32 blocks_per_row = n_in / 32;

    for (u32 row = 0; row < n_out; row++)
    {
        const block_q8_0 *row_blocks = weight + (u64)row * blocks_per_row;
        output[row] = dot_q8_0_f32(row_blocks, blocks_per_row, input);
    }
}

inline u64 q8_0_nbytes(u64 n_elements)
{
    assert(n_elements % 32 == 0);
    return (n_elements / 32) * sizeof(block_q8_0);
}


void embed_token(TensorInfo &embed, u32 token_id, f32 *out, u32 d_model)
{
    const block_q8_0 *table = (const block_q8_0*)embed.tensor_data;
    const block_q8_0 *row   = table + (u64)token_id * (d_model / 32);
    for (u32 i = 0; i < d_model; i++){
        out[i] = ggml_compute_fp16_to_fp32(row[i/32].d) * row[i/32].qs[i%32];
    }
}

void normalize_l2(f32 *vec, u32 n)
{
    f64 sum_sq = 0.0;                       
    for (u32 i = 0; i < n; i++){                          
        sum_sq += (f64)vec[i] * (f64)vec[i];
    }

    f32 norm = (f32)std::sqrt(sum_sq);
    if (norm > 0.0f)
    {
        for (u32 i = 0; i < n; i++){
            vec[i] /= norm;
        }
    }
}