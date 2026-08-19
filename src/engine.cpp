#include "engine.hpp"

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16


static inline f32 f16_to_f32(u16 h)
{
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

f32 dot_q8_0_f32(const block_q8_0 *row, u32 n_blocks, const f32 *x)
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

void mat_vec_mul_q8_0(const block_q8_0 *weight, u32 n_in, u32 n_out, const f32 *input, f32 *output)
{
    ZoneScopedNC("Matrix Multiplication", tracy::Color::Tomato);

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