#pragma once

#include "utils.hpp"
#include "model.hpp"

#pragma pack(push, 1)
struct block_q8_0
{
    u16 d;          // 1/S
    i8  qs[32];     // dequant = qs[i] * d
};
#pragma pack(pop)
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 should be 34 bytes");

void embed_token(TensorInfo &embed, u32 token_id, f32 *out, u32 d_model);
void mat_vec_mul_q8_0(const block_q8_0 *weight, u32 n_in, u32 n_out, const f32 *input, f32 *output);
f32 dot_q8_0_f32(const block_q8_0 *row, u32 n_blocks, const f32 *x);
