#pragma once

#include <utils.hpp>

struct ModelConfig
{
    u32 embedding_length;
    u32 block_count;
    u32 attention_head_count;
    u32 attention_head_count_kv;
    u32 feed_forward_length;
    u32 alignment = 32;         // if absent, it defaults to 32 bytes
    f32 rms_epsilon;
    f32 rope_freq;
};

// Tensor infos, which can be used to locate the tensor data.
struct TensorInfo
{
    u64        dims[4];                // Currently at most 4
    u8         n_dims;
    u64        offset          = 0;    // relative to the `tensor_data` , Must be a multiple of `ALIGNMENT`.
    u32        type            = 0; 
    const u8  *tensor_data     = nullptr; 
    bool       present         = false;   
};

struct BlockInfo
{
    TensorInfo attn_norm;
    TensorInfo attn_q;
    TensorInfo attn_q_bias;
    TensorInfo attn_k;
    TensorInfo attn_k_bias;
    TensorInfo attn_v;
    TensorInfo attn_v_bias;
    TensorInfo attn_output;
    TensorInfo ffn_norm;
    TensorInfo ffn_gate;
    TensorInfo ffn_up;
    TensorInfo ffn_down;
};

struct ModelInfo
{
    ModelConfig             cfg;
    TensorInfo              token_embd;
    TensorInfo              output_norm;
    TensorInfo              output;     
    std::vector<BlockInfo>  blocks;
};