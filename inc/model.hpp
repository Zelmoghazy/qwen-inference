#pragma once

#include <utils.hpp>



#define KV_K_AT(ctx, layer, pos) \
    MAT3D_AT((ctx).kv_cache_k, (layer), (pos), 0, (ctx).max_seq, (ctx).kv_dim)

#define KV_V_AT(ctx, layer, pos) \
    MAT3D_AT((ctx).kv_cache_v, (layer), (pos), 0, (ctx).max_seq, (ctx).kv_dim)

#define KV_HEAD_K(ctx, layer, pos, kv_head) \
    MAT4D_AT((ctx).kv_cache_k, (layer), (pos), (kv_head), 0, \
             (ctx).max_seq, (ctx).kv_dim / (ctx).head_dim, (ctx).head_dim)

#define KV_HEAD_V(ctx, layer, pos, kv_head) \
    MAT4D_AT((ctx).kv_cache_v, (layer), (pos), (kv_head), 0, \
             (ctx).max_seq, (ctx).kv_dim / (ctx).head_dim, (ctx).head_dim)


struct ModelConfig
{
    u32 context_length;
    u32 embedding_length;
    u32 feed_forward_length;
    u32 attention_head_count;
    u32 attention_head_count_kv;
    u32 block_count;
    u32 alignment = 32;         // if absent, it defaults to 32 bytes
    f32 rope_freq;
    f32 rms_epsilon;
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
    std::string             name;
    ModelConfig             cfg;
    TensorInfo              token_embd;
    TensorInfo              output_norm;
    TensorInfo              output;     
    std::vector<BlockInfo>  blocks;
};