import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """将一行 [num_kv_heads, head_dim] 的 K/V 写入分页 cache。"""
    # 当前program的编号 （范围从0～ N-1）
    idx = tl.program_id(0)
    # 查询这个 token 应写入哪个缓存槽位
    slot = tl.load(slot_mapping_ptr + idx)
    # -1 表示该位置没有对应的物理槽位（例如 CUDA Graph 的 padding 行）。
    if slot == -1: return
    # 计算出一个token的k/v所有元素的偏移量
    # 例：第i个token计算得到的key_offsets = [i * key_stride, i * key_stride + 1, i * key_stride + 2, ......  i * key_stride + D -1]
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 加载当前token的所有数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 写入目标缓存槽位
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """检查张量布局并启动 Triton kernel，把当前 token 的 K/V 写入缓存。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 检查每个 head 内部是否连续
    # stride(i) 表示：沿第 i 个维度移动一步，需要在内存中跨过多少个元素
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # 校验槽位数等于本次批处理的token数量
    assert slot_mapping.numel() == N
    # [(N,)] Triton 的内核启动语法，启动N个实例，一个实例处理一个token
    # key.stride(0) 是相邻两个token之间的距离多少个元素；D是一个token的k/v有多少个元素；
    # 因为tp切分了 k、v，相邻两个token的元素并不相邻，因此 key.stride(0) >= D
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """同时支持 prefill 和 decode 的分页 KV-cache attention。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # context 由 ModelRunner 在本轮前写入，是 attention kernel 的隐式参数。
        context = get_context()
        # cache 形状：[num_blocks, block_size, num_kv_heads, head_dim]
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # prefill 写入新 token，decode 写入当前生成位置；之后都可被历史 attention 读取。
            # warmup 阶段 self.k_cache = self.v_cache = torch.tensor([]) numel值为0,跳过执行
            # allocate_kv_cache()方法中，重新为k_cache v_cache 分配了张量，numel不为0了
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                # 命中 prefix cache 时，flash-attn 通过 block table 从物理页读取
                # 历史 K/V；当前 q/k/v 只对应尚未缓存的 token。
                k, v = k_cache, v_cache
            # packed varlen attention 用 cu_seqlens 分隔不同请求，避免 padding。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # decode 每个请求只有一个 query，历史 K/V 完整存放在分页 cache 中。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
