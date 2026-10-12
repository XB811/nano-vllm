from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """当前 forward 所需的变长 attention 和分页 cache 元数据。"""
    # 本轮计算阶段：True 为 prefill（处理 prompt 或其一个 chunk），
    # False 为 decode（每条序列处理一个输入 token，以预测下一个 token）。
    is_prefill: bool = False

    # prefill 使用，shape = [batch_size + 1]。
    # 各序列本轮新计算的 Q 长度的累积和，以 0 开头，用于划分拼接后的 Q。
    # 例如 Q 长度为 [3, 2, 4]，则值为 [0, 3, 5, 9]，第 i 条序列的 Q
    # 位于 q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]；decode 时为 None。
    cu_seqlens_q: torch.Tensor | None = None

    # prefill 使用，shape = [batch_size + 1]。
    # 各序列本轮 attention 可访问的 K/V 长度的累积和，以 0 开头。
    # 每条序列的 K/V 长度 = 已缓存 token 数 + 本轮新处理 token 数；
    # 无历史缓存时与 cu_seqlens_q 相同，复用前缀或继续分块 prefill 时可更大。
    # 使用分页 cache 时，这些值描述逻辑长度，物理位置由 block_tables 指定；
    # decode 时为 None。
    cu_seqlens_k: torch.Tensor | None = None

    # prefill 使用：本轮各序列 Q 长度的最大值，而非整个 batch 的 token 总数。
    # 例如 Q 长度为 [3, 2, 4]，则值为 4；decode 时保持默认值 0。
    max_seqlen_q: int = 0

    # prefill 使用：各序列可访问的 K/V 长度的最大值，包含已缓存的历史 token。
    # 例如已缓存 6 个 token、本轮处理 2 个 token，则该序列的 K/V 长度为 8；
    # decode 时保持默认值 0。
    max_seqlen_k: int = 0

    # prefill 和 decode 都使用，shape = [本轮输入 token 总数]。
    # 按本轮输入 token 的排列顺序，指定其新 K/V 写入 KV cache 的物理槽位：
    # slot = 物理 block ID * block_size + 块内偏移（从 0 开始）。
    # prefill 每条序列可对应多个槽位，decode 每条序列对应一个槽位；
    # -1 表示跳过写入，例如 CUDA Graph 补齐 batch 时的无效 token。
    slot_mapping: torch.Tensor | None = None

    # 仅 decode 使用，shape = [batch_size]。
    # 每条序列本轮可访问的有效 K/V 数，包含历史 token 和当前输入 token，
    # 不包含尚未预测出的下一个 token，也不包含 cache block 的空余槽位。
    # 例如两条序列长度为 8 和 13，则值为 [8, 13]；prefill 时为 None。
    context_lens: torch.Tensor | None = None

    # 分页 KV cache 的地址映射，shape = [batch_size, batch 内最大 block 数]。
    # block_tables[i, j] 表示第 i 条序列的第 j 个逻辑 block 对应的物理 block ID；
    # 物理 block ID 从 0 开始，各行不足的部分填 -1（无效 block）。
    # 例如 block_size = 4、某行是 [3, 7]，则该序列位置 0~3 的 K/V 在
    # 物理 block 3，位置 4~7 的 K/V 在物理 block 7。
    # decode 使用；prefill 需要读取历史缓存（前缀复用或后续 chunk）时也使用，
    # 无历史缓存的 prefill 为 None，此时直接对本轮拼接的 K/V 做 attention。
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """取得当前进程的隐式 attention context。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """在一次模型前向前设置 context；参数由 ModelRunner 按阶段填充。"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """前向完成后清空全局 context，避免下一批误用旧页表。"""
    global _CONTEXT
    _CONTEXT = Context()
