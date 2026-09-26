from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """当前 forward 所需的变长 attention 和分页 cache 元数据。"""
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
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
