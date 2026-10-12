import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """按词表维切分 embedding，并在 forward 后合并各 rank 的局部结果。"""

    def __init__(
        self,
        num_embeddings: int, # 词表token数量
        embedding_dim: int, # embedding后向量长度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() # 当前进程的 rank。
        self.tp_size = dist.get_world_size() # TP GPU数量
        assert num_embeddings % self.tp_size == 0 # 校验是否能完整切分
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # 沿指定维度取连续片段。这里沿第 0 维，也就是词表维度切分
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 不属于本 rank 词表分片的 token 映射为 0（即self.vocab_start_idx），之后用 mask 清零其 embedding。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y   # 不属于本rank的token，全部置0；使用广播机制计算 [N, 1] * [N, embedding_dim]
            # 每个 token 只有一个 rank 有效，all_reduce 后得到完整 embedding。
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """按词表维切分 LM head；rank 0 收集完整 logits 用于采样。"""

    def __init__(
        self,
        num_embeddings: int, # 词表长度
        embedding_dim: int, # embedding维度, 等于hidden_size
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context() # 从上下文获取当前状态
        if context.is_prefill:
            # packed prefill 会产生每个 prompt 的所有 hidden states，但只需每条
            # prompt 最后一个位置的 logits 来采样第一个 completion token。

            # 本批次seqs 长度为 [3, 2, 4]，则cu_seqlens_q值为 [0, 3, 5, 9]，
            # last_indices为 [2, 4, 8],对应各个seq的最后一位token在x中的位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight) # logits = x @ self.weight.T， self.weight的shape是 num_embeddings * embedding_dim
        if self.tp_size > 1:
            # logits 沿词表维拼接；非 0 rank 不需要保留完整 logits。
            # 如果是rank 0，logits.shape= n * (num_embeddings/tp_size)
            # 则all_logits = [空 Tensor, 空 Tensor] length = tp_size, 值全为0
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 将所有 rank 上的 logits 收集到 rank 0。
            # 参数含义：
            # - logits：当前 rank 要发送的数据；
            # - all_logits：接收列表，只需要在目标 rank，即 rank 0 上提供；
            # - 0：目标 rank 是 rank 0。
            # 结果 all_logits = [
            #     rank0 的 logits,
            #     rank1 的 logits,
            #     rank2 的 logits,
            # ]
            dist.gather(logits, all_logits, 0)
            # 如果是rank0,则按照最后一维拼接张量， 最后一维 即 词表维度
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
