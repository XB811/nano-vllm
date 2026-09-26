import torch
from torch import nn


class Sampler(nn.Module):
    """按请求 temperature 执行 categorical sampling。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Gumbel-max 等价实现：softmax 概率除以 Exp(1) 噪声后取 argmax，
        # 避免显式构造 multinomial 的额外开销。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
