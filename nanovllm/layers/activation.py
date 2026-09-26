import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 的融合激活：silu(gate) * up。"""

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up_proj 把两个投影拼在最后一维，这里拆开后逐元素相乘。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
