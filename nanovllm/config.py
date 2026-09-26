import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """推理引擎的全局配置。

    大部分字段来自 ``LLM`` 的关键字参数；``hf_config``、EOS token 和 KV
    cache 容量则会在初始化模型或 tokenizer 后补齐。
    """

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # enforce_eager = False 时 CUDA 图（CUDA Graphs）捕获和 torch.compile 编译优化
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """校验运行约束，并读取 Hugging Face 模型配置。"""
        assert os.path.isdir(self.model)
        # KV cache 页大小必须是 256 的倍数，以满足当前 CUDA kernel/block table
        # 的布局约束（默认值为 256 token/page）。
        assert self.kvcache_block_size % 256 == 0
        # 权重和 attention head 会按 GPU 数量切分，最多支持 8 路并行。
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 模型可能支持更长上下文，但引擎必须遵守模型的位置编码上限。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
