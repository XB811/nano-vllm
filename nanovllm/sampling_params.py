from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """一次生成请求使用的采样参数。"""

    temperature: float = 1.0
    max_tokens: int = 64
    # ignore_eos = True 时，会强制模型忽略结束符（EOS token）并继续生成文本。常用于Benchmark场景时
    # 一般情况下，默认为False即可
    ignore_eos: bool = False

    def __post_init__(self):
        # 当前 Sampler 只实现按温度的随机采样；温度过小会退化为 greedy，
        # 因此这里明确拒绝，而不是静默改变用户的采样语义。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
