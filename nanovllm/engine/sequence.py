from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求在调度器中的生命周期状态。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一个生成请求及其在 KV cache 中的页表元数据。

    ``token_ids`` 保存完整 prompt 和已生成 token；``block_table`` 则把逻辑
    token block 映射到 BlockManager 管理的物理 KV cache block。
    """

    # 所有 Sequence 共享页大小；LLMEngine 会在创建请求前同步配置值。
    block_size = 256
    # 进程内单调递增的请求 ID，便于引擎在乱序完成时恢复输出顺序。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        # 采样参数复制到请求对象，后续调度阶段无需再携带 SamplingParams。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 向上取整：最后一个逻辑 block 可以只包含部分 token。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 ``i`` 个逻辑 block 的 token，用于前缀哈希和缓存匹配。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """把模型采样出的新 token 追加到请求，并更新长度信息。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """只序列化跨进程调度所需状态。

        prefill 阶段仍需完整 token 列表；decode 阶段只需最后一个 token，避免
        通过共享内存反复传输不断增长的完整序列。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """恢复 ``__getstate__`` 保存的请求状态。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
