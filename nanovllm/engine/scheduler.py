from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """负责连续批处理，并协调请求状态与 KV cache 页表。

    一次 ``schedule`` 要么安排一批 prefill token，要么安排一批 decode token；
    两种阶段不会在同一个 batch 中混合，以便 ModelRunner 使用对应的数据布局。
    """

    def __init__(self, config: Config):
        # 单批次最多处理多少个请求
        self.max_num_seqs = config.max_num_seqs
        # 轮调度中，最多允许处理多少个 token
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新请求放入等待队列，等待后续批处理。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # 本轮调度中，已经处理的token数量
        num_batched_tokens = 0

        # prefill：优先消化等待队列。一个请求可能因 token 上限被切成多个
        # 只允许本轮第一个请求被拆分；后续请求必须完整放入剩余 token 预算，
        # 避免同一批次产生多个未完成的 prefill chunk。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 取队列的队头请求
            seq = self.waiting[0]
            # 本batch中，剩余的prefill token额度
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 计算num_tokens：seq需要prefill的token数量
            if not seq.block_table:
                # 首次调度时尝试复用完整的、已哈希的前缀 block；
                # 返回 -1：表示物理 KV cache 不足，当前请求和后续请求都暂不调度。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 如果block_table 存在，说明该请求被chunked prefill, seq.num_cached_tokens = 已经prefill的token数量
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 每批次只允许第一个请求能够chunked prefill
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本轮实际处理的 token 数。如果该请求没有prefill完成（即chunk prefill），则下一轮会从 seq.num_cached_tokens 继续。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 如果请求的缓存token数 + 本地prefix的token数量 = 请求的总token数量，说明该请求已经全部prefill，需要更新seq.status
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # prefill batch 直接返回，不和 decode 混合；这也保证了注意力输入的
        # cu_seqlens 与 block table 具有统一语义。
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode：每个运行中的请求每轮只追加一个 token。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 没有新页可追加时，优先抢占队尾请求；若无其他请求可让出，
                # 则抢占当前请求并回到 waiting，避免整个调度器死锁。
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 如果 token 恰好跨越 block 边界，这里会为它分配新页。
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """释放请求占用的物理 block，并把请求退回等待队列。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """消费模型输出，更新前缀缓存、请求长度和完成状态。"""
        for seq, token_id in zip(seqs, token_ids):
            # 先把本轮刚填满的 block 写入前缀哈希表，再推进 cached 计数。
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 尚未覆盖完整 prompt，此轮没有可采样的输出。
                continue
            seq.append_token(token_id)
            # EOS 或达到用户限制时释放整个请求的 KV cache。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
