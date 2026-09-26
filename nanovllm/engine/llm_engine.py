import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """面向用户的离线生成入口，串联 tokenizer、调度器和模型执行器。"""

    def __init__(self, model, **kwargs):
        # 只把 Config 支持的参数传入配置对象，避免 API 层参数污染内部配置。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # Sequence 与 BlockManager 必须使用同一个页大小。
        Sequence.block_size = config.kvcache_block_size
        # TP 时 GPU0 由主进程管理，其余 GPU 由子进程管理；主进程负责调度和采样。
        self.ps = []
        # 每个 Event 用来通知对应的 TP 子进程共享内存中有新的方法调用。
        self.events = []
        # spawn 创建干净的 Python 进程，避免 CUDA/NCCL 状态被 fork 继承。
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 每个子进程绑定一个 rank，并在 ModelRunner 内初始化本地分片。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主进程使用 rank 0，同时保存所有子进程的同步事件。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # Scheduler 需要 EOS token 判断请求何时完成。
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 当进程停止时，触发exit()方法，关闭model_runner，避免内存泄漏
        atexit.register(self.exit)

    def exit(self):
        """通知各进程退出，并等待子进程回收资源。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将字符串或 token 列表转换为可调度的 Sequence。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 初始化成Sequence
        seq = Sequence(prompt, sampling_params)
        # 添加到等待队列
        self.scheduler.add(seq)

    def step(self):
        """执行一轮调度、模型前向、采样和请求状态更新。"""
        seqs, is_prefill = self.scheduler.schedule()
        # prefill 的工作量按处理 token 计；decode 每个序列恰好生成一个 token。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """批量生成文本，并按请求 ID 返回已完成请求的结果。"""
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 处理采样参数，如果采样参数不是list，进行复制
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 添加请求
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            # 循环直到 waiting 和 running 都为空；step 内部可能只完成部分请求。
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                # 请求可能因不同长度而乱序完成，先按 seq_id 暂存，最后统一排序。
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 对齐输入 prompts 的顺序，并把 token ids 解码成用户可直接使用的文本。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
