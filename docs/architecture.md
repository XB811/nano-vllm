# Nano-vLLM 架构与调用链路

依据当前工作区源码整理。图中方法名对应实际实现；本文描述离线批量推理流程。

## 1. 整体架构

```mermaid
flowchart TB
    U["用户代码：example.py / bench.py"] --> API["LLM：继承 LLMEngine"]
    SP["SamplingParams：温度、生成长度、EOS 策略"] --> API
    CFG["Config：模型配置、批次预算、TP、缓存页大小"] --> API
    subgraph CPU["主进程：请求编排"]
        API --> TOK["AutoTokenizer：编码 / 解码"]
        API --> SCH["Scheduler：waiting / running 队列"]
        SCH --> SEQ["Sequence：token、状态、block_table"]
        SCH --> BM["BlockManager：物理页编号、引用计数、前缀哈希"]
    end
    API --> RUN["ModelRunner：每个 TP rank 一个实例"]
    SEQ -. "本轮序列与页表" .-> RUN
    subgraph GPU["各 rank 的模型执行与 GPU 存储"]
        RUN --> PREP["prepare_prefill / prepare_decode"]
        PREP --> CTX["Context：位置、长度、slot_mapping、block_tables"]
        RUN --> MODEL["Qwen3ForCausalLM：模型分片"]
        LOAD["load_model：加载 safetensors 权重分片"] --> MODEL
        MODEL --> LAYERS["Embedding → Decoder Layers → RMSNorm → LM Head"]
        LAYERS --> ATT["Attention：Triton 写缓存 + FlashAttention"]
        CTX -. "Attention / LM Head 读取" .-> LAYERS
        RUN --> KV["分页 KV Cache：各层 K/V 张量"]
        ATT <--> KV
        MODEL --> SAMPLE["Sampler：仅 rank 0 采样"]
    end
    SAMPLE --> POST["Scheduler.postprocess：追加 token / 完成 / 释放页"]
    POST --> SCH
    POST --> OUT["LLMEngine：收集结果、按请求顺序解码"]
```

- `BlockManager` 管理 CPU 上的页元数据，`ModelRunner.allocate_kv_cache()` 分配 GPU 张量并将每层缓存注入 `Attention`。
- `Sequence.block_table` 把请求的逻辑页映射到物理页；`slot_mapping` 指定本轮新 K/V 的写入槽位。
- `Context` 是每个进程的全局上下文，由执行器设置，Attention 与 LM Head 隐式读取。
- 当前执行器直接构造 `Qwen3ForCausalLM`。

## 2. 初始化调用链

```mermaid
flowchart TD
    A["LLM(model, kwargs)"] --> B["LLMEngine.__init__"]
    B --> C["Config：读取模型配置；同步 Sequence.block_size"]
    C --> D["TP > 1：spawn rank 1..N-1 子进程"]
    D --> E["主进程创建 rank 0 ModelRunner"]
    E --> F["各 rank：初始化 NCCL、绑定 GPU"]
    F --> G["构建 Qwen3ForCausalLM → load_model → Sampler"]
    G --> H["warmup_model：前向预热、测量显存峰值"]
    H --> I["allocate_kv_cache：计算容量、分配各层缓存"]
    I --> J["非 enforce_eager：capture_cudagraph"]
    J --> K["TP > 1：连接共享内存；子进程进入 loop"]
    K --> L["rank 0：加载 tokenizer、设置 EOS、创建 Scheduler"]
    L --> M["注册 exit 清理回调；就绪"]
```

子进程与主进程的模型初始化并行进行；图中汇总了每个 rank 都会执行的步骤。Scheduler 在 rank 0 完成缓存分配后创建，因此能获取实际 KV 页数量。

## 3. generate 端到端调用链

```mermaid
sequenceDiagram
    participant U as 用户
    participant E as LLMEngine
    participant T as Tokenizer
    participant S as Scheduler
    participant B as BlockManager
    participant R as ModelRunner
    participant M as Qwen3ForCausalLM
    participant P as Sampler(rank 0)
    U->>E: generate(prompts, sampling_params)
    loop 每个请求
        E->>T: encode(prompt)，仅字符串输入
        E->>E: 创建 Sequence
        E->>S: add(seq)，加入 waiting
    end
    loop waiting 或 running 非空
        E->>E: step()
        E->>S: schedule()
        alt 可以调度 Prefill
            S->>B: can_allocate / allocate，复用前缀页
            S->>S: 按 token 预算安排完整或分块 Prefill
        else 调度 Decode
            S->>B: can_append / may_append
            opt 页不足
                S->>S: preempt：请求退回 waiting
                S->>B: deallocate：释放页引用
            end
        end
        S-->>E: seqs, is_prefill
        E->>R: call("run", seqs, is_prefill)
        Note over R: TP 模式同时通知其他 rank 执行
        R->>R: prepare_prefill 或 prepare_decode；set_context
        R->>R: rank 0 prepare_sample
        R->>M: run_model：直接前向或重放 CUDA Graph
        M->>M: compute_logits；TP 时 gather 到 rank 0
        M-->>R: logits
        R->>P: 按 temperature 采样
        P-->>R: token_ids
        R->>R: reset_context
        R-->>E: token_ids
        E->>S: postprocess(seqs, token_ids, is_prefill)
        S->>B: hash_blocks：登记已填满的页
        alt Prefill chunk 尚未完成
            S->>S: 更新缓存进度，忽略本轮采样 token
        else 可以接受生成 token
            S->>S: append_token
            opt EOS 或达到 max_tokens
                S->>B: deallocate
                S->>S: FINISHED，移出 running
            end
        end
        E->>E: 收集本轮已完成请求
    end
    E->>T: 按 seq_id 排序后 decode(completion_token_ids)
    E-->>U: 列表，每项包含 text 和 token_ids
```

每轮 batch 只包含 Prefill 或 Decode；Prefill 优先。只有本轮第一个 Prefill 请求允许被拆分，后续请求必须完整放入剩余预算。Prefill 最后一个 chunk 产生第一个有效生成 token，Decode 此后每轮为每条选中序列生成一个 token。

## 4. 模型与 Attention 内部链路

```mermaid
flowchart TD
    IN["input_ids + positions"] --> EMB["VocabParallelEmbedding"]
    EMB --> N1["Decoder：input RMSNorm + 残差处理"]
    N1 --> QKV["QKVParallelLinear → 拆分 Q/K/V"]
    QKV --> QN["无 qkv_bias 时对 Q/K 做 RMSNorm"]
    QN --> ROPE["RoPE：对 Q/K 加入位置信息"]
    ROPE --> STORE["store_kvcache：Triton 写入当前 K/V"]
    STORE --> PHASE{"Prefill？"}
    PHASE -->|是| PF["flash_attn_varlen_func"]
    PHASE -->|否| DC["flash_attn_with_kvcache"]
    CACHE["分页 KV Cache"] --> DC
    CACHE -->|"前缀复用 / 后续 Prefill chunk"| PF
    PF --> O["o_proj：RowParallelLinear"]
    DC --> O
    O --> N2["post_attention RMSNorm + 残差融合"]
    N2 --> MLP["MLP：gate_up_proj → SiluAndMul → down_proj"]
    MLP --> NEXT{"还有 Decoder 层？"}
    NEXT -->|是| N1
    NEXT -->|否| NORM["最终 RMSNorm + 残差融合"]
    NORM --> HEAD["ParallelLMHead：Prefill 选每个 chunk 末位置"]
    HEAD --> GATHER["TP 时汇聚词表 logits 到 rank 0"]
    GATHER --> SAM["Sampler → 下一个 token"]
```

初始化 warmup 尚未分配缓存时跳过 `store_kvcache`。无历史缓存的 Prefill 直接使用本轮 K/V；有历史缓存时通过 block table 读取缓存。未完成 chunk 的末位置 logits 仍会被采样，但调度器丢弃该 token。

`run_model()` 在 Prefill、`enforce_eager=True` 或 Decode batch 大于 512 时走直接前向；其余 Decode 走 CUDA Graph。图捕获覆盖模型主干，`compute_logits()` 和采样在 replay 之后执行。`enforce_eager` 不会关闭代码中独立的 `torch.compile` 装饰器。

## 5. 多 GPU 张量并行链路

```mermaid
sequenceDiagram
    participant E as LLMEngine / Scheduler
    participant R0 as rank 0 ModelRunner
    participant IPC as SharedMemory + Event
    participant RN as rank 1..N-1 ModelRunner
    E->>R0: call("run", seqs, is_prefill)
    R0->>IPC: write_shm：pickle 调用元数据；设置各 Event
    IPC->>RN: loop → read_shm → call → run
    Note over R0,RN: 各 rank 独立准备输入与 Context，执行各自模型分片
    R0->>RN: NCCL collective：Embedding / RowParallelLinear all_reduce
    RN-->>R0: ParallelLMHead gather 局部词表 logits
    R0->>R0: 拼接完整 logits；Sampler 采样
    R0-->>E: token_ids
    E->>E: postprocess 更新请求；下一轮携带新序列状态
```

共享内存传递方法名和请求元数据；GPU 张量通信通过 NCCL 完成。调度与采样集中在 rank 0，其余 rank 参与模型计算。每个 rank 保存自己 KV heads 对应的缓存分片。

## 源码索引

| 职责 | 文件 |
| --- | --- |
| 对外入口与生成循环 | [llm.py](../nanovllm/llm.py)、[llm_engine.py](../nanovllm/engine/llm_engine.py) |
| 调度、抢占、请求状态 | [scheduler.py](../nanovllm/engine/scheduler.py)、[sequence.py](../nanovllm/engine/sequence.py) |
| 缓存页分配与前缀复用 | [block_manager.py](../nanovllm/engine/block_manager.py) |
| 输入准备、GPU 缓存、CUDA Graph、TP 进程通信 | [model_runner.py](../nanovllm/engine/model_runner.py) |
| Qwen3 模型结构 | [qwen3.py](../nanovllm/models/qwen3.py) |
| Attention 与缓存写入 | [attention.py](../nanovllm/layers/attention.py) |
| 并行线性层、Embedding、输出头 | [linear.py](../nanovllm/layers/linear.py)、[embed_head.py](../nanovllm/layers/embed_head.py) |
| 采样与隐式上下文 | [sampler.py](../nanovllm/layers/sampler.py)、[context.py](../nanovllm/utils/context.py) |
