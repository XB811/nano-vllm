import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认的整块权重复制逻辑。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """逐个读取 safetensors，并按模型声明的映射加载 TP 权重分片。

    ``packed_modules_mapping`` 把 Hugging Face 的 q/k/v、gate/up 参数名映射到
    实际合并参数；对应参数的 ``weight_loader`` 负责选择当前 rank 的切片。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
