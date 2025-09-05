# model_config.py
from types import SimpleNamespace

# 初始化为空的全局配置
model_config = SimpleNamespace(
    num_layers=None,
    num_heads=None,
    hidden_size=None,
    seq_len=None,
    vocab_size=None,
    dtype=None,
    micro_bs=None,
    gloa1_bs=None,
    model_path=None,
    top_k=None,
    if_moe=None,
)