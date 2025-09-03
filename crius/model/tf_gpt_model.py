import torch
import torch.nn as nn

class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=0,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.device = device
        self.dtype = dtype

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, device=config.device, dtype=config.dtype)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd, device=config.device, dtype=config.dtype)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, device=config.device, dtype=config.dtype)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states
    
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, device=config.device, dtype=config.dtype)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, device=config.device, dtype=config.dtype)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5 if config.scale_attn_weights else 1.0

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, device=config.device, dtype=config.dtype)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd, device=config.device, dtype=config.dtype)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2), qkv)

        att = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        att = att.masked_fill(causal_mask[:T, :T] == 0, float('-inf'))
        att = att.softmax(-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, device=config.device, dtype=config.dtype)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, device=config.device, dtype=config.dtype)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x    