"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import copy
import math
from functools import partial
from dataclasses import dataclass, field

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.attention.kimi import KimiLinearAttention
from nanochat.kv_cache import KVCache
from nanochat.utils import apply_rotary_pos_emb, rotate_half

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch's LayerNorm
        doesn't support bias=False
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a single batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make fast and easy
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cos_sin, episodic_kv=None, state=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        attn = self.resid_dropout(self.c_proj(attn))
        return attn

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config) if config.attention_type == "mha" else KimiLinearAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, episodic_kv=None, state=None):
        x = x + self.attn(self.ln_1(x), cos_sin, episodic_kv, state)
        x = x + self.mlp(self.ln_2(x))
        return x, state

@dataclass
class GPTConfig:
    block_size: int = 2048000 # was 1024
    sequence_len: int = 2048000 # Added sequence_len
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    num_layers: int = 12 # Added num_layers for KVCache
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rotary_seq_len: int = 16
    # LongRoPE additions:
    rotary_base: float = 10000.0  # same as usual RoPE base
    rotary_rescale_factors: Optional[torch.Tensor] = None  # shape (rotary_dim,)
    rotary_rescale_headwise: bool = False  # if using per-head vs per-dim
    rotary_short_threshold: int = 8192  # length under which we use short-readjust factors
    rotary_critical_dim: Optional[int] = None
    rotary_interpolation_strategy: Optional[str] = None
    longrope_progressive_stages: Optional[List[int]] = None
    rotary_readjust_short_k: Optional[bool] = None
    rotary_readjust_lengths: Optional[List[int]] = None
    rotary_readjust_map: Optional[List[int]] = None # Added this based on the previous error
    # progressive strategy
    longrope_stages: List[int] = field(default_factory=lambda: [256_000, 2_048_000])  # staged target lengths
    attention_type: str = "mha" # "mha" for MultiHeadSelfAttention, "kimi" for KimiLinearAttention
    state_size: int = 128 # For KimiLinearAttention
    rank: int = 16 # For KimiLinearAttention
    chunk_size: int = 16 # For KimiLinearAttention
    
    # For KimiLinearAttention
    decay_rate: float = 0.999
    clamp_val: float = 1.0
    
    # For Abacus
    num_concept_ids: int = 1000
    hypercube_dim: int = 128
    abacus_input_dim: int = 256
    
    # For RoPE
    rope_theta: int = 10000
    
    # For Muon
    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    
    # For training
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_iters: int = 0
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    
    # For checkpointing
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    
    # For distributed training
    backend: str = 'nccl'
    
    # For system
    device: str = 'cpu'
    dtype: str = 'bfloat16'
    compile: bool = False
    
    # For data
    dataset: str = 'openwebtext'
    
    # For inference
    init_from: str = 'scratch'
    
    # For chat
    chat: bool = False
    
    # For concept
    concept_memory_size: int = 1000
    concept_memory_top_k: int = 5
    use_concept_attention: bool = False
    
    # For psyche
    psyche_id_lr_scale: float = 1.0
    psyche_ego_lr_scale: float = 1.0
    psyche_superego_lr_scale: float = 1.0
    id_loss_weight: float = 0.2 # New hyperparameter for deep supervision
    ego_loss_weight: float = 0.2 # New hyperparameter for deep supervision
    superego_loss_weight: float = 0.2 # New hyperparameter for deep supervision




class GPT(nn.Module):
    def __init__(self, config: GPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.dtype = dtype

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            id_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ego_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            superego_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.concept_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.psyche_controller = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() and DDP, this avoids expensive fp32 copy
        self.transformer.wte.weight = self.lm_head.weight 

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model. If non_embedding=True, then only count parameters in the transformer blocks."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def to_empty(self, device: torch.device):
        return super().to_empty(device=device)

        # token embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd)
        ))

        # psyche layers
        self.id_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ego_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.superego_layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # psyche heads
        self.id_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ego_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.superego_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # concept head
        self.concept_head = nn.Linear(config.n_embd, config.num_concept_ids, bias=False)

        # psyche controller
        self.psyche_controller = nn.Linear(config.n_embd, 3, bias=False) # 3 for id, ego, superego

        # kv cache
        self.kv_cache = KVCache(config.block_size, config.n_head, config.n_embd // config.n_head, config.batch_size, config.num_layers)

        # rotary embeddings
        self.init_rotary_cache(self.config.rotary_seq_len, self.get_device())

    def init_rotary_cache(self, max_seq_len: int, device: str):
        head_dim = self.config.n_embd // self.config.n_head

        # Precompute LongRoPE scaled embeddings
        cos_longrope, sin_longrope = self.build_rotary_cos_sin(
            max_seq_len,
            head_dim,
            device,
            base=self.config.rotary_base,
            rescale_factors=self.config.rotary_rescale_factors,
            longrope_progressive_stages=self.config.longrope_progressive_stages,
            rotary_interpolation_strategy=self.config.rotary_interpolation_strategy
        )
        self.register_buffer("cos_longrope", cos_longrope, persistent=False)
        self.register_buffer("sin_longrope", sin_longrope, persistent=False)

        # Precompute original RoPE embeddings and readjusted ones if configured
        if self.config.rotary_readjust_short_k and self.config.rotary_readjust_map is not None and self.config.rotary_readjust_lengths is not None:
            self.cos_readjusted = {}
            self.sin_readjusted = {}
            for length in self.config.rotary_readjust_lengths:
                if length in self.config.rotary_readjust_map:
                    readjust_factors = torch.tensor(self.config.rotary_readjust_map[length], dtype=torch.float32)
                    cos_adj, sin_adj = self.build_rotary_cos_sin(
                        length,
                        head_dim,
                        device,
                        base=self.config.rotary_base,
                        rescale_factors=readjust_factors
                    )
                    self.cos_readjusted[length] = cos_adj
                    self.sin_readjusted[length] = sin_adj
                else:
                    # Fallback to original if no specific readjustment factors are found
                    cos_original, sin_original = self.build_rotary_cos_sin(
                        max_seq_len,
                        head_dim,
                        device,
                        base=self.config.rotary_base
                    )
                    self.register_buffer("cos_original", cos_original, persistent=False)
                    self.register_buffer("sin_original", sin_original, persistent=False)
        else:
            cos_original, sin_original = self.build_rotary_cos_sin(
                max_seq_len,
                head_dim,
                device,
                base=self.config.rotary_base
            )
            self.register_buffer("cos_original", cos_original, persistent=False)
            self.register_buffer("sin_original", sin_original, persistent=False)

    def get_device(self):
        # Get device from concept_head weight
        return self.concept_head.weight.device

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def build_rotary_cos_sin(self, max_seq_len: int, head_dim: int, device: str, base: float, rescale_factors: torch.Tensor | None = None, longrope_progressive_stages: list[int] | None = None, rotary_interpolation_strategy: str | None = None):
        # Original RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))

        # Apply rescale factors if provided
        if rescale_factors is not None:
            if rescale_factors.shape[0] != head_dim // 2:
                raise ValueError(f"rescale_factors shape mismatch. Expected ({head_dim // 2},), got {rescale_factors.shape}")
            inv_freq = inv_freq * rescale_factors.to(device)

        # Apply LongRoPE modifications if configured
        if longrope_progressive_stages is not None and rotary_interpolation_strategy == "non_uniform":
            rescale_map = torch.ones_like(inv_freq)
            current_seq_len = 0
            for i, stage_len in enumerate(longrope_progressive_stages):
                if current_seq_len < max_seq_len:
                    factor = rescale_factors[i] if rescale_factors is not None and i < len(rescale_factors) else 1.0
                    rescale_map[current_seq_len:min(current_seq_len + stage_len, max_seq_len)] *= factor
                current_seq_len += stage_len
            inv_freq = inv_freq * rescale_map

        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, max_seq_len, 1, head_dim)
        sin = emb.sin().view(1, max_seq_len, 1, head_dim)
        return cos, sin

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out concept_head weights
        torch.nn.init.zeros_(self.concept_head.weight)
        # zero out c_proj weights in all blocks
        for block_list in [self.transformer.id_layers, self.transformer.ego_layers, self.transformer.superego_layers]:
            for block in block_list:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
        # Initialize rotary cache
        self.init_rotary_cache(self.config.rotary_seq_len, self.get_device())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)



    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = 0 # No separate embedding layer now
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, concept_head)
        matrix_params = []
        for block_list in [self.transformer.id_layers, self.transformer.ego_layers, self.transformer.superego_layers]:
            for block in block_list:
                matrix_params.extend(list(block.parameters()))

        id_params = list(self.transformer.id_layers.parameters())
        ego_params = list(self.transformer.ego_layers.parameters())
        superego_params = list(self.transformer.superego_layers.parameters())

        embedding_params = [] # No separate embedding layer now
        concept_head_params = list(self.concept_head.parameters()) # New concept head params
        psyche_controller_params = list(self.psyche_controller.parameters()) # Psyche controller params

        # assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(concept_head_params)
        # Create the AdamW optimizer for the embedding and concept_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=concept_head_params, lr=unembedding_lr * dmodel_lr_scale), # Use unembedding_lr for concept_head
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=id_params, lr=3e-4 * dmodel_lr_scale), # Id layers learning rate
            dict(params=ego_params, lr=1e-4 * dmodel_lr_scale), # Ego layers learning rate
            dict(params=superego_params, lr=5e-5 * dmodel_lr_scale), # Superego layers learning rate
            dict(params=psyche_controller_params, lr=1e-4 * dmodel_lr_scale), # Psyche controller learning rate
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def _run_layers(self, layers, x, cos_sin, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, state: torch.Tensor | None = None):
        for block in layers:
            x, state = block(x, cos_sin, self.kv_cache, episodic_kv, state)
        return x, state

    def forward(self, idx, targets=None, input_embeddings=None, kv_cache=None, abacus_embedding=None, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, long_term_memory_embeddings: torch.Tensor | None = None, psyche_weights: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # If input_embeddings are provided, use them directly
        if input_embeddings is not None:
            x = input_embeddings
        else:
            # token embeddings of shape (B, T, C)
            x = self.transformer.wte(idx)

        # Determine which rotary embeddings to use
        if self.config.rotary_readjust_short_k and self.config.rotary_readjust_map is not None and self.config.rotary_readjust_lengths is not None:
            cos_sin_selected = None
            for length in self.config.rotary_readjust_lengths:
                if T == length and length in self.cos_readjusted:
                    cos_sin_selected = (self.cos_readjusted[length], self.sin_readjusted[length])
                    break
            if cos_sin_selected is None:
                if T < self.config.rotary_short_threshold:
                    cos_sin_selected = (self.cos_original, self.sin_original)
                else:
                    cos_sin_selected = (self.cos_longrope, self.sin_longrope)
        else:
            if T < self.config.rotary_short_threshold:
                cos_sin_selected = (self.cos_original, self.sin_original)
            else:
                cos_sin_selected = (self.cos_longrope, self.sin_longrope)

        # Slice the selected rotary embeddings to the current sequence length
        cos, sin = cos_sin_selected
        cos_sin = (cos[:, :T, :, :], sin[:, :T, :, :])

        # If kv_cache is provided, use it
        if kv_cache is not None:
            self.kv_cache = kv_cache

        state = None # Initialize state for KimiLinearAttention

        # Generate psyche weights if not provided
        if psyche_weights is None:
            psyche_weights = self.psyche_controller(x)

        # Process Id layers
        x_id, state = self._run_layers(self.id_layers, x, cos_sin, episodic_kv, state)

        # Process Ego layers
        x_ego = x_id
        if abacus_embedding is not None:
            # Broadcast abacus_embedding to match the sequence length of x_ego
            # Assuming abacus_embedding is (B, C) and x_ego is (B, T, C)
            abacus_broadcast = abacus_embedding.unsqueeze(1).expand(-1, x_ego.size(1), -1)
            x_ego = x_ego + abacus_broadcast # Inject abacus_embedding into ego layer
        if long_term_memory_embeddings is not None:
            # Broadcast long_term_memory_embeddings to match the sequence length of x_ego
            # Assuming long_term_memory_embeddings is (B, C) and x_ego is (B, T, C)
            long_term_memory_broadcast = long_term_memory_embeddings.unsqueeze(1).expand(-1, x_ego.size(1), -1)
            x_ego = x_ego + long_term_memory_broadcast # Inject long_term_memory_embeddings into ego layer
        x_ego, state = self._run_layers(self.ego_layers, x_ego, cos_sin, episodic_kv, state)

        # Process Superego layers
        x_superego = x_ego
        if long_term_memory_embeddings is not None:
            # Broadcast long_term_memory_embeddings to match the sequence length of x_superego
            # Assuming long_term_memory_embeddings is (B, C) and x_superego is (B, T, C)
            long_term_memory_broadcast = long_term_memory_embeddings.unsqueeze(1).expand(-1, x_superego.size(1), -1)
            x_superego = x_superego + long_term_memory_embeddings.unsqueeze(1).expand(-1, x_superego.size(1), -1)
        x_superego, state = self._run_layers(self.superego_layers, x_superego, cos_sin, episodic_kv, state)

        # Apply auxiliary heads to the outputs of each psyche layer
        id_logits = self.id_head(x_id)
        ego_logits = self.ego_head(x_ego)
        superego_logits = self.superego_head(x_superego)

        # Blend the outputs of the psyche layers using psyche_weights
        # For now, we'll use the final output of the last layer for concept_head
        # In the future, this will be a weighted sum based on psyche_weights
        x = x_id * psyche_weights[:, 0].unsqueeze(1).unsqueeze(2) + \
            x_ego * psyche_weights[:, 1].unsqueeze(1).unsqueeze(2) + \
            x_superego * psyche_weights[:, 2].unsqueeze(1).unsqueeze(2)

        # Final concept head for the blended output
        x = norm(x)
        return self.concept_head(x), self.kv_cache, id_logits, ego_logits, superego_logits

    def forward_step(self, next_embedding: torch.Tensor, kv_cache_param=None, abacus_embedding: torch.Tensor | None = None, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, long_term_memory_embeddings: torch.Tensor | None = None, psyche_weights: torch.Tensor | None = None):
        B, C = next_embedding.size()
        T = self.kv_cache.current_seq_len + 1 # Current sequence length after adding next_embedding

        if kv_cache_param is not None:
            self.kv_cache = kv_cache_param

        # Determine which rotary embeddings to use for forward_step
        if self.config.rotary_readjust_short_k and self.config.rotary_readjust_map is not None and self.config.rotary_readjust_lengths is not None:
            cos_sin_selected = None
            for length in self.config.rotary_readjust_lengths:
                if T == length and length in self.cos_readjusted:
                    cos_sin_selected = (self.cos_readjusted[length], self.sin_readjusted[length])
                    break
            if cos_sin_selected is None:
                if T < self.config.rotary_short_threshold:
                    cos_sin_selected = (self.cos_original, self.sin_original)
                else:
                    cos_sin_selected = (self.cos_longrope, self.sin_longrope)
        else:
            if T < self.config.rotary_short_threshold:
                cos_sin_selected = (self.cos_original, self.sin_original)
            else:
                cos_sin_selected = (self.cos_longrope, self.sin_longrope)

        # Slice the selected rotary embeddings to the current sequence length
        cos, sin = cos_sin_selected
        cos_sin = (cos[:, T-1:T, :, :], sin[:, T-1:T, :, :])

        x = next_embedding.unsqueeze(1) # Add sequence dimension for consistency
        state = None # Initialize state for KimiLinearAttention

        # Generate psyche weights if not provided
        if psyche_weights is None:
            psyche_weights = self.psyche_controller(x)

        # Process Id layers
        x_id, state = self._run_layers(self.id_layers, x, cos_sin, episodic_kv, state)

        # Process Ego layers
        x_ego = x_id
        if abacus_embedding is not None:
            # Broadcast abacus_embedding to match the sequence length of x_ego
            # Assuming abacus_embedding is (B, C) and x_ego is (B, T, C)
            abacus_broadcast = abacus_embedding.unsqueeze(1).expand(-1, x_ego.size(1), -1)
            x_ego = x_ego + abacus_broadcast # Inject abacus_embedding into ego layer
        if long_term_memory_embeddings is not None:
            # Broadcast long_term_memory_embeddings to match the sequence length of x_ego
            # Assuming long_term_memory_embeddings is (B, C) and x_ego is (B, T, C)
            long_term_memory_broadcast = long_term_memory_embeddings.unsqueeze(1).expand(-1, x_ego.size(1), -1)
            x_ego = x_ego + long_term_memory_broadcast # Inject long_term_memory_embeddings into ego layer
        x_ego, state = self._run_layers(self.ego_layers, x_ego, cos_sin, episodic_kv, state)

        # Process Superego layers
        x_superego = x_ego
        if long_term_memory_embeddings is not None:
            # Broadcast long_term_memory_embeddings to match the sequence length of x_superego
            # Assuming long_term_memory_embeddings is (B, C) and x_superego is (B, T, C)
            long_term_memory_broadcast = long_term_memory_embeddings.unsqueeze(1).expand(-1, x_superego.size(1), -1)
            x_superego = x_superego + long_term_memory_embeddings.unsqueeze(1).expand(-1, x_superego.size(1), -1)
        x_superego, state = self._run_layers(self.superego_layers, x_superego, cos_sin, episodic_kv, state)

        # Apply auxiliary heads to the outputs of each psyche layer
        id_logits = self.id_head(x_id)
        ego_logits = self.ego_head(x_ego)
        superego_logits = self.superego_head(x_superego)

        # Blend the outputs of the psyche layers using psyche_weights
        # For now, we'll use the final output of the last layer for concept_head
        # In the future, this will be a weighted sum based on psyche_weights
        x = x_id * psyche_weights[:, 0].unsqueeze(1).unsqueeze(2) + \
            x_ego * psyche_weights[:, 1].unsqueeze(1).unsqueeze(2) + \
            x_superego * psyche_weights[:, 2].unsqueeze(1).unsqueeze(2)

        # Final concept head for the blended output
        x = norm(x)
        return self.concept_head(x.squeeze(1)), self.kv_cache, id_logits, ego_logits, superego_logits

        # Dynamically blend the outputs based on psyche_weights
        if psyche_weights is None:
            psyche_weights = self.psyche_controller(x)
        # Reshape psyche_weights for broadcasting: (B, 1, 3)
        psyche_weights_reshaped = psyche_weights.unsqueeze(1)

        # Stack the outputs and apply weighted sum
        # Stack will result in (B, T, 3, C)
        stacked_outputs = torch.stack([x_id, x_ego, x_superego], dim=2)
        # Weighted sum: (B, T, 1, C) after sum, then squeeze to (B, T, C)
        x = (stacked_outputs * psyche_weights_reshaped.unsqueeze(-1)).sum(dim=2)

        # Final concept head
        return self.concept_head(x), self.kv_cache, x_id, x_ego, x_superego
