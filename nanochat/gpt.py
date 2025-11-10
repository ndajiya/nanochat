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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.attention.kimi import KimiLinearAttention
from nanochat.kv_cache import KVCache

@dataclass
class GPTConfig:
    block_size: int = 16000 # was 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rotary_seq_len: int = 16384
    attention_type: str = "mha" # "mha" for MultiHeadSelfAttention, "kimi" for KimiLinearAttention
    state_size: int = 128 # For KimiLinearAttention
    rank: int = 16 # For KimiLinearAttention
    chunk_size: int = 16 # For KimiLinearAttention

    def __init__(self,
                 block_size=1024,
                 vocab_size=50304,
                 n_layer=12,
                 n_head=12,
                 n_embd=768,
                 dropout=0.0,
                 bias=True,
                 attention_type="mha",
                 state_size=128,
                 rank=16,
                 chunk_size=16,
                 
                 # For KimiLinearAttention
                 decay_rate=0.999,
                 clamp_val=1.0,
                 
                 # For Abacus
                 num_concept_ids=1000,
                 hypercube_dim=128,
                 abacus_input_dim=256,
                 
                 # For RoPE
                 rope_theta=10000,
                 
                 # For Muon
                 n_kv_head=None,
                 multiple_of=256,
                 norm_eps=1e-5,
                 
                 # For training
                 batch_size=1,
                 gradient_accumulation_steps=1,
                 max_iters=0,
                 lr=6e-4,
                 min_lr=6e-5,
                 weight_decay=1e-1,
                 beta1=0.9,
                 beta2=0.95,
                 grad_clip=1.0,
                 decay_lr=True,
                 warmup_iters=2000,
                 lr_decay_iters=600000,
                 
                 # For checkpointing
                 out_dir='out',
                 eval_interval=2000,
                 log_interval=1,
                 eval_iters=200,
                 eval_only=False,
                 always_save_checkpoint=True,
                 
                 # For distributed training
                 backend='nccl',
                 
                 # For system
                 device='cpu',
                 dtype='bfloat16',
                 compile=False,
                 
                 # For data
                 dataset='openwebtext',
                 
                 # For inference
                 init_from='scratch',
                 
                 # For chat
                 chat=False,
                 
                 # For concept
                 concept_memory_size=1000,
                 concept_memory_top_k=5,
                 use_concept_attention=False,
                 
                 # For psyche
                 psyche_id_lr_scale=1.0,
                 psyche_ego_lr_scale=1.0,
                 psyche_superego_lr_scale=1.0,
                 id_loss_weight=0.2, # New hyperparameter for deep supervision
                 ego_loss_weight=0.2, # New hyperparameter for deep supervision
                 superego_loss_weight=0.2, # New hyperparameter for deep supervision
                 **kwargs):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.attention_type = attention_type
        self.state_size = state_size
        self.rank = rank
        self.chunk_size = chunk_size
        self.decay_rate = decay_rate
        self.clamp_val = clamp_val
        self.num_concept_ids = num_concept_ids
        self.hypercube_dim = hypercube_dim
        self.abacus_input_dim = abacus_input_dim
        self.rope_theta = rope_theta
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_iters = max_iters
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.out_dir = out_dir
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.eval_iters = eval_iters
        self.eval_only = eval_only
        self.always_save_checkpoint = always_save_checkpoint
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.dataset = dataset
        self.init_from = init_from
        self.chat = chat
        self.concept_memory_size = concept_memory_size
        self.concept_memory_top_k = concept_memory_top_k
        self.use_concept_attention = use_concept_attention
        self.psyche_id_lr_scale = psyche_id_lr_scale
        self.psyche_ego_lr_scale = psyche_ego_lr_scale
        self.psyche_superego_lr_scale = psyche_superego_lr_scale
        self.id_loss_weight = id_loss_weight
        self.ego_loss_weight = ego_loss_weight
        self.superego_loss_weight = superego_loss_weight
        self.__dict__.update(kwargs) # Capture any additional arguments

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        if config.attention_type == "full":
            self.attn = MultiHeadSelfAttention(config, layer_idx)
        elif config.attention_type == "kimi":
            self.attn = KimiLinearAttention(config)
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")

    def forward(self, x, cos_sin, kv_cache, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        if isinstance(self.attn, KimiLinearAttention):
            output, state = self.attn(x, cos_sin, kv_cache, episodic_kv)
            # For KimiLinearAttention, the 'state' is the new 'kv_cache' for the next layer/timestep
            # We need to pass this state up, so we'll return it as the updated episodic_kv
            return output, (state,)
        else:
            return self.attn(x, cos_sin, kv_cache, episodic_kv), None # MultiHeadSelfAttention doesn't return a state

        
        
        

        
            

        
        
            
            
            
            

        
        

        
        
        
            
            
                
            
            
            
            
            

        
        
        



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = KimiLinearAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, state: torch.Tensor | None = None):
        attn_output, state = self.attn(norm(x), cos_sin, kv_cache, episodic_kv, state, self.layer_idx)
        x = x + attn_output
        x = x + self.mlp(norm(x))
        return x, state


class PsycheController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.controller_head = nn.Linear(config.n_embd, 3) # 3 for id, ego, superego

    def forward(self, x):
        # For now, just return equal weights for each psyche layer
        # In the future, this will be trained to dynamically blend psyche outputs
        # Average x over the sequence dimension to get (B, C)
        x_averaged = x.mean(dim=1)
        return torch.softmax(self.controller_head(x_averaged), dim=-1)


class GPT(nn.Module):
    def __init__(self, config, kv_cache=None):
        super().__init__()
        self.config = config
        self.kv_cache = kv_cache

        # Assign attention types layer-wise
        h_blocks = []
        for layer_idx in range(config.n_layer):
            layer_config = copy.deepcopy(self.config) # Start with a copy of the main config
            layer_config.attention_type = "kimi"  # Use KimiLinearAttention
            h_blocks.append(Block(layer_config, layer_idx))

        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList(h_blocks),
        })

        # Partition transformer layers into psyche layers
        total_layers = config.n_layer
        id_end = total_layers // 3
        ego_end = 2 * total_layers // 3

        self.id_layers = self.transformer.h[:id_end]
        self.ego_layers = self.transformer.h[id_end:ego_end]
        self.superego_layers = self.transformer.h[ego_end:]

        self.psyche_registry = {
            "id": self.id_layers,
            "ego": self.ego_layers,
            "superego": self.superego_layers
        }

        self.concept_head = nn.Linear(config.n_embd, config.num_concept_ids, bias=False) # New concept head
        self.psyche_controller = PsycheController(config) # Initialize PsycheController
        self.id_head = nn.Linear(config.n_embd, config.num_concept_ids, bias=False)
        self.ego_head = nn.Linear(config.n_embd, config.num_concept_ids, bias=False)
        self.superego_head = nn.Linear(config.n_embd, config.num_concept_ids, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = 16384 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out concept_head weights
        torch.nn.init.zeros_(self.concept_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

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

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.concept_head.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :].repeat(1, 1, 1, 2), sin[None, :, None, :].repeat(1, 1, 1, 2) # add batch and head dims for later broadcasting and expand to full head_dim
        return cos, sin

    def get_device(self):
        # Get device from concept_head weight
        return self.concept_head.weight.device

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
        # matrix_params = list(self.transformer.h.parameters())
        id_params = list(self.id_layers.parameters())
        ego_params = list(self.ego_layers.parameters())
        superego_params = list(self.superego_layers.parameters())

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
        # muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        # MuonFactory = DistMuon if ddp else Muon
        # muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def _run_layers(self, layers, x, cos_sin, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, state: torch.Tensor | None = None):
        for block in layers:
            x, state = block(x, cos_sin, self.kv_cache, episodic_kv, state)
        return x, state

    def forward(self, input_embeddings: torch.Tensor, kv_cache_param=None, abacus_embedding: torch.Tensor | None = None, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, long_term_memory_embeddings: torch.Tensor | None = None, psyche_weights: torch.Tensor | None = None):
        B, T, C = input_embeddings.size()

        if kv_cache_param is not None:
            self.kv_cache = kv_cache_param
        elif self.kv_cache is None:
            # Initialize KVCache if not provided
            num_heads = self.config.n_head
            head_dim = self.config.n_embd // num_heads
            num_layers = self.config.n_layer
            state_size = self.config.state_size
            device = input_embeddings.device
            self.kv_cache = KVCache(B, num_heads, T, head_dim, num_layers, self.config.attention_type, state_size, device)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T, :, :], self.sin[:, :T, :, :]

        x = input_embeddings
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

    def forward_prefill(self, input_embeddings: torch.Tensor, kv_cache_param=None, abacus_embedding: torch.Tensor | None = None, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, long_term_memory_embeddings: torch.Tensor | None = None, psyche_weights: torch.Tensor | None = None):
        B, T, C = input_embeddings.size()

        if kv_cache_param is not None:
            self.kv_cache = kv_cache_param
        elif self.kv_cache is None:
            # Initialize KVCache if not provided
            num_heads = self.config.n_head
            head_dim = self.config.n_embd // num_heads
            num_layers = self.config.n_layer
            state_size = self.config.state_size
            device = input_embeddings.device
            self.kv_cache = KVCache(B, num_heads, T, head_dim, num_layers, self.config.attention_type, state_size, device)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T, :, :], self.sin[:, :T, :, :]

        x = input_embeddings
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

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, T-1:T, :, :], self.sin[:, T-1:T, :, :]

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
