import torch

class KimiState:
    def __init__(self, batch_size, state_size, device):
        self.s_t = torch.zeros(batch_size, state_size, device=device) # Finite-state memory
        self.decay_accumulators = None # Placeholder for chunk cumulative decays
        self.last_chunk_index = -1 # To track the last processed chunk

class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, attention_type="full", state_size=None, device=None):
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        if self.attention_type == "full":
            self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
            self.kv_cache = None
            self.pos = 0
        elif self.attention_type == "kimi":
            assert state_size is not None, "state_size must be provided for kimi attention_type"
            assert device is not None, "device must be provided for kimi attention_type"
            self.kimi_cache = [KimiState(batch_size, state_size, device) for _ in range(num_layers)]
            self.pos = 0 # Still track position for sequence length
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def reset(self):
        self.pos = 0
        if self.attention_type == "kimi":
            for state in self.kimi_cache:
                state.s_t.zero_()
                state.decay_accumulators = None
                state.last_chunk_index = -1

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                assert dim1 == dim2
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1
            elif ix == 4:
                assert dim1 >= dim2
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024
            t_needed = (t_needed + 1023) & ~1023
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view

    def retrieve_episode(self, start_pos, end_pos):
        assert self.kv_cache is not None, "KV cache is empty, cannot retrieve episode."
        assert start_pos >= 0 and end_pos <= self.pos, "Invalid start or end position for episode retrieval."
        assert start_pos < end_pos, "Start position must be less than end position."
        episode_k = self.kv_cache[:, 0, :, :, start_pos:end_pos, :]
        episode_v = self.kv_cache[:, 1, :, :, start_pos:end_pos, :]
        return episode_k, episode_v

    def store(self, layer_idx, k=None, v=None, kimi_state=None, chunk_meta=None):
        if self.attention_type == "full":
            if self.kv_cache is None:
                self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
            B, H, T_add, D = k.size()
            t0, t1 = self.pos, self.pos + T_add
            if t1 > self.kv_cache.size(4):
                t_needed = t1 + 1024
                t_needed = (t_needed + 1023) & ~1023
                additional_shape = list(self.kv_cache.shape)
                additional_shape[4] = t_needed - self.kv_cache.size(4)
                additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
                self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
                self.kv_shape = self.kv_cache.shape
            self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
            self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
            key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
            value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
            if layer_idx == self.num_layers - 1:
                self.pos = t1
            return key_view, value_view
        elif self.attention_type == "kimi":
            assert kimi_state is not None, "kimi_state must be provided for kimi attention_type"
            self.kimi_cache[layer_idx].s_t = kimi_state.s_t
            self.kimi_cache[layer_idx].decay_accumulators = kimi_state.decay_accumulators
            self.kimi_cache[layer_idx].last_chunk_index = kimi_state.last_chunk_index
            if layer_idx == self.num_layers - 1:
                self.pos += 1 # Increment position for each token processed
            return kimi_state

    def fetch(self, layer_idx, mode="full"):
        if mode == "full":
            assert self.kv_cache is not None, "KV cache is empty, cannot fetch."
            t1 = self.pos
            key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
            value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
            return key_view, value_view
        elif mode == "kimi":
            return self.kimi_cache[layer_idx]
        else:
            raise ValueError(f"Unknown cache mode: {mode}")