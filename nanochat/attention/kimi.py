import torch
import torch.nn as nn
import torch.nn.functional as F

class KimiLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.state_size = config.state_size
        self.rank = config.rank
        self.chunk_size = config.chunk_size # Get chunk_size from config
        self.n_head = config.n_head # Get n_head from config

        # DPLR parameters
        self.U = nn.Parameter(torch.randn(self.state_size, self.rank))
        self.V = nn.Parameter(torch.randn(self.d_model, self.rank))
        self.diag = nn.Parameter(torch.zeros(self.state_size)) # logit for decay -> clamp

        # Gating parameters
        self.gate_proj = nn.Linear(self.d_model, self.state_size, bias=False)

        # Projection layers
        self.to_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.to_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.to_out = nn.Linear(self.state_size, self.d_model, bias=False) # Projection for output

    def _forward_chunk(self, state, chunk_x, chunk_k, chunk_v):
        # state: (B, state_size)
        # chunk_x: (B, chunk_len, d_model)
        # chunk_k: (B, chunk_len, d_model)
        # chunk_v: (B, chunk_len, d_model)
        B, chunk_len, _ = chunk_k.shape

        # Compute delta for the entire chunk
        # V: (d_model, rank), chunk_k: (B, chunk_len, d_model)
        # V_T_k_chunk: (B, chunk_len, rank)
        V_T_k_chunk = torch.einsum('dr,btd->btr', self.V, chunk_k)
        # U: (state_size, rank), V_T_k_chunk: (B, chunk_len, rank)
        # delta_chunk: (B, chunk_len, state_size)
        delta_chunk = torch.einsum('sr,btr->bts', self.U, V_T_k_chunk)

        # Apply gating for the entire chunk
        g_chunk = torch.sigmoid(self.gate_proj(chunk_x))

        # Compute h_chunk = g_chunk * delta_chunk
        h_chunk = g_chunk * delta_chunk # (B, chunk_len, state_size)

        # Prepare decay for vectorized state update
        decay = torch.clamp(F.softplus(self.diag), 0.0, 1.0) # (state_size)
        # Create powers of decay for each time step in the chunk
        decay_powers = decay.unsqueeze(0).unsqueeze(0) ** torch.arange(1, chunk_len + 1, device=chunk_k.device).unsqueeze(-1) # (1, chunk_len, state_size)

        # Compute h_prime_chunk = h_chunk / decay_powers
        h_prime_chunk = h_chunk / decay_powers

        # Compute y_chunk_history = state.unsqueeze(1) + torch.cumsum(h_prime_chunk, dim=1)
        # state.unsqueeze(1): (B, 1, state_size)
        y_chunk_history = state.unsqueeze(1) + torch.cumsum(h_prime_chunk, dim=1) # (B, chunk_len, state_size)

        # Compute s_chunk_history = y_chunk_history * decay_powers
        s_chunk_history = y_chunk_history * decay_powers # (B, chunk_len, state_size)

        # The final state after this chunk
        state_next = s_chunk_history[:, -1, :] # (B, state_size)

        # Produce output for the entire chunk
        # s_chunk_history: (B, chunk_len, state_size), chunk_v: (B, chunk_len, state_size)
        # output_chunk: (B, chunk_len, D)
        output_chunk = self.to_out(s_chunk_history) + chunk_v

        return state_next, output_chunk

    def forward(self, x, cos_sin, kv_cache, episodic_kv: tuple[torch.Tensor, torch.Tensor] | None = None, state: torch.Tensor | None = None, layer_idx: int | None = None):
        # x: (B, T, D)
        B, T, D = x.shape

        k = self.to_k(x) # (B, T, d_model)
        v = self.to_v(x) # (B, T, d_model)

        # Reshape k and v for rotary embeddings
        k = k.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head)

        # Apply Rotary Embeddings
        k = k * cos_sin[0] + k * cos_sin[1]
        v = v * cos_sin[0] + v * cos_sin[1]

        # Reshape k and v back to (B, T, d_model)
        k = k.view(B, T, D)
        v = v.view(B, T, D)

        outputs = []

        # If kv_cache is provided and attention_type is 'kimi', fetch the initial state
        if kv_cache is not None and kv_cache.attention_type == "kimi":
            kimi_state = kv_cache.fetch(layer_idx, mode="kimi")
            state = kimi_state.s_t
        elif episodic_kv is not None:
            state = episodic_kv[0] # Assuming episodic_kv[0] contains the initial state
        else:
            state = torch.zeros(B, self.state_size, device=x.device)

        for i in range(0, T, self.chunk_size):
            chunk_x = x[:, i:i+self.chunk_size, :]
            chunk_k = k[:, i:i+self.chunk_size, :]
            chunk_v = v[:, i:i+self.chunk_size, :]

            state, output_chunk = self._forward_chunk(state, chunk_x, chunk_k, chunk_v)
            outputs.append(output_chunk)

        output = torch.cat(outputs, dim=1)

        # If kv_cache is provided and attention_type is 'kimi', store the final state
        if kv_cache is not None and kv_cache.attention_type == "kimi":
            kimi_state.s_t = state
            kv_cache.store(layer_idx, kimi_state=kimi_state) # head_idx is 0 for Kimi

        return self.to_out(output), state