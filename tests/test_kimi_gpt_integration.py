import torch
import pytest
from nanochat.gpt import GPT, GPTConfig

def test_kimi_gpt_integration():
    # Define a minimal GPTConfig for testing
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
        attention_type="kimi",
        state_size=32 # KimiLinearAttention requires state_size
    )

    # Initialize the GPT model with the KimiLinearAttention config
    model = GPT(config)
    model.eval() # Set to eval mode for consistent behavior

    # Create dummy input
    batch_size = 1
    seq_len = 8
    # Create a dummy 3D input tensor representing embeddings
    x = torch.randn(batch_size, seq_len, config.n_embd)

    # Run a forward pass
    with torch.no_grad():
        y = model(x)

    # Assert the output shape
    assert y[0].shape == (batch_size, seq_len, config.num_concept_ids)

    print("KimiLinearAttention GPT integration test passed!")