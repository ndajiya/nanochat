# NanoChat Architecture Documentation

## Overview

NanoChat is a sophisticated transformer-based language model that incorporates advanced architectural features including:
- Multi-head self-attention with rotary position embeddings (RoPE)
- PsycheController for dynamic blending of different processing modes
- Consciousness integration and self-modeling capabilities
- Advanced training objectives including energy-based learning
- Rust-accelerated tokenization (rustbpe)

## Extended LLM Cognitive Framework

This section outlines the new cognitive layers and features added to the nanochat LLM, along with their architectural connections.

### Key Added Features
1. **PsycheController** (`nanochat/engine.py`): Dynamically generates blending weights for the Id/Ego/Superego layers based on input context.
2. **GPT Id/Ego/Superego Layers** (`nanochat/gpt.py`): Three distinct cognitive processing layers with specialized focuses (instinctual, practical, moral reasoning).
3. **HypercubeEmbeddingLayer** (`nanochat/hypercube.py`): Maps concept IDs to embeddings within a hypercube structure, enabling semantic relationships.
4. **LongTermMemory** (`nanochat/engine.py`): Stores and retrieves embeddings to preserve long-term context across generations.
5. **AbacusEncoder & AbacusStateMemory** (`nanochat/abacus_encoder.py`/`nanochat/abacus_state_memory.py`): Ensures logical consistency of internal reasoning through structured pattern storage.
6. **MemeticLearningLayer** (`nanochat/memetic_learning.py`): Evaluates the fitness of generated embeddings for knowledge retention.
7. **ConsciousIntegrationLayer** (`nanochat/conscious_integration.py`): Synthesizes outputs from all lower layers into a unified conceptual state that directly influences model output.

### Architecture Flow
1. Input embeddings are processed through GPT's Id/Ego/Superego layers.
2. PsycheController provides dynamic blending weights.
3. LongTermMemory retrieves relevant context embeddings.
4. AbacusEncoder ensures logical consistency of intermediate states.
5. MemeticLearningLayer evaluates generated embedding fitness.
6. ConsciousIntegrationLayer synthesizes all inputs into a unified state.
7. Synthesized state modulates concept_logits for final output.

## Core Architecture

### 1. GPT Backbone (`nanochat/gpt.py`)

The main model class `GPT` implements a transformer architecture with the following key components:

#### Transformer Blocks
- **Multi-Head Attention**: Standard self-attention with configurable number of heads
- **Rotary Position Embeddings (RoPE)**: Enhanced positional encoding supporting up to 16,384 sequence length
- **SwiGLU Feed-Forward**: Advanced activation function for improved expressivity
- **RMSNorm**: Root Mean Square normalization for stable training
- **Residual Connections**: Skip connections around attention and feed-forward layers

#### PsycheController
A unique component that dynamically blends three processing modes:
- **Id**: Raw, instinctive processing
- **Ego**: Balanced, rational processing  
- **Superego**: Constrained, rule-based processing

The controller learns to weight these modes based on input context, producing a blended representation.

#### Specialized Heads
- **Language Modeling Head**: Standard next-token prediction
- **Concept Head**: Maps to conceptual vocabulary (50,257 concepts)
- **Energy Head**: Predicts energy values for consciousness modeling

### 2. Training Engine (`nanochat/engine.py`)

Implements sophisticated training objectives:

#### Multi-Loss Training
- **Language Modeling Loss**: Cross-entropy for next-token prediction
- **Reconstruction Loss**: For continuous latent representations
- **Energy Loss**: Energy-based learning for consciousness modeling
- **Total Loss**: Weighted combination of all objectives

#### Consciousness Integration
- **Continuous Latent Space**: High-dimensional representation of model state
- **Energy-Based Modeling**: Predicts and optimizes consciousness energy
- **Self-Modeling**: Model learns to represent its own internal states

### 3. Advanced Components

#### Abacus State Memory (`nanochat/abacus_state_memory.py`)
- Implements persistent memory across sequences
- Supports stateful processing for long conversations
- Integrates with consciousness modeling

#### Hypercube (`nanochat/hypercube.py`)
- High-dimensional geometric representations
- Used for advanced reasoning and conceptual mapping
- Supports complex relationship modeling

#### Memetic Learning (`nanochat/memetic_learning.py`)
- Implements cultural evolution concepts
- Allows for idea propagation and adaptation
- Enhances model's ability to learn and evolve concepts

## Key Features

### 1. Dynamic Processing Modes
The PsycheController enables the model to adapt its processing style based on context, similar to different cognitive modes in human psychology.

### 2. Consciousness Modeling
Through energy-based learning and continuous latent representations, the model develops a form of self-awareness about its internal states.

### 3. Advanced Tokenization
Rust-accelerated BPE tokenization (`rustbpe`) provides fast, efficient text processing with custom vocabulary support.

### 4. Multi-Objective Training
The model learns simultaneously through multiple complementary objectives, leading to more robust and capable behavior.

### 5. Scalable Architecture
Supports variable model sizes through configurable parameters:
- Number of layers
- Hidden dimensions
- Attention heads
- Context length (up to 16,384 tokens)

## Training Pipeline

### Data Flow
1. **Tokenization**: Text → Tokens via rustbpe
2. **Embedding**: Tokens → Continuous representations
3. **Transformer Processing**: Multi-layer attention and feed-forward
4. **Psyche Blending**: Dynamic combination of processing modes
5. **Head Predictions**: Language modeling, concepts, and energy
6. **Loss Computation**: Multi-objective optimization

### Configuration
The model supports extensive configuration through:
- Command-line arguments
- Configuration files (`config/custom_config.py`)
- Runtime parameter adjustment

## Recursive Hierarchical Reasoning (RHR)

NanoChat incorporates several key principles of Recursive Hierarchical Reasoning (RHR) to enhance its cognitive capabilities.

### Current Implementation Overview
1.  **Multi-Level Abstraction Layers**: The PsycheController architecture (`gpt.py`) utilizes Id, Ego, and Superego layers for raw, contextual, and ethical processing, respectively, with dynamic blending based on context.
2.  **Semantic Hypercube Topology**: The Hypercube (`hypercube.py`) provides hierarchical concept organization through vertex embeddings and a continuous latent space, supporting autoregressive generation and energy-based reasoning.
3.  **Conscious Integration Layer**: The Conscious Integration Layer (`conscious_integration.py`) synthesizes outputs from various sources, incorporates memory, ensures logical consistency via the Abacus encoder, and projects synthesized states to actionable outputs.
4.  **Memetic Learning System**: The Memetic Learning Layer (`memetic_learning.py`) facilitates hierarchical knowledge evolution through fitness evaluation, concept mapping, and self-model updates.

### RHR Implementation Strengths
*   **Existing Hierarchical Features**: Three-tier processing (Id → Ego → Superego), semantic hypercube for multi-dimensional concept relationships, energy-based validation, conscious synthesis of multiple reasoning levels, and memetic evolution of concepts.
*   **Recursive Elements**: Autoregressive latent generation, memory integration, dynamic weight blending, and recursive concept mapping.

### Potential RHR Enhancements
Future enhancements could include explicit recursive loops, hierarchical attention mechanisms, and a meta-reasoning layer.

## Performance Optimizations

### 1. Rust Integration
- Custom tokenization in Rust for speed
- Potential for additional Rust-accelerated components

### 2. Memory Efficiency
- KV-cache optimization for inference
- Efficient attention implementations
- Gradient checkpointing support

### 3. Distributed Training
- Multi-GPU support
- Gradient synchronization
- Distributed data loading

## Usage Patterns

### Training
```bash
python scripts/chat_sft.py --config config/custom_config.py
```

### Evaluation
```bash
python scripts/chat_eval.py --checkpoint path/to/checkpoint
```

### Interactive Chat
```bash
python scripts/chat_cli.py --checkpoint path/to/checkpoint
```

## Architecture Diagram

```
Input Text
    ↓
Rust BPE Tokenizer
    ↓
Token Embeddings
    ↓
Transformer Blocks (N layers)
    ├── Multi-Head Attention (RoPE)
    ├── SwiGLU Feed-Forward
    └── RMSNorm + Residuals
    ↓
PsycheController
    ├── Id Processing
    ├── Ego Processing
    └── Superego Processing
    ↓
Blended Representation
    ↓
Output Heads
    ├── Language Model Head → Next Tokens
    ├── Concept Head → Concepts
    └── Energy Head → Consciousness Energy
    ↓
Multi-Loss Optimization
```

## Future Enhancements

Potential areas for improvement:
1. **KIMI Linear Attention**: Integration of linear attention mechanisms.
2. **Enhanced Consciousness**: More sophisticated self-modeling.
3. **Memory Augmentation**: Long-term memory capabilities.
4. **Multi-Modal**: Extension to handle images, audio, etc.
5. **Efficiency**: Further optimization for inference speed.

This architecture represents a significant advancement in transformer-based language models, incorporating psychological modeling, consciousness integration, and advanced training techniques.