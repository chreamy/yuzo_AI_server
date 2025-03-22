# YuzoGPT: A Minimalist Implementation of GPT

## Overview
YuzoGPT is a streamlined, educational implementation of the GPT (Generative Pre-trained Transformer) architecture. This project aims to provide a simple, hackable codebase that allows researchers and developers to experiment with transformer-based language models.

## Key Features
- Clean, minimalist implementation of GPT architecture
- Support for training from scratch or finetuning existing models
- Compatible with OpenAI's GPT-2 checkpoints
- Efficient multi-GPU training using PyTorch DDP
- Support for character-level and BPE tokenization

## Installation

### Dependencies
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

#### Required Packages:
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing library
- **Transformers**: Huggingface's transformer library (for GPT-2 checkpoints)
- **Datasets**: Huggingface's dataset library (for OpenWebText processing)
- **Tiktoken**: OpenAI's fast BPE tokenizer
- **Weights & Biases**: Optional logging and visualization
- **TQDM**: Progress bar utilities

## Quick Start Guide

### Shakespeare Character-Level Model

#### For Beginners
Start with training a character-level GPT on Shakespeare's works - perfect for getting familiar with the basics.

1. **Prepare the Data**:
```bash
python data/shakespeare_char/prepare.py
```
This creates `train.bin` and `val.bin` in the data directory.

2. **Training Options**:

##### GPU Training (Recommended)
```bash
python train.py config/train_shakespeare_char.py
```
- Training time: ~3 minutes on an A100 GPU
- Model architecture: 6-layer transformer, 384 channels, 6 attention heads
- Context size: 256 characters

##### CPU Training (Limited Resources)
```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
    --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
    --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 \
    --lr_decay_iters=2000 --dropout=0.0
```
- Training time: ~3 minutes on modern CPU
- Smaller model for resource-constrained environments

3. **Generate Samples**:
```bash
python sample.py --out_dir=out-shakespeare-char
```

### Advanced Usage: GPT-2 Reproduction

#### OpenWebText Dataset Preparation
```bash
python data/openwebtext/prepare.py
```
Downloads and tokenizes OpenWebText using GPT-2 BPE tokenization.

#### Multi-GPU Training
```bash
# Single node, 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Multi-node training (example with 2 nodes)
# On master node (123.456.123.456):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# On worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

## Model Performance Benchmarks

### Language Model Performance on OpenWebText

| Model | Parameters | Train Loss | Val Loss | Status |
|-------|------------|------------|----------|---------|
| GPT-2 | 124M | 3.11 | 3.12 | ✓ |
| GPT-2 Medium | 350M | 2.85 | 2.84 | ✓ |
| GPT-2 Large | 774M | 2.66 | 2.67 | ✓ |
| GPT-2 XL | 1558M | 2.56 | 2.54 | ✓ |
| GPT-3 | 175B | 2.27 | 2.28 | Theoretical |
| GPT-4 | ~1.76T* | 1.95 | 1.97 | Theoretical |

\* GPT-4's parameter count is an estimate as OpenAI has not disclosed the exact number.

**Note**: There is a domain gap between OpenAI's WebText and OpenWebText. Finetuning GPT-2 (124M) on OpenWebText achieves ~2.85 loss, providing a more appropriate baseline for reproduction.

## Finetuning Guide

Finetuning adapts pretrained models to specific datasets with minimal training:

1. **Prepare Custom Dataset**:
```bash
# Example with Shakespeare dataset
python data/shakespeare/prepare.py
```

2. **Launch Finetuning**:
```bash
python train.py config/finetune_shakespeare.py
```

Key considerations:
- Initialize from pretrained checkpoints using `init_from`
- Use smaller learning rates than training from scratch
- Adjust model size and context length based on available memory
- Best checkpoint saves to `out_dir` specified in config

## Sampling / Inference

Generate text from pretrained or custom models:

```bash
# Sample from OpenAI's GPT-2 XL
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100

# Sample from custom model
python sample.py --out_dir=your_model_dir
# Load prompt from file
python sample.py --start=FILE:prompt.txt
```

## Performance Optimization

- Uses PyTorch 2.0's `torch.compile()` for significant speedup
- Benchmark your models using `bench.py`
- For multi-node training, verify network performance with `iperf3`
- Use `NCCL_IB_DISABLE=1` if running without Infiniband

## Future Development

Planned improvements:
- [ ] FSDP implementation alongside DDP
- [ ] Zero-shot perplexity evaluation on standard benchmarks
- [ ] Hyperparameter optimization for finetuning
- [ ] Dynamic batch size scheduling
- [ ] Additional embedding options (rotary, alibi)
- [ ] Optimized checkpoint storage
- [ ] Enhanced network health monitoring
- [ ] Initialization improvements

