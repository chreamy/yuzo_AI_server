"""
Optimized version of the benchmarking script
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class BenchConfig:
    batch_size = 12
    block_size = 1024
    bias = False
    real_data = True
    seed = 1337
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = True
    profile = False
    gradient_accumulation_steps = 4  # New: accumulate gradients
    use_amp = True                   # New: automatic mixed precision
    num_workers = 4                  # New: dataloader workers
    pin_memory = True               # New: pin memory for faster data transfer
    use_ddp = False                 # New: distributed training option

# -----------------------------------------------------------------------------
# Optimization 1: Better Data Loading
# -----------------------------------------------------------------------------
def create_dataloader(train_data, config):
    from torch.utils.data import Dataset, DataLoader
    
    class GPTDataset(Dataset):
        def __init__(self, data, block_size):
            self.data = data
            self.block_size = block_size
            
        def __len__(self):
            return len(self.data) - self.block_size
            
        def __getitem__(self, idx):
            x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
            y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
            return x, y
    
    dataset = GPTDataset(train_data, config.block_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True
    )

# -----------------------------------------------------------------------------
# Optimization 2: Training Loop with Optimizations
# -----------------------------------------------------------------------------
def train_loop(model, optimizer, dataloader, config, scaler):
    model.train()
    total_steps = 20  # Configurable steps for benchmark
    accumulated_loss = 0
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        with_flops=True,
    ) if config.profile else nullcontext() as prof:
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        for step, (X, Y) in enumerate(dataloader):
            if step >= total_steps:
                break
                
            # Optimization 3: Gradient Accumulation
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                logits, loss = model(X.to(config.device), Y.to(config.device))
                loss = loss / config.gradient_accumulation_steps
            
            # Optimization 4: Mixed Precision Training
            if config.use_amp:
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            accumulated_loss += loss.item()
            
            if config.profile:
                prof.step()
            
            if step % config.gradient_accumulation_steps == 0:
                print(f"Step {step}/{total_steps}, Loss: {accumulated_loss:.4f}")
                accumulated_loss = 0

        torch.cuda.synchronize()
        dt = time.time() - t0
        mfu = model.estimate_mfu(config.batch_size * total_steps, dt)
        print(f"Time per iteration: {dt/total_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    config = BenchConfig()
    
    # Set up distributed training if enabled
    if config.use_ddp:
        dist.init_process_group(backend='nccl')
        config.device = f'cuda:{dist.get_rank()}'
        torch.cuda.set_device(config.device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize model
    model = GPT(GPTConfig(
        block_size=config.block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0,
        bias=config.bias
    )).to(config.device)
    
    if config.use_ddp:
        model = DDP(model, device_ids=[dist.get_rank()])
    
    if config.compile:
        model = torch.compile(model)
    
    # Optimization 5: Configure optimizer with parameters
    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in config.device else 'cpu'
    )
    
    # Optimization 6: Gradient Scaler for AMP
    scaler = GradScaler() if config.use_amp else None
    
    # Data loading
    if config.real_data:
        train_data = np.memmap(
            os.path.join('data', 'openwebtext', 'train.bin'),
            dtype=np.uint16,
            mode='r'
        )
        dataloader = create_dataloader(train_data, config)
    else:
        # Use synthetic data for benchmarking
        X = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
        Y = torch.randint(50304, (config.batch_size, config.block_size), device=config.device)
        dataloader = [(X, Y)] * 20  # 20 synthetic batches
    
    # Run training loop
    train_loop(model, optimizer, dataloader, config, scaler)
    
    if config.use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
