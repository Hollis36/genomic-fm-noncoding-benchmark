# Device Compatibility Guide

## Your Current Device

**MacBook Pro (M4 Pro)**
- **Chip**: Apple M4 Pro
- **Memory**: 24 GB Unified Memory
- **GPU**: Integrated GPU cores (Metal Performance Shaders)
- **PyTorch**: 2.6.0 with MPS support ✅

## Compatibility Analysis

### ✅ **Models You CAN Run**

#### 1. **DNABERT-2 (117M)** - ✅ Fully Compatible
- **VRAM Needed**: ~2 GB
- **Status**: Will run smoothly on MPS
- **Performance**: Good inference speed
- **Recommendation**: **Best choice for your device**

#### 2. **Nucleotide Transformer v2 - 500M** - ✅ Compatible
- **VRAM Needed**: ~3 GB
- **Status**: Should run well on MPS
- **Performance**: Moderate speed
- **Recommendation**: **Good option**

#### 3. **Caduceus (~200M)** - ⚠️ Likely Compatible
- **VRAM Needed**: ~3 GB
- **Status**: Should work, but Mamba models may need testing
- **Recommendation**: Try with CPU fallback ready

### ⚠️ **Models with Limitations**

#### 4. **Nucleotide Transformer v2 - 2.5B** - ⚠️ Challenging
- **VRAM Needed**: ~6 GB
- **Status**: May work but slow
- **Issue**: Large model on integrated GPU
- **Recommendation**: Use CPU mode or skip

#### 5. **HyenaDNA (1.6B)** - ⚠️ Challenging
- **VRAM Needed**: ~4 GB
- **Status**: FP16 may work, but slow
- **Issue**: Long-range attention on MPS
- **Recommendation**: CPU mode or reduce batch size

### ❌ **Models NOT Recommended**

#### 6. **Evo-1 (7B)** - ❌ Not Compatible
- **VRAM Needed**: ~10 GB (4-bit) / ~28 GB (FP16)
- **Status**: Too large for integrated GPU
- **Issue**: Requires NVIDIA GPU for 4-bit quantization
- **Recommendation**: **Skip or use cloud GPU**

## Detailed Compatibility Matrix

| Model | Parameters | Your Device | Speed | Recommendation |
|-------|-----------|-------------|-------|----------------|
| DNABERT-2 | 117M | ✅ Excellent | Fast | **Use this** |
| NT-v2-500M | 500M | ✅ Good | Moderate | **Use this** |
| NT-v2-2.5B | 2.5B | ⚠️ Possible | Slow | CPU mode |
| HyenaDNA | 1.6B | ⚠️ Possible | Slow | CPU mode |
| Caduceus | ~200M | ⚠️ Likely OK | Moderate | Test first |
| Evo-1 | 7B | ❌ No | N/A | Cloud GPU |

## Key Limitations on Apple Silicon

### 1. **No CUDA Support**
- ❌ No `bitsandbytes` (4-bit quantization)
- ❌ No `flash-attn` (Flash Attention)
- ✅ Can use MPS (Metal) instead

### 2. **MPS Compatibility Issues**
Some operations may not be supported on MPS:
- Certain advanced attention mechanisms
- Some custom CUDA kernels
- 4-bit/8-bit quantization

### 3. **Memory Constraints**
- 24 GB shared between CPU and GPU
- OS uses ~4-6 GB
- Available for models: ~18 GB
- Can run models up to ~6GB comfortably

## Recommended Workflow for Your Device

### Option 1: Run Small Models Locally (RECOMMENDED)

```bash
# Test with DNABERT-2 first
python -m scripts.run_experiment \
    --mode zero_shot \
    --models dnabert2 \
    --negative-sets N1 N3 \
    --batch-size 8

# If successful, try NT-v2-500M
python -m scripts.run_experiment \
    --mode zero_shot \
    --models nucleotide_transformer_500m \
    --negative-sets N1 N3 \
    --batch-size 4
```

### Option 2: Use CPU Mode for Larger Models

Modify model loading to force CPU:

```python
# In src/models/base.py or model-specific files
self.device = torch.device("cpu")  # Force CPU
```

### Option 3: Cloud GPU for Large Models

See "Cloud GPU Options" below.

## Optimization Tips for Apple Silicon

### 1. Enable MPS Acceleration

```python
# Already in code, but verify:
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### 2. Reduce Batch Size

```bash
# Use smaller batches to fit in memory
--batch-size 4  # or even 2
```

### 3. Use Mixed Precision (FP16)

```python
# Already implemented in most models
model = model.half()  # Convert to FP16
```

### 4. Enable Gradient Checkpointing (for fine-tuning)

```python
model.gradient_checkpointing_enable()
```

### 5. Reduce Context Size

```bash
# Use smaller context windows
python data/scripts/05_extract_sequences.py \
    --context-size 512  # Instead of 1024
```

## Cloud GPU Options

### Option 1: Google Colab (FREE tier available)

**Pros**:
- Free T4 GPU (16GB VRAM)
- Can run all models except Evo-1
- Jupyter notebook interface
- Easy to use

**Cons**:
- Session timeout after 12 hours
- May disconnect
- Limited to ~12 hours/day

**Setup**:
```bash
# Upload your code to Google Drive
# Open Colab notebook
# Mount Drive and run experiments
```

**Cost**: Free (or $10/month for Pro)

### Option 2: Kaggle Notebooks (FREE)

**Pros**:
- Free P100 (16GB) or T4 GPU
- 30 hours/week GPU quota
- Good for medium models

**Cons**:
- Weekly quota limit
- Session management

**Cost**: Free

### Option 3: Lambda Labs (Cost-Effective)

**Pros**:
- A100 (40GB/80GB) available
- Pay per hour
- Can run ALL models including Evo-1
- Good performance

**GPU Options**:
- **RTX 4090 (24GB)**: $0.75/hour - Perfect for your benchmark
- **A100 (40GB)**: $1.10/hour
- **A100 (80GB)**: $1.29/hour

**Cons**:
- Need to pay
- Availability can vary

**Setup**:
1. Create account at https://lambdalabs.com
2. Launch instance with PyTorch
3. Clone your repo and run experiments

**Estimated Cost for Full Benchmark**:
- DNABERT-2, NT-v2-500M: ~2-4 hours = $3-6
- All models (excluding Evo-1): ~8-12 hours = $12-18
- Full benchmark with Evo-1: ~16-24 hours = $24-36

### Option 4: Vast.ai (CHEAPEST)

**Pros**:
- RTX 4090: $0.30-0.50/hour
- Very affordable
- Wide GPU selection

**Cons**:
- Quality varies
- Need to manage instances
- Some learning curve

**Cost**: ~$10-15 for full benchmark

### Option 5: Paperspace Gradient (Balanced)

**Pros**:
- Free tier with limited hours
- Easy to use
- Good documentation

**GPU Options**:
- Free tier: M4000 (limited)
- Pro tier: A100, V100

**Cost**: $8-15/month subscription + usage

### Option 6: AWS SageMaker / Google Cloud / Azure

**Pros**:
- Enterprise-grade
- Scalable
- Good for large studies

**Cons**:
- More expensive
- Complex setup
- Overkill for this benchmark

**Cost**: $1-3/hour for A100

## Recommended Strategy

### Phase 1: Local Development (Your M4 Pro)
✅ **What to do locally**:
- Data preparation (all steps)
- Code development and testing
- Small model experiments (DNABERT-2, NT-v2-500M)
- Result analysis and visualization
- Jupyter notebooks

### Phase 2: Cloud GPU for Large Models
🌩️ **What to run on cloud**:
- NT-v2-2.5B experiments
- HyenaDNA experiments
- Evo-1 experiments
- LoRA fine-tuning (faster on GPU)
- Full benchmark runs

### Recommended Split:

**Local (Your M4 Pro)** - FREE:
```bash
# Zero-shot evaluation
make zero-shot  # DNABERT-2, NT-v2-500M only

# Linear probe
python -m scripts.run_experiment \
    --mode linear_probe \
    --models dnabert2 nucleotide_transformer_500m

# LoRA fine-tuning (small models)
python -m scripts.run_experiment \
    --mode lora \
    --models dnabert2 \
    --epochs 5 \
    --batch-size 4
```

**Cloud GPU (Lambda/Vast.ai)** - $15-30:
```bash
# All models
bash scripts/run_all.sh

# Or specific large models
python -m scripts.run_experiment \
    --mode zero_shot \
    --models hyenadna evo1 nucleotide_transformer_2500m \
    --negative-sets N1 N3
```

## Modified Configuration for Your Device

Create `configs/models_m4pro.yaml`:

```yaml
# Models optimized for Apple M4 Pro

dnabert2:
  name: "zhihan1996/DNABERT-2-117M"
  type: encoder
  device: "mps"  # Use Metal
  batch_size: 8
  # ... rest of config

nucleotide_transformer_500m:
  name: "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
  type: encoder
  device: "mps"
  batch_size: 4  # Smaller batch
  # ... rest of config

# Skip or use CPU for large models
nucleotide_transformer_2500m:
  device: "cpu"  # Force CPU
  batch_size: 1
  # ... rest of config
```

## Testing Your Setup

Run this test script:

```python
# test_device.py
import torch
import sys

print("=" * 60)
print("Device Compatibility Test")
print("=" * 60)

# Check PyTorch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

# Test MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"✅ MPS computation successful!")
else:
    print(f"❌ MPS not available, using CPU")
    device = torch.device("cpu")

# Memory test
try:
    # Allocate ~4GB on device
    large_tensor = torch.randn(1024, 1024, 1024, device=device, dtype=torch.float16)
    print(f"✅ Can allocate 4GB on {device}")
    del large_tensor
except RuntimeError as e:
    print(f"⚠️  Memory allocation failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
```

Run: `python test_device.py`

## My Recommendation

### For Your Situation:

1. **Start Locally** (Your M4 Pro):
   - ✅ Run DNABERT-2 and NT-v2-500M
   - ✅ Do all data preparation
   - ✅ Test code and debug
   - ✅ Analysis and visualization
   - **Cost**: FREE
   - **Time**: 2-4 hours

2. **Use Cloud GPU** for remaining models:
   - 🌩️ Rent Lambda Labs RTX 4090 ($0.75/hour)
   - 🌩️ Run HyenaDNA, NT-v2-2.5B, Evo-1
   - 🌩️ Complete fine-tuning experiments
   - **Cost**: ~$15-25
   - **Time**: 8-12 hours

3. **Total Cost**: $15-25 for complete benchmark
4. **Total Time**: ~2-3 days

This is **much cheaper** than:
- Buying RTX 4090 (~$1,600)
- Renting workstation ($100+/month)
- Cloud GPUs full-time ($50+/month)

## Summary

### ✅ Your M4 Pro IS Suitable For:
- Development and testing
- Small model experiments (DNABERT-2, NT-v2-500M)
- Data preparation and analysis
- Most of the workflow (~60-70%)

### ❌ Need Cloud GPU For:
- Large models (HyenaDNA, NT-v2-2.5B, Evo-1)
- Fast fine-tuning
- Full benchmark runs
- ~30-40% of compute-intensive tasks

### 💰 Most Cost-Effective Solution:
**Local (M4 Pro) + Lambda Labs RTX 4090 for 12-16 hours = ~$20 total**

Would you like me to:
1. Create a modified run script optimized for your M4 Pro?
2. Provide detailed Lambda Labs setup instructions?
3. Create a Colab notebook version?
