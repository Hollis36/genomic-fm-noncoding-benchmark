## Genomic Foundation Models

This document describes the genomic foundation models included in the benchmark.

## Supported Models

### 1. DNABERT-2 (117M parameters)

**Type**: Masked Language Model (MLM) Encoder

**Architecture**: BERT-based with BPE tokenization

**Key Features**:
- Trained on multi-species genomes
- BPE tokenization for efficient sequence encoding
- Max sequence length: 512 tokens (~3kb genomic sequence)
- Embedding dimension: 768

**Zero-shot Method**: Cosine distance between ref/alt embeddings

**Citation**:
```bibtex
@article{zhou2023dnabert2,
  title={DNABERT-2: Efficient Foundation Model for Multi-Species Genome Understanding},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
  journal={arXiv preprint arXiv:2306.15006},
  year={2023}
}
```

**HuggingFace**: `zhihan1996/DNABERT-2-117M`

---

### 2. Nucleotide Transformer v2 (500M / 2.5B parameters)

**Type**: Masked Language Model (MLM) Encoder

**Architecture**: Transformer with 6-mer tokenization

**Key Features**:
- Trained on 850+ species genomes
- 6-mer tokenization (non-overlapping)
- Max sequence length: 2048 tokens (~12kb genomic sequence)
- Two variants: 500M and 2.5B parameters

**Zero-shot Method**: Cosine distance between ref/alt embeddings

**Citation**:
```bibtex
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and Lopez-Carranza, Nicolas and Henryk Grywaczewski, Adam and Oteri, Francesco and Dallago, Christian and Trop, Evan and Sirelkhatim, Hassan and Richard, Guillaume and others},
  journal={bioRxiv},
  year={2023}
}
```

**HuggingFace**:
- 500M: `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`
- 2.5B: `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`

---

### 3. HyenaDNA (1.6B parameters)

**Type**: Causal Language Model with Sub-Quadratic Attention

**Architecture**: Hyena operator (alternative to attention)

**Key Features**:
- Can handle up to 1 million bp context
- Sub-quadratic complexity O(N log N)
- Character-level tokenization
- Efficient for long-range interactions

**Zero-shot Method**: Log-likelihood ratio (LLR)

**Citation**:
```bibtex
@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Armin and Birch-Sykes, Callum and Wornow, Michael and Patel, Aman and Rabideau, Clayton and Massaroli, Stefano and Bengio, Yoshua and others},
  journal={arXiv preprint arXiv:2306.15794},
  year={2023}
}
```

**HuggingFace**: `LongSafari/hyenadna-large-1m-seqlen-hf`

---

### 4. Caduceus (~200M parameters)

**Type**: Bidirectional Mamba (State Space Model)

**Architecture**: Mamba with bidirectional processing

**Key Features**:
- Combines causal and reverse-causal Mamba layers
- Linear complexity in sequence length
- Character-level tokenization
- Efficient bidirectional context

**Zero-shot Method**: Cosine distance between ref/alt embeddings

**Citation**:
```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2403.03234},
  year={2024}
}
```

**HuggingFace**: `kuleshov-group/caduceus-ph_seqlen-131k_d256_n4`

---

### 5. Evo-1 (7B parameters)

**Type**: Causal Language Model at Single-Nucleotide Resolution

**Architecture**: StripedHyena architecture

**Key Features**:
- 7 billion parameters (requires 4-bit quantization on RTX 4090)
- Can handle 131kb context
- Character-level tokenization
- State-of-the-art long-range modeling

**Zero-shot Method**: Log-likelihood ratio (LLR)

**Citation**:
```bibtex
@article{nguyen2024sequence,
  title={Sequence Modeling and Design from Molecular to Genome Scale with Evo},
  author={Nguyen, Eric and Poli, Michael and Durrant, Matthew G and Kang, Brian and Katrekar, Dhruva and Li, David B and Bartie, Liam J and Thomas, Armin W and King, Samuel H and Brixi, Garyk and others},
  journal={bioRxiv},
  year={2024}
}
```

**HuggingFace**: `togethercomputer/evo-1-131k-base`

---

## Model Comparison

| Model | Type | Parameters | Context Length | Tokenization | VRAM (Inference) |
|-------|------|------------|----------------|--------------|------------------|
| DNABERT-2 | Encoder | 117M | 512 tokens (~3kb) | BPE | ~2 GB |
| NT-v2-500M | Encoder | 500M | 2048 tokens (~12kb) | 6-mer | ~3 GB |
| NT-v2-2.5B | Encoder | 2.5B | 2048 tokens (~12kb) | 6-mer | ~6 GB |
| HyenaDNA | Causal | ~1.6B | 8192 bp | Character | ~4 GB |
| Caduceus | Bidirectional Mamba | ~200M | 8192 bp | Character | ~3 GB |
| Evo-1 | Causal | 7B | 8192 bp | Character | ~10 GB (4-bit) |

## Adding New Models

To add a new model to the benchmark:

1. **Implement the model wrapper** in `src/models/`:
   - Inherit from `GenomicModelBase`
   - Implement `load()`, `get_embedding()`, and `score_variant()`

2. **Add model configuration** to `configs/models.yaml`:
   ```yaml
   your_model:
     name: "huggingface/model-name"
     type: encoder  # or causal
     params: 1B
     embedding_dim: 1024
     max_length: 2048
     zero_shot_method: embedding_distance  # or log_likelihood_ratio
     lora:
       r: 8
       alpha: 16
       target_modules: ["query", "value"]
   ```

3. **Register the model** in `src/models/registry.py`

4. **Write tests** in `tests/unit/test_your_model.py`

5. **Run experiments** and document results

## Model Selection Guidelines

**For most non-coding variant tasks**: Start with DNABERT-2 or NT-v2-500M
- Fast inference
- Good balance of performance and efficiency
- Lower hardware requirements

**For long-range regulatory elements**: Use HyenaDNA or Evo-1
- Better at capturing distant interactions
- Required for sequences >3kb

**For deep intronic variants**: Use models with long context
- Caduceus or Evo-1
- Can model splicing effects at a distance

**For production deployment**: Consider NT-v2-500M or Caduceus
- Reasonable size and speed
- Good performance across region types

## Hardware Requirements

**Minimum**: 1x NVIDIA RTX 4090 (24GB VRAM)
- Can run all models with appropriate quantization
- DNABERT-2, NT-v2-500M, Caduceus run without quantization
- Evo-1 requires 4-bit quantization

**Recommended**: 1x NVIDIA A100 (80GB VRAM)
- Run all models without quantization
- Faster training and inference
- Support larger batch sizes

**For CPU-only**: Use DNABERT-2 or smaller models
- Expect significantly slower inference
- Not recommended for large-scale benchmarking
