# awesome-compression4LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo collects efficient approaches for LLM (Large Language Model) to cope with its huge demand for computing resources. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. 

## Table of Contents

- [Pruning](#Pruning)
- [Knowledge Distillation](#Knowledge Distillation)
- [Quantization](#Quantization)
- [Low-Rank Factorization](Low-Rank Factorization)

## Pruning

### Unstructured Pruning

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[Paper](https://arxiv.org/abs/2301.00774)]
- Prune and Tune: Improving Efficient Pruning Techniques for Massive Language Models [[Paper](https://openreview.net/forum?id=cKlgcx7nSZ)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- A Simple and Effective Pruning Approach for Large Language Models [[Paper](https://arxiv.org/abs/2306.11695)]

### Structured Pruning

- LLM-Pruner: On the Structural Pruning of Large Language Models [[Paper](https://arxiv.org/abs/2305.11627)]

## Knowledge Distillation

- Knowledge Distillation of Large Language Models [[Paper](https://arxiv.org/abs/2306.08543)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models [[Paper](https://arxiv.org/abs/2306.13649)]

## Quantization

### Quantization-Aware Training

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models [[Paper](https://arxiv.org/abs/2305.17888)]

### Quantization-Aware Fine-tuning

- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization [[Paper](https://arxiv.org/abs/2305.14152)]
- QLoRA: Efficient Finetuning of Quantized LLMs [[Paper](https://arxiv.org/abs/2305.14314)]

### Post-Training Quantization

#### Weight Quantization

- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models [[Paper](https://arxiv.org/abs/2206.09557)]
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale [[Paper](https://arxiv.org/abs/2208.07339)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)]
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [[Paper](https://arxiv.org/abs/2210.17323)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration [[Paper](https://arxiv.org/abs/2306.00978)]
- OWQ: Lessons learned from activation outliers for weight quantization in large language models [[Paper](https://arxiv.org/abs/2306.02272)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [[Paper](https://arxiv.org/abs/2306.03078)]

#### Weight and Activation Quantization

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2211.10438)]
- RPTQ: Reorder-based Post-training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2304.01089)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization [[Paper](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)]
- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling [[Paper](https://arxiv.org/abs/2304.09145)]
- Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models [[Paper](https://arxiv.org/abs/2305.12356)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]

## Low-Rank Factorization

- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]
