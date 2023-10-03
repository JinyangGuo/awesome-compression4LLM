# awesome-compression4LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo collects efficient approaches for LLM (Large Language Model) to cope with its huge demand for computing resources. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. 

## Table of Contents

- [Pruning](#Pruning)
- [Knowledge Distillation](#Knowledge-Distillation)
- [Quantization](#Quantization)
- [Low-Rank Factorization](#Low-Rank-Factorization)

## Pruning

### Unstructured Pruning

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[Paper](https://arxiv.org/abs/2301.00774)]
- Prune and Tune: Improving Efficient Pruning Techniques for Massive Language Models [[Paper](https://openreview.net/forum?id=cKlgcx7nSZ)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- A Simple and Effective Pruning Approach for Large Language Models [[Paper](https://arxiv.org/abs/2306.11695)]

### Structured Pruning

- LLM-Pruner: On the Structural Pruning of Large Language Models [[Paper](https://arxiv.org/abs/2305.11627)]
- Knowledge-preserving Pruning for Pre-trained Language Models without Retraining [[Paper](https://arxiv.org/pdf/2308.03449.pdf)]
- Pruning Large Language Models via Accuracy Predictor [[Paper](https://arxiv.org/pdf/2309.09507.pdf)]

### Semi-structured Pruning

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[Paper](https://arxiv.org/abs/2301.00774)]
- A Simple and Effective Pruning Approach for Large Language Models [[Paper](https://arxiv.org/abs/2306.11695)]

## Knowledge Distillation

### Standard Distillation

- Knowledge Distillation of Large Language Models [[Paper](https://arxiv.org/abs/2306.08543)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models [[Paper](https://arxiv.org/abs/2306.13649)]

### EA-based Distillation

#### In-Context Learning

- In-context learning distillation: Transferring few-shot learning ability of pre-trained language models [[Paper](https://arxiv.org/pdf/2212.10670)]

#### Chain-of-Thought

- Explanations from Large Language Models Make Small Reasoners Better [[Paper](https://arxiv.org/pdf/2210.06726.pdf)]
- Large Language Models Are Reasoning Teachers [[Paper](https://arxiv.org/pdf/2212.10071.pdf)]
- Specializing Smaller Language Models towards Multi-Step Reasoning [[Paper](https://arxiv.org/pdf/2301.12726.pdf)]
- Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes [[Paper](https://arxiv.org/pdf/2305.02301.pdf)]
- Distilling Reasoning Capabilities into Smaller Language Models [[Paper](https://aclanthology.org/2023.findings-acl.441.pdf)]
- DISCO: distilling counterfactuals with large language models [[Paper](https://aclanthology.org/2023.acl-long.302/)]
- SCOTT: self-consistent chain-of-thought distillation [[Paper](https://arxiv.org/abs/2305.01879)]
- Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step [[Paper](https://arxiv.org/abs/2306.14050)]
- Effective Distillation of Table-based Reasoning Ability from LLMs [[Paper](https://arxiv.org/abs/2309.13182)]

#### Instruction Following

- Lion: Adversarial distillation of closed-source large language model [[Ppaer](https://arxiv.org/abs/2305.12870)]

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
- SqueezeLLM: Dense-and-Sparse Quantization [[Paper](https://arxiv.org/abs/2306.07629)]
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs [[Paper](https://arxiv.org/abs/2308.09723)]
- eDKM: An Efficient and Accurate Train-time Weight Clustering for Large Language Models [[Paper](https://arxiv.org/abs/2309.00964)]

#### Weight and Activation Quantization

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2211.10438)]
- RPTQ: Reorder-based Post-training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2304.01089)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization [[Paper](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)]
- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling [[Paper](https://arxiv.org/abs/2304.09145)]
- Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models [[Paper](https://arxiv.org/abs/2305.12356)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation [[Paper](https://arxiv.org/abs/2303.08302)]
- QuantEase: Optimization-based Quantization for Language Models - An Efficient and Intuitive Algorithm [[Paper](https://arxiv.org/abs/2309.01885)]
- Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs [[Paper](https://arxiv.org/abs/2309.05516)]
- OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2308.13137)]
- FPTQ: Fine-grained Post-Training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2308.15987)]
- Norm Tweaking: High-performance Low-bit Quantization of Large Language Models [[Paper](https://arxiv.org/abs/2309.02784)]

## Low-Rank Factorization

- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation [[Paper](https://arxiv.org/abs/2306.11222)]
- IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning [[Paper](https://arxiv.org/abs/2308.12043)]
