# awesome-compression4LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo collects efficient approaches for LLM (Large Language Model) to cope with its huge demand for computing resources. We are continuously improving the project. Welcome to PR the works (papers, repositories) that are missed by the repo. 

## Table of Contents

- [Pruning](#Pruning)
- [Quantization](#Quantization)
- [Knowledge Distillation](#Knowledge-Distillation)
- [Low-Rank Factorization](#Low-Rank-Factorization)

## Pruning

### Unstructured Pruning

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- Prune and Tune: Improving Efficient Pruning Techniques for Massive Language Models [[Paper](https://openreview.net/forum?id=cKlgcx7nSZ)]
- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- A Simple and Effective Pruning Approach for Large Language Models [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]

### Structured Pruning

- LLM-Pruner: On the Structural Pruning of Large Language Models [[Paper](https://arxiv.org/abs/2305.11627)] [[Code](https://github.com/horseee/LLM-Pruner)]
- Knowledge-preserving Pruning for Pre-trained Language Models without Retraining [[Paper](https://arxiv.org/abs/2308.03449)]
- Pruning Large Language Models via Accuracy Predictor [[Paper](https://arxiv.org/abs/2309.09507)]

### Semi-structured Pruning

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot [[Paper](https://arxiv.org/abs/2301.00774)] [[Code](https://github.com/IST-DASLab/sparsegpt)]
- A Simple and Effective Pruning Approach for Large Language Models [[Paper](https://arxiv.org/abs/2306.11695)] [[Code](https://github.com/locuslab/wanda)]

## Quantization

### Quantization-Aware Training

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models [[Paper](https://arxiv.org/abs/2305.17888)] [[Code](https://github.com/facebookresearch/LLM-QAT)]

### Quantization-Aware Fine-tuning

- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization [[Paper](https://arxiv.org/abs/2305.14152)]
- QLoRA: Efficient Finetuning of Quantized LLMs [[Paper](https://arxiv.org/abs/2305.14314)] [[Code](https://github.com/artidoro/qlora)]

### Post-Training Quantization

#### Weight Quantization

- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models [[Paper](https://arxiv.org/abs/2206.09557)]
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale [[Paper](https://arxiv.org/abs/2208.07339)]
- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)]
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [[Paper](https://arxiv.org/abs/2210.17323)] [[Code](https://github.com/IST-DASLab/gptq)]
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration [[Paper](https://arxiv.org/abs/2306.00978)] [[Code](https://github.com/mit-han-lab/llm-awq)]
- OWQ: Lessons learned from activation outliers for weight quantization in large language models [[Paper](https://arxiv.org/abs/2306.02272)] [[Code](https://github.com/xvyaward/owq)]
- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [[Paper](https://arxiv.org/abs/2306.03078)] [[Code](https://github.com/Vahe1994/SpQR)]
- SqueezeLLM: Dense-and-Sparse Quantization [[Paper](https://arxiv.org/abs/2306.07629)] [[Code](https://github.com/SqueezeAILab/SqueezeLLM)]
- FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs [[Paper](https://arxiv.org/abs/2308.09723)]
- eDKM: An Efficient and Accurate Train-time Weight Clustering for Large Language Models [[Paper](https://arxiv.org/abs/2309.00964)]

#### Weight and Activation Quantization

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2211.10438)] [[Code](https://github.com/mit-han-lab/smoothquant)] 
- RPTQ: Reorder-based Post-training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2304.01089)] [[Code](https://github.com/hahnyuan/RPTQ4LLM)]
- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization [[Paper](https://dl.acm.org/doi/abs/10.1145/3579371.3589038)]
- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling [[Paper](https://arxiv.org/abs/2304.09145)] [[Code](https://github.com/ModelTC/Outlier_Suppression_Plus)]
- Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models [[Paper](https://arxiv.org/abs/2305.12356)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]
- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation [[Paper](https://arxiv.org/abs/2303.08302)]
- QuantEase: Optimization-based Quantization for Language Models - An Efficient and Intuitive Algorithm [[Paper](https://arxiv.org/abs/2309.01885)]
- Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs [[Paper](https://arxiv.org/abs/2309.05516)]
- OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2308.13137)] [[Code](https://github.com/OpenGVLab/OmniQuant)]
- FPTQ: Fine-grained Post-Training Quantization for Large Language Models [[Paper](https://arxiv.org/abs/2308.15987)]
- Norm Tweaking: High-performance Low-bit Quantization of Large Language Models [[Paper](https://arxiv.org/abs/2309.02784)]

## Knowledge Distillation

### Standard Distillation

- Knowledge Distillation of Large Language Models [[Paper](https://arxiv.org/abs/2306.08543)] [[Code](https://github.com/microsoft/LMOps/tree/main/minillm)]
- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models [[Paper](https://arxiv.org/abs/2306.13649)]

### EA-based Distillation

#### In-Context Learning

- In-context learning distillation: Transferring few-shot learning ability of pre-trained language models [[Paper](https://arxiv.org/abs/2212.10670)]

#### Chain-of-Thought

- Explanations from Large Language Models Make Small Reasoners Better [[Paper](https://arxiv.org/abs/2210.06726)]
- Large Language Models Are Reasoning Teachers [[Paper](https://arxiv.org/abs/2212.10071)] [[Code](https://github.com/itsnamgyu/reasoning-teacher)]
- Specializing Smaller Language Models towards Multi-Step Reasoning [[Paper](https://arxiv.org/abs/2301.12726)] [[Code](https://github.com/FranxYao/FlanT5-CoT-Specialization)]
- Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes [[Paper](https://arxiv.org/abs/2305.02301)] [[Code](https://github.com/google-research/distilling-step-by-step)]
- Distilling Reasoning Capabilities into Smaller Language Models [[Paper](https://aclanthology.org/2023.findings-acl.441.pdf)]
- DISCO: distilling counterfactuals with large language models [[Paper](https://aclanthology.org/2023.acl-long.302/)]
- SCOTT: self-consistent chain-of-thought distillation [[Paper](https://arxiv.org/abs/2305.01879)] [[Code](https://github.com/wangpf3/consistent-CoT-distillation)]
- Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step [[Paper](https://arxiv.org/abs/2306.14050)]
- Effective Distillation of Table-based Reasoning Ability from LLMs [[Paper](https://arxiv.org/abs/2309.13182)]

#### Instruction Following

- Lion: Adversarial distillation of closed-source large language model [[Ppaer](https://arxiv.org/abs/2305.12870)] [[Code](https://github.com/YJiangcm/Lion)]

## Low-Rank Factorization

- Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning [[Paper](https://arxiv.org/abs/2305.18403)]
- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [[Paper](https://arxiv.org/abs/2307.09782)]
- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation [[Paper](https://arxiv.org/abs/2306.11222)] [[Code](https://github.com/yxli2123/LoSparse)]
- IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning [[Paper](https://arxiv.org/abs/2308.12043)] [[Code](https://github.com/FeiyuZhang98/IncreLoRA)]
