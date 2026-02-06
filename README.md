# Select-Then-Train: Efficient Transformer Adaptation via Neuron Selection

This repository contains the official implementation of **Select-Then-Train (STT)**, an efficient method for adapting large transformer models by selecting and training only the most important neurons.

## Overview

STT identifies and retains critical neurons in pre-trained transformers, then selectively trains only these neurons to achieve efficient adaptation with minimal performance degradation.

## Installation

### Requirements

- Python 3.9+
- CUDA 11.7+ (for GPU support)
- PyTorch 2.2+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sizchode/Select-Then-Train.git
cd Select-Then-Train
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Quick Start

We provide SLURM job scripts in the `scripts/` directory for running experiments. Before using them, you need to:

1. **Configure the scripts**: Edit the scripts in `scripts/` directory and fill in the TODO items:
   - Set your SLURM job parameters (partition, GPU type, etc.)
   - Set your environment paths (conda path, Python path, etc.)
   - Set your personal tokens (HuggingFace token, WandB entity/project)
   - Define your experimental parameters (models, datasets, modes, hyperparameters)

2. **Run LLM experiments**:
   ```bash
   cd scripts
   sbatch test_llm        # For LLM training experiments
   sbatch analyze_llm      # For LLM ablation studies
   ```

3. **Run classification experiments**:
   ```bash
   cd scripts
   sbatch image_exp       # For ViTs
   sbatch text_exp        # For Bert 
   ```

**Available training modes:**
- `stt`: Select-Then-Train with neuron selection
- `stt_lora`: STT combined with LoRA
- `magnitude_pruning`: Magnitude-based pruning baseline
- `wanda_adapt`: Wanda adaptive pruning
- `baseline`: Full fine-tuning baseline

For direct Python execution without SLURM, you can run the training scripts directly:
- `python train_llm.py --help` for LLM training options
- `python train_classifier.py --help` for classification training options

## Tutorial: How to Use STT

This section provides code examples for applying STT to different model types. The general workflow consists of three steps: (1) neuron selection using `STTTracker`, (2) model transformation using `STTTransformer`, and (3) training the transformed model.

### For Vision Transformers (ViTs)

```python
import torch
from stt.stt_tracker import STTTracker
from stt.stt_transformer import STTTransformer
from stt.mlps.stt_linear2 import STTLinear
from transformers import AutoModelForImageClassification

# 1. Load your model
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Prepare a small sample of training data for neuron selection
# active_dataloader should contain a subset of your training data
from torch.utils.data import DataLoader
active_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
active_dataloader = list(active_dataloader)[:100]  # Use first 100 batches

# 3. Initialize STTTracker and select neurons
tracker = STTTracker(
    model=model,
    tokenizer=None,  # No tokenizer for image models
    threshold=0.01,
    topk_ratio=0.1,  # Keep top 10% of neurons
    use_abs_threshold=True,
    device=device,
    track_attention_proj=False,
    verbose=True
)

active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
layer_name_map = tracker.get_layer_name_map()

# 4. Transform the model using STTTransformer
transformer = STTTransformer(
    model=model,
    active_neurons=active_neurons,
    layer_name_map=layer_name_map,
    verbose=True,
    tune_pruned=False,
    device=device,
    inference_time=False  # Set to True for faster inference
)

pruned_model = transformer.transform().to(device)

# 5. Set trainable parameters (only STTLinear layers and classifier head)
for param in pruned_model.parameters():
    param.requires_grad = False

for name, module in pruned_model.named_modules():
    if isinstance(module, STTLinear):
        for param in module.parameters():
            param.requires_grad = True
    # Also train classifier head
    if any(key in name for key in ["classifier", "score"]):
        for param in module.parameters():
            param.requires_grad = True

# 6. Train the model as usual
optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=1e-4)
# ... training loop ...
```

### For BERT and Text Classification Models

```python
import torch
from stt.stt_tracker import STTTracker
from stt.stt_transformer import STTTransformer
from stt.mlps.stt_linear2 import STTLinear
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Prepare active dataloader (subset of training data)
active_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
active_dataloader = list(active_dataloader)[:100]

# 3. Initialize STTTracker
tracker = STTTracker(
    model=model,
    tokenizer=tokenizer,  # Tokenizer required for text models
    threshold=0.01,
    topk_ratio=0.1,
    use_abs_threshold=True,
    device=device,
    track_attention_proj=False,
    verbose=True
)

active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
layer_name_map = tracker.get_layer_name_map()

# 4. Transform model
transformer = STTTransformer(
    model=model,
    active_neurons=active_neurons,
    layer_name_map=layer_name_map,
    verbose=True,
    tune_pruned=False,
    device=device,
    inference_time=False
)

pruned_model = transformer.transform().to(device)

# 5. Set trainable parameters
for param in pruned_model.parameters():
    param.requires_grad = False

for name, module in pruned_model.named_modules():
    if isinstance(module, STTLinear):
        for param in module.parameters():
            param.requires_grad = True
    if any(key in name for key in ["classifier", "pooler"]):
        for param in module.parameters():
            param.requires_grad = True

# 6. Train the model
optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=1e-4)
# ... training loop ...
```

### For Large Language Models (LLMs)

```python
import torch
from stt.stt_tracker import STTTracker
from stt.stt_transformer import STTTransformer
from stt.mlps.stt_linear2 import STTLinear
from stt.stt_lora import STTLoraLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load model and tokenizer
model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Prepare active dataloader (subset of training data)
# For LLMs, you typically use a small sample ratio (e.g., 1% of training data)
active_subset = train_dataset[:len(train_dataset) // 100]  # 1% sample
active_dataloader = DataLoader(active_subset, batch_size=8)

# 3. Initialize STTTracker
tracker = STTTracker(
    model=model,
    tokenizer=tokenizer,
    threshold=0.01,
    topk_ratio=0.1,
    use_abs_threshold=True,
    device=device,
    track_attention_proj=False,
    verbose=True
)

active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
layer_name_map = tracker.get_layer_name_map()

# 4. Transform model
transformer = STTTransformer(
    model=model,
    active_neurons=active_neurons,
    layer_name_map=layer_name_map,
    verbose=True,
    tune_pruned=False,
    device=device,
    inference_time=False
)

pruned_model = transformer.transform().to(device)

# 5. Optional: Apply STT+LoRA for even more parameter efficiency
# Replace STTLinear layers with STTLoraLinear
for name, module in pruned_model.named_modules():
    if isinstance(module, STTLinear):
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        attr_name = name.split('.')[-1]
        parent = pruned_model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        
        stt_lora = STTLoraLinear(
            stt_linear=module,
            r=32,  # LoRA rank
            lora_alpha=64,
            lora_dropout=0.1,
            merge_weights=False
        )
        setattr(parent, attr_name, stt_lora)

# 6. Set trainable parameters
for param in pruned_model.parameters():
    param.requires_grad = False

for name, module in pruned_model.named_modules():
    if isinstance(module, (STTLinear, STTLoraLinear)):
        for param in module.parameters():
            param.requires_grad = True
    # Also train output head (lm_head)
    if "lm_head" in name:
        for param in module.parameters():
            param.requires_grad = True

# 7. Train with your preferred training framework (e.g., HuggingFace Trainer)
# ... training setup ...
```

### Key Parameters

- **`topk_ratio`**: Fraction of neurons to keep (e.g., 0.1 = keep top 10%)
- **`threshold`**: Activation threshold for neuron selection
- **`inference_time`**: If `True`, uses direct pruning for faster inference (requires padding). If `False`, uses scatter mode for flexible training.
- **`tune_pruned`**: Whether to fine-tune the pruned weights (typically `False`)

For more details, refer to the implementation in `train_classifier.py` (ViTs/BERTs) and `train_llm.py` (LLMs).

## Project Structure

```
stt/
├── stt/                    # Core STT implementation
│   ├── stt_tracker.py      # Neuron selection tracker
│   ├── stt_transformer.py  # Transformer adaptation
│   ├── stt_linear.py       # Selective linear layers
│   ├── stt_lora.py         # STT + LoRA integration
│   ├── wanda_adapt_tracker.py  # Wanda adaptive pruning tracker
│   ├── ablation_tracker.py     # Ablation study tracker
│   ├── dataset/            # Dataset loaders
│   ├── trainers/           # Custom trainers
│   └── evaluate/           # Evaluation utilities
├── util/                   # Utilities
│   ├── torch_flops.py      # FLOPs estimation
│   └── utils.py            # Helper functions
├── scripts/                # SLURM job scripts
│   ├── analyze_llm         # LLM ablation experiments
│   ├── test_llm            # LLM training experiments
│   ├── image_exp           # Image classification experiments
│   └── text_exp            # Text classification experiments
├── datasets/               # Dataset files
│   └── clutrr/           # CLUTRR dataset (included)
├── train_llm.py           # LLM training script
├── train_classifier.py    # Classification training script
├── llm_ablation.py        # LLM ablation studies
├── classifier_ablation.py # Classification ablation studies
├── META.py                # Dataset path configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
