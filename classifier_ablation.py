import argparse
import os
from datetime import datetime
import time

import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    SiglipVisionModel,
    SiglipModel,
    CLIPModel,
    CLIPVisionModel
)
from stt.stt_lora import NSLoraLinear  # Import the NSLoraLinear class
from stt.mlps.stt_linear2 import NeuroselectiveLinear  # Use the selective linear layer from stt_linear2.py
from stt.stt_transformer import NeuroselectiveTransformer5  # Use the pruner from stt_transformer.py
from stt.stt_tracker import NeuronTracker6 as NeuronTracker
from stt.ablation_tracker import NeuronTracker7 as AblationNeuronTracker

from stt.dataset import get_dataloader
from util.torch_flops import (
    flops_forward,          
    flops_train_step,    
    extract_inputs_from_batch,
    LLM_MAXLEN,
    TEXT_MAXLEN
)
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
# Import helper functions
from util.utils import (
    set_seed,
    print_model_layers,
    sample_active_set,
    evaluate_classification,
    setup_lora,
    setup_stt_lora,
    VisionEncoderWithClassifier,
    calculate_jaccard_similarity,
    calculate_directed_coverage
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support 
from typing import Any, Optional
import math
import torch
from torch.utils.data import DataLoader
from torch import nn
MAG_TP_K_MAP = None
MAG_TP_LAYER_NAME_MAP = None


def infer_vit_seq_len(model: Optional[nn.Module] = None,
                      H: Optional[int] = None,
                      W: Optional[int] = None,
                      add_cls: bool = True,
                      default_image_size: int = 224,
                      default_patch: int = 16) -> int:

    img_size = None
    patch = None
    if model is not None and hasattr(model, "config"):
        cfg = model.config
        img_size = getattr(cfg, "image_size", img_size)
        patch    = getattr(cfg, "patch_size", patch)
        vcfg     = getattr(cfg, "vision_config", None)
        if vcfg is not None:
            img_size = getattr(vcfg, "image_size", img_size)
            patch    = getattr(vcfg, "patch_size", patch)
    if H is None or W is None:
        s = int(img_size) if img_size is not None else int(default_image_size)
        H = H or s
        W = W or s
    P = int(patch) if patch is not None else int(default_patch)
    h = math.ceil(H / P)
    w = math.ceil(W / P)
    toks = h * w + (1 if add_cls else 0)
    return int(toks)

def peek_vit_dataloader(dl: DataLoader,
                        model: Optional[nn.Module] = None,
                        n_batches: int = 2,
                        prefix: str = "[VIT]") -> int:

    print(f"{prefix} dl.type = {type(dl)}  dataset.type = {type(getattr(dl,'dataset', None))}")
    try:
        print(f"{prefix} dataset.len = {len(dl.dataset)}  batch_size = {getattr(dl, 'batch_size', None)}")
    except Exception:
        pass
    cf = getattr(dl, "collate_fn", None)
    print(f"{prefix} collate_fn = {getattr(cf, '__name__', str(cf))}")

    D_partial = 0
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        print(f"{prefix} ----- batch {i} -----")
        H = W = None
        B = None

        if isinstance(batch, dict):
            keys = list(batch.keys())
            print(f"{prefix} dict.keys = {keys}")
            if "pixel_values" in batch and torch.is_tensor(batch["pixel_values"]):
                pv = batch["pixel_values"]  # (B,C,H,W)
                B, C, H, W = int(pv.size(0)), int(pv.size(1)), int(pv.size(2)), int(pv.size(3))
                print(f"{prefix} pixel_values: shape={(B,C,H,W)} dtype={pv.dtype} device={pv.device}")
            else:
                for k, v in batch.items():
                    if torch.is_tensor(v) and v.dim() == 4:
                        B, C, H, W = int(v.size(0)), int(v.size(1)), int(v.size(2)), int(v.size(3))
                        print(f"{prefix} {k}: shape={(B,C,H,W)} dtype={v.dtype} device={v.device}")
                        break

        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            x = batch[0]
            if torch.is_tensor(x) and x.dim() == 4:
                B, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
                print(f"{prefix} tensor[0]: shape={(B,C,H,W)} dtype={x.dtype} device={x.device}")
            elif isinstance(x, (list, tuple)) and len(x) > 0:
                try:
                    im0 = x[0]
                    if hasattr(im0, "size"):
                        W, H = im0.size  # PIL (W,H)
                        B = len(x)
                        print(f"{prefix} PIL list: B={B} HxW={H}x{W}")
                except Exception:
                    print(f"{prefix} unknown list/tuple content; preview type={type(x)}")
            else:
                print(f"{prefix} unsupported batch content: type={type(x)}")

        else:
            print(f"{prefix} unsupported batch type: {type(batch)}")

        if B is not None and H is not None and W is not None:
            seq_len = infer_vit_seq_len(model=model, H=H, W=W, add_cls=True)
            D_this = int(B * seq_len)
            D_partial += D_this
            print(f"{prefix} seq_len_per_image ≈ {seq_len}  -> batch_tokens ≈ {D_this}")
        else:
            print(f"{prefix} cannot infer (B,H,W); batch_tokens += 0")

    print(f"{prefix} D_partial (first {n_batches} batches) ≈ {D_partial}")
    return D_partial

def main():
    parser = argparse.ArgumentParser(description="Train Classifier with Neuron Selection")

    # Dataset and model options
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name from HuggingFace (e.g., google/vit-base-patch16-224, roberta-base)")
    parser.add_argument("--dataset", "--task", type=str, required=True,
                        help="Dataset name (e.g., cifar10, eurosat, agnews)")
    parser.add_argument("--modality", type=str, default="image", choices=["image", "text"],
                        help="Data modality: image or text")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes in the dataset (if None, will be inferred)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for text models")
    parser.add_argument("--image_size", type=int, default=224,
                        help="The resolution to resize images to (e.g., 224, 384)")


    # Run mode
    parser.add_argument("--mode", type=str, default="stt",
                        choices=["stt", "stt_lora", "lora", "adalora", "loha", "lokr", "baseline","mag_pt", "mag_tp", "wanda_p", "transfer", "cross_task_plot",
                        "activation_mean_value", "activation_rate", "calibration","wanda_tp"],
                        help="Run mode: 'stt' for neuron selection, 'stt_lora' for STT+LoRA, 'baseline' for full finetuning, 'activation_mean_value' for activation magnitude ablation, 'activation_rate' for activation frequency ablation")

    # Neuron selection options
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Activation threshold for neuron selection")
    parser.add_argument("--use_abs_threshold", action="store_true",
                        help="Use absolute values for thresholding")
    parser.add_argument("--sample_ratio", type=float, default=0.005,
                        help="Ratio of training data to sample for finding active neurons")
    parser.add_argument("--disable_dropout", action="store_true",
                        help="Disable dropout in replaced MLPs after neuron selection")
    parser.add_argument("--topk_ratio",type=float, default=0.30, metavar="R",
                        help="Fraction (0,1] of neurons kept per score when selecting active neurons")

    # LoRA options
    parser.add_argument("--lora_r", type=int, default=32,
                        help="Rank of the LoRA updates")
    parser.add_argument("--lora_alpha", type=float, default=64,
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout rate for LoRA layers")

    # Training options
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Number of steps between evaluations")
    parser.add_argument("--max_grad_norm", type=float, default=0.01,
                        help="Maximum gradient norm for gradient clipping")

    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")


    # WandB options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="neuron-selective-classification",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="your-entity",
                        help="WandB entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional run name for wandb logging")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log to WandB every N steps")
    parser.add_argument('--recovery_lr', "--recovery learning_rate", type=float, default=1e-5, help='recovery Learning rate')

    parser.add_argument("--schedule", action="store_true", help="enable linear schedule")
    parser.add_argument("--tune_attn", action="store_true", help="tune attn proj")
    parser.add_argument("--dev_mode", action="store_true", help="Enable dev mode: use a subset of training data as pseudo-dev set for tuning")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Fraction of train set to use as dev set when dev_mode is on")
    parser.add_argument("--mag_norm", type=str, default="l1", choices=["l1"],  
                        help="Per-output-channel norm used for magnitude pruning")
    parser.add_argument("--mag_use_tracker_budget", action="store_true",
                        help="Use tracker's active-set to determine per-layer k (budget). If false, fallback to topk_ratio")
    parser.add_argument("--source_task", "--source_dataset", type=str, default=None,
                        help="Source dataset used ONLY for subnetwork selection (e.g., eurosat). If None, defaults to --dataset.")
    parser.add_argument("--source_ratio", type=float, default=None,
                        help="Num to keep for source network")

    args = parser.parse_args()

    # For backwards compatibility
    args.lr = args.learning_rate
    args.wd = args.weight_decay
    print(f"[DEBUG] args.topk_ratio = {args.topk_ratio}")   # 应显示 0.45
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup wandb logging
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model_name.split('/')[-1]}_{args.dataset}_{args.mode}_{timestamp}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    print(f"Run name: {run_name}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    print(f"Loading dataset: {args.dataset}")

    # Create a temporary config dict for the dataloader
    temp_config = {
        "model": {"name": args.model_name},
        "dataset": {
            "name": args.dataset,
            "modality": args.modality,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "image_size": args.image_size 
        }
    }
    model_name = args.model_name
    # num_classes, train_dataloader, eval_dataloader = get_dataloader(temp_config)
    num_classes, train_dataloader, val_dataloader, eval_dataloader, non_shuffle_train_dataloader  = get_dataloader(
        temp_config,
        dev_mode=args.dev_mode,
        dev_ratio=args.dev_ratio
    )
    if args.dev_mode:
        print("[DEV MODE] Using validation set as pseudo-test set")
        eval_dataloader = val_dataloader

    if args.modality == "image":
        if "clip" in model_name.lower() or "siglip" in model_name.lower():
            if "clip" in model_name.lower():
                try:
                    vision_encoder = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
                except:
                    full_model = CLIPModel.from_pretrained(model_name)
                    full_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            else:
                try:
                    vision_encoder = SiglipVisionModel.from_pretrained(model_name)
                except:
                    full_model = SiglipModel.from_pretrained(model_name)
                    vision_encoder = full_model.vision_model
            model = VisionEncoderWithClassifier(vision_encoder, num_classes)
        else:
            model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=num_classes,
                                                                    ignore_mismatched_sizes=True)
    else:  # BERT, RoBERTa, etc
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes,
                                                                   ignore_mismatched_sizes=True)
        tokenizer = None
        if args.modality == "text":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
            print(f"[FLOPs] Assuming max token length = {TEXT_MAXLEN} for FLOPs estimation.")

    model = model.to(device).to(torch.bfloat16)
    if args.modality == "image":
        _ = peek_vit_dataloader(train_dataloader, model=model, n_batches=2, prefix="[VIT-TRAIN]")
        _ = peek_vit_dataloader(eval_dataloader, model=model, n_batches=2, prefix="[VIT-TEST]")
        F_fwd_base      = None  
        F_fwd_variant   = None  
        F_train_variant = None  
        selection_flops = 0     

        _batch0 = next(iter(train_dataloader))
        if isinstance(_batch0, (list, tuple)) and len(_batch0) >= 1 and torch.is_tensor(_batch0[0]):
            _x1 = _batch0[0][:1].to(device).to(torch.bfloat16)  # (1,3,H,W)
            _y1 = _batch0[1][:1].to(device) if (len(_batch0) > 1 and torch.is_tensor(_batch0[1]) and _batch0[1].dim() == 1) else None
        elif isinstance(_batch0, dict) and "pixel_values" in _batch0 and torch.is_tensor(_batch0["pixel_values"]):
            _x1 = _batch0["pixel_values"][:1].to(device).to(torch.bfloat16)
            _y1 = _batch0["labels"][:1].to(device) if ("labels" in _batch0 and torch.is_tensor(_batch0["labels"])) else None
        else:
            raise RuntimeError("[FLOPs] Unsupported image batch structure.")
        FLOPS_DEBUG = True
        _flops_dbg_once = {"done": False}

        try:
            _ = model(pixel_values=_x1)  
            _inp1_fwd   = {"pixel_values": _x1}  
            _inp1_train = {"pixel_values": _x1}
            if FLOPS_DEBUG and not _flops_dbg_once["done"]:
                print(f"[FLOPs][debug] call path = keyword; x1.shape={tuple(_x1.shape)}, dtype={_x1.dtype}, device={_x1.device}")
        except TypeError:
            _inp1_fwd   = _x1
            _inp1_train = _x1
            if FLOPS_DEBUG and not _flops_dbg_once["done"]:
                print(f"[FLOPs][debug] call path = positional; x1.shape={tuple(_x1.shape)}, dtype={_x1.dtype}, device={_x1.device}")

        import torch.nn.functional as F
        def _vit_criterion(out, _inputs_unused):
            def _pick_tensor(x):
                if isinstance(x, dict):
                    for k in ("logits", "logit", "scores", "last_hidden_state", "pooler_output"):
                        v = x.get(k, None)
                        if torch.is_tensor(v):
                            return v, k
                    for k, v in x.items():
                        if torch.is_tensor(v):
                            return v, k
                if hasattr(x, "logits") and torch.is_tensor(getattr(x, "logits")):
                    return getattr(x, "logits"), "logits(attr)"
                if hasattr(x, "__dict__"):
                    for k in ("scores", "last_hidden_state", "pooler_output"):
                        if hasattr(x, k) and torch.is_tensor(getattr(x, k)):
                            return getattr(x, k), f"{k}(attr)"
                if isinstance(x, (list, tuple)):
                    for i, e in enumerate(x):
                        if torch.is_tensor(e):
                            return e, f"seq[{i}]"
                if torch.is_tensor(x):
                    return x, "tensor"
                return None, "none"

            t, tag = _pick_tensor(out)
            if t is None:
                raise RuntimeError(f"[FLOPs] cannot extract tensor from model output; type(out)={type(out)}")
            if FLOPS_DEBUG and not _flops_dbg_once["done"]:
                lab = ( _y1 is not None and torch.is_tensor(_y1) )
                print(f"[FLOPs][debug] out.type={type(out).__name__}; pick='{tag}', shape={tuple(t.shape)}, dtype={t.dtype}; labels={lab}")
                _criterion = _vit_criterion
                _flops_dbg_once["done"] = True

            if (_y1 is not None and torch.is_tensor(_y1)
                and t.dim() >= 2 and t.size(0) == _y1.size(0)):
                return F.cross_entropy(t, _y1)
            return t.float().mean()
            # ---- Baseline（NS/LoRA/NS-LoRA）----
        _criterion = _vit_criterion
        F_fwd_base   = flops_forward(model, _inp1_fwd, device=str(device))
        print(f"[FLOPs] per-image forward (baseline): {F_fwd_base/1e9:.3f} GFLOPs")

    elif args.modality == "text":
        # ===== [FLOPs] Prepare single-sample & baseline counts (BERT/SeqCls) =====
        F_fwd_base      = None   # baseline per-example forward FLOPs (L=TEXT_MAXLEN)
        F_fwd_variant   = None   # variant per-example forward FLOPs
        F_train_variant = None   # variant per-example train FLOPs
        selection_flops = 0
        # Assume fixed text length TEXT_MAXLEN(=512): build 1×L dummy inputs (no labels)
        L = TEXT_MAXLEN
        _inp1_fwd = {
            "input_ids": torch.ones(1, L, dtype=torch.long, device=str(device)),
           "attention_mask": torch.ones(1, L, dtype=torch.long, device=str(device)),
        }
        _inp1_train = dict(_inp1_fwd)  # training FLOPs uses mean() loss without labels
        _inp1_extracted = _inp1_fwd
        _lab = _inp1_extracted.get("labels", None)
        if _lab is None:
            _norm_lab = torch.zeros(1, dtype=torch.long, device=str(device))
        elif torch.is_tensor(_lab):
            _lab = _lab.to(device)
            if _lab.dim() == 0:
               norm_lab = _lab.view(1).to(torch.long)
            elif _lab.dim() == 1:
               _norm_lab = _lab[:1].to(torch.long)
            else:
              norm_lab = _lab[:1, 0].to(torch.long)
        else:
            _norm_lab = torch.zeros(1, dtype=torch.long, device=str(device))
            _inp1_train = dict(_inp1_fwd); _inp1_train["loss_labels"] = _norm_lab

        
        print(f"[DEBUG] Forward input keys (no labels): {list(_inp1_fwd.keys())}")
        print(f"[DEBUG] Training input keys (with labels): {list(_inp1_train.keys())}")

        # Simple criterion mirroring the ViT path: use CE when shapes match, else mean()
        import torch.nn.functional as F
        FLOPS_DEBUG = True
        _flops_dbg_once = {"done": False}
        def _bert_criterion(out, inputs_dict):
            def _pick_tensor(x):
                if isinstance(x, dict):
                    for k in ("logits","logit","scores","last_hidden_state","pooler_output"):
                        v = x.get(k, None)
                        if torch.is_tensor(v): return v, k
                    for k, v in x.items():
                        if torch.is_tensor(v): return v, k
                if hasattr(x, "logits") and torch.is_tensor(getattr(x, "logits")):
                    return getattr(x, "logits"), "logits(attr)"
                if hasattr(x, "__dict__"):
                    for k in ("scores","last_hidden_state","pooler_output"):
                        if hasattr(x, k) and torch.is_tensor(getattr(x, k)):
                            return getattr(x, k), f"{k}(attr)"
                if isinstance(x, (list, tuple)):
                    for i, e in enumerate(x):
                        if torch.is_tensor(e): return e, f"seq[{i}]"
                if torch.is_tensor(x): return x, "tensor"
                return None, "none"
            
            t, tag = _pick_tensor(out)
            if t is None:
                raise RuntimeError(f"[FLOPs] cannot extract tensor from model output; type(out)={type(out)}")
            if FLOPS_DEBUG and not _flops_dbg_once["done"]:
                print(f"[FLOPs][debug][BERT] out.pick='{tag}', shape={tuple(t.shape)} (TEXT_MAXLEN={TEXT_MAXLEN})")
                _flops_dbg_once["done"] = True
                y = inputs_dict.get("loss_labels", None)
                return t.float().mean()

        # Baseline forward FLOPs (used for selection accounting in NS/NS-LoRA)
        print(f"[DEBUG] About to call flops_forward with inputs: {list(_inp1_fwd.keys())}")
        F_fwd_base = flops_forward(model, _inp1_fwd, device=str(device))
        print(f"[FLOPs][BERT] per-example forward baseline (L={TEXT_MAXLEN}): {F_fwd_base/1e9:.3f} GFLOPs")
        _criterion = _bert_criterion
    # Calculate original parameter count
    orig_param_count = sum(p.numel() for p in model.parameters())
    orig_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print model layers for debugging
    print_model_layers(model)
    print("Model loaded successfully")
    if args.mode == "calibration":
        print("--- [MODE: ANALYSIS] Running Sub-network Stability Analysis for ViT ---")
            
        ratios_to_test = [0.01, 0.02, 0.05, 0.1]
        results_by_ratio = {} 
    
        print(f"[*] Testing active set stability for ratios: {ratios_to_test}")

        for ratio in ratios_to_test:
            print(f"\n----- Processing ratio: {ratio} -----")
                
            active_dataloader = sample_active_set(
                    non_shuffle_train_dataloader,
                    ratio=ratio
            )
                
            tracker = NeuronTracker(
                    model=model,
                    tokenizer=None, 
                    threshold=args.threshold, 
                    topk_ratio=args.topk_ratio, 
                    use_abs_threshold=args.use_abs_threshold,
                    device=device,
                    track_attention_proj=args.tune_attn,
                    verbose=False 
            )
                
            active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
            if active_neurons:
                print(f"[*] Found active indices for {len(active_neurons)} layers.")
                results_by_ratio[ratio] = active_neurons
            else:
                print("[!] Warning: No active neurons found for this ratio.")
            
        print("\n[*] Calculating similarity matrix for the heatmap...")
        tested_ratios = sorted(results_by_ratio.keys())
        num_ratios = len(tested_ratios)

        if num_ratios == 0:
            print("[!] No active-neuron results collected. Skip heatmap.")
            return

        similarity_matrix = np.eye(num_ratios, dtype=float)
        for i in range(num_ratios):
            for j in range(i + 1, num_ratios):
                r1, r2 = tested_ratios[i], tested_ratios[j]
                sim = calculate_jaccard_similarity(results_by_ratio[r1], results_by_ratio[r2])
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim

        print("[*] Generating and saving heatmap...")

        N = num_ratios
        fig_w = max(1.6 * N, 6.0)
        fig_h = fig_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        tick_fs = max(8, int(14 - 0.3 * N))
        ann_fs  = max(8, int(12 - 0.3 * N))

        hm = sns.heatmap(
            similarity_matrix,
            annot=True, fmt=".3f",
            annot_kws={"fontsize": ann_fs, "fontweight": "bold"},
            xticklabels=tested_ratios,
            yticklabels=tested_ratios,
            cmap="viridis",
            vmin=0.5, vmax=1.0,
            square=True,
            linewidths=1.0, linecolor="white",
            cbar=True,
            cbar_kws=dict(fraction=0.035, pad=0.02, shrink=0.9, aspect=40, ticks=[0.5, 0.75, 1.0]),
            ax=ax
        )

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=tick_fs)
        ax.tick_params(axis="x", labelrotation=0, labelsize=tick_fs, length=6, width=1.2)
        ax.tick_params(axis="y", labelsize=tick_fs, length=6, width=1.2)

        ax.set_xlabel("Calibration Set Sample Ratio", fontsize=tick_fs + 2)
        ax.set_ylabel("Calibration Set Sample Ratio", fontsize=tick_fs + 2)

        plt.tight_layout()

        base = f"stability_heatmap_{args.model_name.replace('/', '_')}_{args.dataset}"
        plt.savefig(f"{base}.pdf", bbox_inches="tight")   
        plt.savefig(f"{base}.svg", bbox_inches="tight")   
        print(f"[*] Heatmap saved to: {base}.pdf / {base}.svg")

        return
    # ---- NS analysis: cross-task ONLY (hardcoded rho per task) ----
    elif args.mode == "cross_task_plot":
        print("--- [MODE: ANALYSIS] NS Layerwise Cross-Task (Hardcoded rhos) ---")
        from collections import OrderedDict

        print("--- [MODE: ANALYSIS] NS Layerwise Cross-Task (Hardcoded rhos; dataloaders via get_dataloader) ---")

        TASK_SPECS = OrderedDict({
            "eurosat":      {"rho": 0.3},
            "gtsrb":        {"rho": 0.3},
            "clevr_count":  {"rho": 0.5}, 
        })

        ratio = getattr(args, "analysis_ratio", 0.10)
        print(f"[*] Tasks & hardcoded rhos: {{"
            + ", ".join([f"{k}: {v['rho']}" for k, v in TASK_SPECS.items()])
            + f"}}, calibration ratio={ratio}")

        task_dataloaders = OrderedDict()
        for ds_name, spec in TASK_SPECS.items():
            temp_config = {
                "model": {"name": args.model_name},
                "dataset": {
                    "name": ds_name,               
                    "modality": args.modality,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "image_size": args.image_size
                }
            }
            try:
                (num_classes,
                train_dataloader,
                val_dataloader,
                eval_dataloader,
                non_shuffle_train_dataloader) = get_dataloader(
                    temp_config,
                    dev_mode=args.dev_mode,
                    dev_ratio=args.dev_ratio
                )
            except Exception as e:
                raise RuntimeError(f"[dataloader] Failed to load dataset '{ds_name}': {e}")

            if non_shuffle_train_dataloader is None:
                raise RuntimeError(f"[dataloader] non_shuffle_train_dataloader is None for '{ds_name}'")

            task_dataloaders[ds_name] = {
                "non_shuffle_train": non_shuffle_train_dataloader,
                "rho": float(spec["rho"])
            }

        results_by_task = OrderedDict()
        for task, cfg in task_dataloaders.items():
            print(f"\n----- Processing task: {task} (rho={cfg['rho']}) -----")
            active_dataloader = sample_active_set(cfg["non_shuffle_train"], ratio=ratio)

            tracker = NeuronTracker(
                model=model,
                tokenizer=None,
                threshold=args.threshold,
                topk_ratio=cfg["rho"],               
                use_abs_threshold=args.use_abs_threshold,
                device=device,
                track_attention_proj=args.tune_attn,
                verbose=False
            )
            active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
            if active_neurons:
                print(f"[*] [{task}] Found active indices for {len(active_neurons)} layers.")
                results_by_task[task] = active_neurons
            else:
                print(f"[!] [{task}] No active neurons found. Using empty selections.")
                results_by_task[task] = {}

        task_names = list(results_by_task.keys())
        N = len(task_names)
        if N < 2:
            raise RuntimeError("[cross-task] Need at least 2 tasks to compare.")

        # ========= 计算 Cov(A|B)（有向覆盖）与 Jaccard =========
        print("\n[*] Calculating matrices (Directed Coverage & Jaccard)...")
        cov_matrix  = np.zeros((N, N), dtype=float)  # 行A=target, 列B=donor
        jacc_matrix = np.zeros((N, N), dtype=float)

        for i, A in enumerate(task_names):
            for j, B in enumerate(task_names):
                cov_matrix[i, j]  = calculate_directed_coverage(results_by_task[A], results_by_task[B], weighted=True)
                jacc_matrix[i, j] = calculate_jaccard_similarity(results_by_task[A], results_by_task[B])

        model_tag = args.model_name.replace("/", "_")
        base_tag  = f"cross_task_min_{model_tag}"

        # 刻度：下划线->换行并 Title Case，避免拥挤
        tick_labels = [t.replace("_", "\n").title() for t in task_names]
        N = len(task_names)

        def _plot_matrix(mat, out_prefix, vmin=0.0, vmax=1.0, fmt=".3f"):
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib import colors, cm

            mat = np.asarray(mat, dtype=float)

            # 尺寸与字号自适应
            fig_w = max(1.2 * N, 3.8)
            fig_h = max(1.2 * N, 3.8)
            tick_fs = max(8, int(14 - 0.3 * N))
            ann_fs  = max(8, int(12 - 0.3 * N))

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            # 细窄色条：fraction 越小越细；shrink 控制高度；ticks 固定为 0/0.5/1
            hm = sns.heatmap(
                mat,
                annot=False,
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                linewidths=1.0, linecolor="white",
                square=True,
                cbar=True,
                cbar_kws=dict(fraction=0.03, pad=0.02, shrink=0.9, aspect=40, ticks=[0.0, 0.5, 1.0]),
                ax=ax
            )
            cbar = hm.collections[0].colorbar
            cbar.ax.tick_params(labelsize=tick_fs)

            # 轴刻度与 tick marks（sticks）
            ax.tick_params(axis="x", labelrotation=0, pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)
            ax.tick_params(axis="y", pad=6, length=6, width=1.1, direction="out", labelsize=tick_fs)

            # 轴标签（最简）
            ax.set_xlabel("Source task", fontsize=max(10, tick_fs))
            ax.set_ylabel("Target task", fontsize=max(10, tick_fs))

            # 去边框 + 细白网格
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-.5, N, 1), minor=True)
            ax.set_yticks(np.arange(-.5, N, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)

            # 数字严格居中到每个 block（j+0.5, i+0.5），并按背景亮度自动黑/白
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")
            for i in range(N):
                for j in range(N):
                    val = mat[i, j]
                    rgba = cmap(norm(val))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    txt_color = "white" if lum < 0.5 else "black"
                    ax.text(j + 0.5, i + 0.5, format(val, fmt),
                            ha="center", va="center", fontsize=ann_fs, color=txt_color, clip_on=True, zorder=3)

            # 锁定坐标边界，确保文本与方块完美对齐
            ax.set_xlim(0, N)
            ax.set_ylim(N, 0)

            plt.tight_layout()
            png = f"{out_prefix}.png"
            pdf = f"{out_prefix}.pdf"
            plt.savefig(png, dpi=300, bbox_inches="tight")
            plt.savefig(pdf, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[*] Saved: {png} and {pdf}")

        _plot_matrix(cov_matrix, out_prefix=f"cov_heatmap_{base_tag}", vmin=0.0, vmax=1.0, fmt=".3f")
        _plot_matrix(jacc_matrix, out_prefix=f"jacc_heatmap_{base_tag}", vmin=0.0, vmax=1.0, fmt=".3f")

        print("\n[INFO] Cross-task (hardcoded rhos; minimal text + slim colorbar) analysis complete. Exiting script.")
        return  
    # Create active dataset for neuron selection if needed
    elif args.mode == "activation_mean_value":
        print("[ABLT] one-shot, sparsity-based selection; budget-matched to NS per layer")
        sel_loader = sample_active_set(non_shuffle_train_dataloader, float(args.sample_ratio))
        
        # First run NS tracker (tracker6) to get K-per-layer budget
        tracker_budget = NeuronTracker(
            model, threshold=0.01, topk_ratio=args.topk_ratio,
            device=device, track_attention_proj=False, verbose=False
        )
        active = tracker_budget.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}

        # Now use AblationNeuronTracker (ablation_tracker) with ONLY sparsity metric
        print("[ABLT] Using ablation_tracker with ONLY sparsity metric (fraction of samples > threshold)")
        tracker_a = AblationNeuronTracker(model, topk_ratio=args.topk_ratio, device=device, verbose=False)  
        # Use activation-rate-based selection (sparsity) with use_activation_rate=True
        sparsity_indices_all = tracker_a.get_active_indices(dataloader=sel_loader, use_activation_rate=True)

        def _is_fc1(lname: str) -> bool:
            return any(k in lname for k in ["gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense"])

        sel_indices = {}
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = layer_name_map.get(m, None)
            if lname is None or not _is_fc1(lname):
                continue
            idx = sparsity_indices_all.get(lname, None)
            if idx is None or len(idx) == 0:
                continue

            k = int(k_map.get(lname, 0))     
            if k <= 0:
                continue

            # Trim to match budget
            keep = min(k, len(idx))
            idx_t = torch.as_tensor(idx[:keep], dtype=torch.long)
            sel_indices[lname] = idx_t


        nst = NeuroselectiveTransformer5(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            tune_pruned=False, device=device, verbose=True
        )
        model = nst.transform().to(device).to(torch.bfloat16)
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable output layers parameters (classifier, etc.)
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)

        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward ({args.mode}): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   ({args.mode}): {F_train_variant/1e9:.3f} GFLOPs")

    elif args.mode == "activation_rate":
        print("[ABLT2] one-shot, activation-rate-based selection; budget-matched to NS per layer")
        sel_loader = sample_active_set(non_shuffle_train_dataloader, float(args.sample_ratio))
        
        # First run NS tracker (tracker6) to get K-per-layer budget
        tracker_budget = NeuronTracker(
            model, threshold=0.01, topk_ratio=args.topk_ratio,
            device=device, track_attention_proj=False, verbose=False
        )
        active = tracker_budget.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}

        # Now use AblationNeuronTracker (ablation_tracker) with activation rate metric
        print("[ABLT2] Using ablation_tracker with activation rate metric (fraction of samples > threshold)")
        tracker_a = AblationNeuronTracker(model, topk_ratio=args.topk_ratio, device=device, verbose=False)  
        # Use activation-rate-based selection (sparsity) with use_activation_rate=True
        sparsity_indices_all = tracker_a.get_active_indices(dataloader=sel_loader, use_activation_rate=True)

        def _is_fc1(lname: str) -> bool:
            return any(k in lname for k in ["gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense"])

        sel_indices = {}
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = layer_name_map.get(m, None)
            if lname is None or not _is_fc1(lname):
                continue
            idx = sparsity_indices_all.get(lname, None)
            if idx is None or len(idx) == 0:
                continue

            k = int(k_map.get(lname, 0))     
            if k <= 0:
                continue

            # Trim to match budget
            keep = min(k, len(idx))
            idx_t = torch.as_tensor(idx[:keep], dtype=torch.long)
            sel_indices[lname] = idx_t


        nst = NeuroselectiveTransformer5(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            tune_pruned=False, device=device, verbose=True
        )
        model = nst.transform().to(device).to(torch.bfloat16)
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable output layers parameters (classifier, etc.)
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)

        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward ({args.mode}): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   ({args.mode}): {F_train_variant/1e9:.3f} GFLOPs")

    elif args.mode == "transfer":
        temp_config = {
            "model": {"name": args.model_name},
            "dataset": {
                 "name": args.source_task,
                "modality": args.modality,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "image_size": args.image_size
            }
        }
        _, source_train_dl, _, _, source_nonshuffle_dl = get_dataloader(temp_config)

        source_active_dl = sample_active_set(source_nonshuffle_dl, ratio=args.sample_ratio)

        source_tracker = NeuronTracker(
            model=model,
            tokenizer=None,
            threshold=args.threshold,
            topk_ratio=args.source_ratio,
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )
        source_active_neurons = source_tracker.get_active_indices(dataloader=source_active_dl)
        source_layer_name_map = source_tracker.get_layer_name_map()

        nstransformer = NeuroselectiveTransformer5(
            model=model,  # Use the fresh or original model instance
            active_neurons=source_active_neurons,  
            layer_name_map=source_layer_name_map, 
            verbose=True,
            tune_pruned=False, 
            device=device  
        )

        model = nstransformer.transform().to(device).to(torch.bfloat16)
        stats = nstransformer.get_parameter_stats()         
        print(f"[ns] overall_reduction = "
            f"{stats['overall_model_reduction_perc']:.2f}%")

        if hasattr(nstransformer, "parameter_stats"):
            stats = nstransformer.parameter_stats
            if "overall_reduction" in stats:
                print(f"Parameter reduction: {stats['overall_reduction']:.2%}")

        print(f"Model transformed with neuron selection")

        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable output layers parameters (classifier, etc.)
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
            # if any(key in name for key in ["classifier"]):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Create optimizer with only trainable parameters
        optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        F_fwd_variant   = flops_forward(model, _inp1_fwd,   device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (NS): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (NS): {F_train_variant/1e9:.3f} GFLOPs")
    elif args.mode == "stt":
        print(f"Creating active dataset with sample ratio: {args.sample_ratio}")
        active_dataloader = sample_active_set(
                non_shuffle_train_dataloader,
                ratio=args.sample_ratio
        )
        try:
            sel_samples = len(active_dataloader.dataset)
        except Exception:
            sel_samples = int(args.sample_ratio * len(train_dataloader.dataset))
        F_fwd_base = flops_forward(model, _inp1_fwd, device=str(device))
        selection_flops = int(F_fwd_base) * int(sel_samples)
        print(f"[FLOPs] selection (NS; baseline fwd  {sel_samples} images): {selection_flops/1e9:.2f} GFLOPs")

        # Check if active_dataloader is empty
        if sel_samples == 0:
            raise ValueError(f"Active dataloader is empty! sample_ratio={args.sample_ratio} resulted in 0 samples. "
                           f"Please increase sample_ratio or check dataset size.")

        first_batch = next(iter(active_dataloader))
        print("Doing NS now, using trakcer-6")
        print("\n=== Debug: first batch ===")
        print("type :", type(first_batch))

        if isinstance(first_batch, dict):
            for k, v in first_batch.items():
                print(f"  {k:<15} → {type(v)}  shape={tuple(v.shape) if torch.is_tensor(v) else 'N/A'}")
        elif isinstance(first_batch, (list, tuple)):
            for idx, item in enumerate(first_batch):
                if torch.is_tensor(item):
                    print(f"  [{idx}] Tensor shape={tuple(item.shape)}")
                else:
                    print(f"  [{idx}] {type(item)}")
        else:
            print("  Batch Content:", first_batch[:3], "...")
        print("=" * 30)

        print("\n--- Performing neuron selection ---")

        tracker = NeuronTracker(
                        model=model,
                        tokenizer=None,                        
                        threshold=args.threshold,                  
                        topk_ratio=args.topk_ratio, 
                        use_abs_threshold=args.use_abs_threshold,
                        device=device,
                        track_attention_proj=args.tune_attn,
                        verbose=True
                    )

        active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker.get_layer_name_map()
        # print(f"Active neurons selected: {len(active_neurons)}")
        model_clean = args.model_name.replace('/', '_').replace('-', '_')
        dataset_clean = args.dataset.replace('/', '_').replace('-', '_')
        
        print("\n--- Generating dual metric composition visualizations ---")
        composition_stats = tracker.compute_dual_metric_composition(delta=args.threshold)

        # Extract model and dataset names for clean filenames
        model_clean = args.model_name.replace('/', '_').replace('-', '_')
        dataset_clean = args.dataset.replace('/', '_').replace('-', '_')

        tracker.visualize_layer_composition(
            composition_stats, 
            model_name=f"{model_clean}_topk{args.topk_ratio}", 
            dataset_name=dataset_clean
        )

        non_empty_layers = {k: v for k, v in active_neurons.items() if v.numel() > 0}
        print(f"[tracker] non-empty layers = {len(non_empty_layers)} / {len(active_neurons)}")
        if not non_empty_layers:
            raise RuntimeError("Tracker failed.")

        nstransformer = NeuroselectiveTransformer5(
            model=model,  # Use the fresh or original model instance
            active_neurons=active_neurons,  
            layer_name_map=layer_name_map, 
            verbose=True,
            tune_pruned=False, 
            device=device  
        )

        model = nstransformer.transform().to(device).to(torch.bfloat16)
        stats = nstransformer.get_parameter_stats()         
        print(f"[ns] overall_reduction = "
            f"{stats['overall_model_reduction_perc']:.2f}%")

        if hasattr(nstransformer, "parameter_stats"):
            stats = nstransformer.parameter_stats
            if "overall_reduction" in stats:
                print(f"Parameter reduction: {stats['overall_reduction']:.2%}")

        print(f"Model transformed with neuron selection")

        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable output layers parameters (classifier, etc.)
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
            # if any(key in name for key in ["classifier"]):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Create optimizer with only trainable parameters
        optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        F_fwd_variant   = flops_forward(model, _inp1_fwd,   device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (NS): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (NS): {F_train_variant/1e9:.3f} GFLOPs")
    elif args.mode == "baseline":
        print("Running baseline: full fine-tuning (all parameters trainable)")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        F_fwd_variant   = flops_forward(model, _inp1_fwd,   device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (Baseline): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (Baseline): {F_train_variant/1e9:.3f} GFLOPs")

        # Recompute final counts
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif args.mode == "stt_lora":
        print(f"Creating active dataset with sample ratio: {args.sample_ratio}")
        active_dataloader = sample_active_set(
            non_shuffle_train_dataloader,
            ratio=args.sample_ratio
        )
        try:
            sel_samples = len(active_dataloader.dataset)
        except Exception:
            sel_samples = int(args.sample_ratio * len(train_dataloader.dataset))
        F_fwd_base = flops_forward(model, _inp1_fwd, device=str(device))
        selection_flops = int(F_fwd_base) * int(sel_samples)
        print(f"[FLOPs] selection (NS; baseline fwd  {sel_samples} images): {selection_flops/1e9:.2f} GFLOPs")

        # Check if active_dataloader is empty
        if sel_samples == 0:
            raise ValueError(f"Active dataloader is empty! sample_ratio={args.sample_ratio} resulted in 0 samples. "
                           f"Please increase sample_ratio or check dataset size.")

        first_batch = next(iter(active_dataloader))
        print("Doing NSLora now, using trakcer-6")
        print("\n=== Debug: first batch ===")
        print("type :", type(first_batch))

        if isinstance(first_batch, dict):
            for k, v in first_batch.items():
                print(f"  {k:<15} → {type(v)}  shape={tuple(v.shape) if torch.is_tensor(v) else 'N/A'}")
        elif isinstance(first_batch, (list, tuple)):
            for idx, item in enumerate(first_batch):
                if torch.is_tensor(item):
                    print(f"  [{idx}] Tensor shape={tuple(item.shape)}")
                else:
                    print(f"  [{idx}] {type(item)}")
        else:
            print("  Batch Content:", first_batch[:3], "...")
        print("=" * 30)

        print("\n--- Performing neuron selection ---")
        tracker = NeuronTracker(
            model=model,
            tokenizer=tokenizer if args.modality == "text" else None,  # Only use tokenizer for text models
            threshold=args.threshold,                   # == active_threshold
            topk_ratio=args.topk_ratio,             
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker.get_layer_name_map()

        # print(f"Active neurons selected: {len(active_neurons)}")
        non_empty_layers = {k: v for k, v in active_neurons.items() if v.numel() > 0}
        print(f"[tracker] non-empty layers = {len(non_empty_layers)} / {len(active_neurons)}")
        if not non_empty_layers:
            raise RuntimeError("Tracker failed.")

        nstransformer = NeuroselectiveTransformer5(
            model=model,  # Use the fresh or original model instance
            active_neurons=active_neurons,  # Corrected: Use the dict from tracker
            layer_name_map=layer_name_map,  # Correct: Pass the map
            verbose=True,
            tune_pruned=False,  # Set True to allow fine-tuning pruned parts
            device=device  # Pass device if ns_transformer needs it explicitly
        )

        model = nstransformer.transform().to(device).to(torch.bfloat16)
        # Only apply NSLoraLinear to pruned NeuroselectiveLinear layers
        print("Pruning complete")

        # 3) Wrap each NeuroselectiveLinear in NSLoraLinear
        from stt.stt_lora import NSLoraLinear
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                parent_path, attr = name.rsplit(".", 1)
                parent = model
                for p in parent_path.split("."):
                    parent = getattr(parent, p)
                ns_lora = NSLoraLinear(
                    stt_linear=module,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    merge_weights=False
                )
                ns_lora = ns_lora.to(device).to(torch.bfloat16)

                setattr(parent, attr, ns_lora)
                print(f"  Wrapped {name}")

        # 4) Freeze everything except adapter weights and classifier head
        for param in model.parameters():
            param.requires_grad = False
        for nm, mod in model.named_modules():
            if isinstance(mod, NSLoraLinear):
                mod.lora_A.weight.requires_grad = True
                mod.lora_B.weight.requires_grad = True
            if nm.endswith("classifier"):
                for pname, p in mod.named_parameters(recurse=False):
                    p.requires_grad = True
                    print(f"  Unfroze classifier.{pname}")

        # 5) Build optimizer on trainable parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {len(trainable)}")
        optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        # Update parameter stats
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        F_fwd_variant   = flops_forward(model, _inp1_fwd,   device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)        
        print(f"[FLOPs] per-image train   (Baseline): {F_train_variant/1e9:.3f} GFLOPs")
    elif args.mode == "mag_pt":
        sel_loader = sample_active_set(non_shuffle_train_dataloader, float(args.sample_ratio))
        tracker = NeuronTracker(model, threshold=0.01, topk_ratio=args.topk_ratio,
                                device=device, track_attention_proj=False, verbose=False)
        active = tracker.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}  
        mag_indices = {}
        for m in model.modules():
            if isinstance(m, nn.Linear):
                lname = layer_name_map.get(m, None)
                if lname is None or not any(k in lname for k in ["gate_proj", "fc1", "lin1", "wi", "mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense",]):
                    continue
                k = k_map.get(lname, 0)
                if k <= 0: continue
                score = m.weight.abs().sum(dim=1)   
                topk = torch.topk(score, min(k, score.numel())).indices
                mag_indices[lname] = topk
        nst = NeuroselectiveTransformer5(model, active_neurons=mag_indices,
                                        layer_name_map=layer_name_map,
                                        tune_pruned=False, device=device, verbose=True)
        model = nst.transform().to(device).to(torch.bfloat16)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Set up optimizer (same as baseline, all trainable params)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)
        # Optional: FLOPs counting like other modes
        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward ({args.mode}): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   ({args.mode}): {F_train_variant/1e9:.3f} GFLOPs") 
    elif args.mode == "wanda_p":  
        print("[WANDA] one-shot, input-side (L1), per-layer top-k")

        sel_loader = sample_active_set(non_shuffle_train_dataloader, float(args.sample_ratio))
        tracker_budget = NeuronTracker(
            model, threshold=0.01, topk_ratio=args.topk_ratio,
            device=device, track_attention_proj=False, verbose=False
        )
        active = tracker_budget.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}

        tracker_w = NeuronTracker(model, topk_ratio=1.0, device=device, verbose=False)  
        wanda_indices_all = tracker_w.get_wanda_indices(
            dataloader=sel_loader,
            scan_batches=int(getattr(args, "wanda_calib_batches", 4))
        )

        def _is_fc1(lname: str) -> bool:
            return any(k in lname for k in ["gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense"])

        sel_indices = {}
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = layer_name_map.get(m, None)
            if lname is None or not _is_fc1(lname):
                continue
            idx = wanda_indices_all.get(lname, None)
            if idx is None or len(idx) == 0:
                continue

            k = int(k_map.get(lname, 0))     # 仅使用预算
            if k <= 0:
                continue

            # 预算截断（此时 idx 长度应≥k；若个别层仍短，会被安全裁剪为 idx 长度）
            idx_t = torch.as_tensor(idx[:k], dtype=torch.long)
            sel_indices[lname] = idx_t


        nst = NeuroselectiveTransformer5(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            tune_pruned=False, device=device, verbose=True
        )
        model = nst.transform().to(device).to(torch.bfloat16)
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable output layers parameters (classifier, etc.)
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    trainable.append(param)
                print(f"new trainable: {name}")

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)

        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward ({args.mode}): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   ({args.mode}): {F_train_variant/1e9:.3f} GFLOPs")
    elif args.mode == "mag_tp":
        sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=args.sample_ratio)
        tracker = NeuronTracker(model, threshold=0.01, topk_ratio=args.topk_ratio,
                                device=device, track_attention_proj=False, verbose=False)
        active = tracker.get_active_indices(dataloader=sel_loader)   # {lname: LongTensor}
        layer_name_map = tracker.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}         # {lname: k}
        MAG_TP_K_MAP = k_map
        MAG_TP_LAYER_NAME_MAP = layer_name_map
        ex = list(k_map.items())[:3]
        print(f"[mag_tp][pre] captured budget for {len(k_map)} layers, e.g. {ex}")
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Set up optimizer (same as baseline, all trainable params)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)

        # Optional: FLOPs counting like other modes
        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (mag_tp-pre): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (mag_tp-pre): {F_train_variant/1e9:.3f} GFLOPs")
    elif args.mode == "wanda_tp":
        sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=args.sample_ratio)
        tracker = NeuronTracker(model, threshold=0.01, topk_ratio=args.topk_ratio,
                            device=device, track_attention_proj=False, verbose=False)
        active = tracker.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}
        MAG_TP_K_MAP = k_map
        MAG_TP_LAYER_NAME_MAP = layer_name_map
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=args.wd)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Set up optimizer (same as baseline, all trainable params)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)

        # Optional: FLOPs counting like other modes
        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (mag_tp-pre): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (mag_tp-pre): {F_train_variant/1e9:.3f} GFLOPs")


    else:
        # Apply LoRA
        lora_config = {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'bias': 'none',
            'type': args.mode,
            'task_type': 'SEQ_CLS',
        }
        model = setup_lora(model, lora_config)

        # Calculate params after LoRA
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd
        )
        F_fwd_variant   = flops_forward(model, _inp1_fwd,   device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward (Baseline): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   (Baseline): {F_train_variant/1e9:.3f} GFLOPs")

    # Print parameter statistics
    print("\nParameter Statistics:")
    print(f"Original model parameters: {orig_param_count:,}")
    print(f"Original trainable parameters: {orig_trainable_params:,}")
    print(f"Final model parameters: {final_param_count:,}")
    print(f"Final trainable parameters: {final_trainable_params:,}")
    print(f"Trainable parameter reduction: {(1 - final_trainable_params/orig_trainable_params)*100:.2f}%")

    if args.use_wandb:
        wandb.log({
            'orig_params': orig_param_count,
            'orig_trainable_params': orig_trainable_params,
            'final_params': final_param_count,
            'final_trainable_params': final_trainable_params,
            'param_reduction': (1 - final_trainable_params/orig_trainable_params)*100
        })

    print("\n--- Setting up training ---")

    # Track parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    if args.use_wandb:
        wandb.log({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_parameters_pct": trainable_params/total_params
        })

    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    train_size = len(train_dataloader.dataset)
    test_size  = len(eval_dataloader.dataset)
    epochs     = args.num_epochs

    training_flops  = int(F_train_variant) * int(train_size) * int(epochs)
    inference_flops = int(F_fwd_variant)  * int(test_size)
    total_flops     = int(selection_flops) + training_flops + inference_flops


    print("\n[FLOPs] ===== Complexity Summary (FLOPs) =====")
    print(f"Selection : {selection_flops/1e9:.2f} GFLOPs")   # NS/NS-LoRA > 0；Baseline/LoRA == 0
    print(f"Training  : {training_flops/1e9:.2f} GFLOPs")
    print(f"Inference : {inference_flops/1e9:.2f} GFLOPs")
    print(f"TOTAL     : {total_flops/1e9:.2f} GFLOPs")
    print(f"Per-image forward : {F_fwd_variant/1e9:.3f} GFLOPs")
    print(f"Per-image training: {F_train_variant/1e9:.3f} GFLOPs")

    if args.use_wandb:
        wandb.log({
            "flops/selection": selection_flops,
            "flops/training":  training_flops,
            "flops/inference": inference_flops,
            "flops/total":     total_flops,
            "flops/per_image_forward": F_fwd_variant,
            "flops/per_image_train":   F_train_variant,
        })

    loss_fn = nn.CrossEntropyLoss() if num_classes > 1 else nn.MSELoss()

    # Training loop
    print("\n--- Starting training ---")
    best_accuracy = 0
    global_step = 0

    if args.schedule:
        num_training_steps = len(train_dataloader) * args.num_epochs
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    global_train_start = time.time()
    global_train_samples = 0  
    total_eval_time = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(pbar):
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            if args.modality == "image":
                # Handle image data
                inputs, labels = batch
                inputs = inputs.to(device).to(torch.bfloat16)
                labels = labels.long().to(device) if num_classes > 1 else labels.float().to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = loss_fn(logits.float(), labels)

            elif args.modality == "text":
                inputs, attention_mask, labels = batch[0], batch[1], batch[-1]
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.long().to(device) if num_classes > 1 else labels.float().to(device)

                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = loss_fn(logits.float(), labels)

            else:
                raise ValueError("Unsupported modality type. Choose 'text' or 'image'.")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Update weights if we've accumulated enough gradients
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                # gradient clipping extreme
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                if args.schedule:
                    scheduler.step()
                global_step += 1
                if args.use_wandb and global_step % args.log_interval == 0:
                    rep = {'train/lr': scheduler.get_last_lr()[0]} if args.schedule else {}
                    log = {
                        "train/loss": loss.item() * (
                            1 if args.gradient_accumulation_steps == 1 else args.gradient_accumulation_steps),
                        "train/step": global_step,
                    }
                    log.update(rep)
                    wandb.log(
                        log
                    )

            epoch_loss += loss.item() * (
                1 if args.gradient_accumulation_steps == 1 else args.gradient_accumulation_steps)
            pbar.set_postfix(
                loss=loss.item() * (1 if args.gradient_accumulation_steps == 1 else args.gradient_accumulation_steps))
            if torch.is_tensor(labels):
                bs = labels.size(0)
            else:
                # fallback if labels could be a list/tuple
                bs = len(labels)
            global_train_samples += bs

        model.eval()
        eval_start = time.time()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}/{args.num_epochs} | Average loss: {avg_epoch_loss:.4f}")

        accuracy, eval_loss, eval_throughput, timing_stats = evaluate_classification(
            model,
            eval_dataloader,
            device,
            args.modality,
            description=f"Evaluating epoch {epoch + 1}",
            cola=args.dataset.lower() == "cola",
            f1=args.dataset.lower() in ["mrpc", "qqp"],
            stsb=args.dataset.lower() == "stsb",
        )
        eval_time = time.time() - eval_start  
        total_eval_time += eval_time         

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")

        print(f"\nEpoch {epoch + 1} Final Evaluation:")
        print(f"Accuracy: {accuracy:.4f} | Loss: {eval_loss:.4f}")
        print(f"Throughput Statistics:")
        print(f"  - Average: {timing_stats['avg_throughput']:.1f} samples/sec")
        print(f"  - Median (P50): {timing_stats['p50_throughput']:.1f} samples/sec")
        print(f"  - P90: {timing_stats['p90_throughput']:.1f} samples/sec")
        print(f"  - P95: {timing_stats['p95_throughput']:.1f} samples/sec")
        print(f"  - Min: {timing_stats['min_throughput']:.1f} samples/sec")
        print(f"  - Max: {timing_stats['max_throughput']:.1f} samples/sec")
        print(f"Batch Statistics:")
        print(f"  - Average Time: {timing_stats['avg_batch_time']*1000:.1f} ms")
        print(f"  - Std Dev Time: {timing_stats['std_batch_time']*1000:.1f} ms")
        print(f"  - Average Size: {timing_stats['avg_batch_size']:.1f} samples")

        if args.use_wandb:
            wandb.log({
                "train/epoch": epoch + 1,
                "train/epoch_loss": avg_epoch_loss,
                "eval/epoch_accuracy": accuracy,
                "eval/epoch_loss": eval_loss,
                #best_accuracy
                "best/epoch_best_accuracy": best_accuracy,
                "eval/epoch_throughput/avg": timing_stats['avg_throughput'],
                "eval/epoch_throughput/p50": timing_stats['p50_throughput'],
                "eval/epoch_throughput/p90": timing_stats['p90_throughput'],
                "eval/epoch_throughput/p95": timing_stats['p95_throughput'],
                "eval/epoch_throughput/min": timing_stats['min_throughput'],
                "eval/epoch_throughput/max": timing_stats['max_throughput'],
                "eval/epoch_batch/avg_time_ms": timing_stats['avg_batch_time']*1000,
                "eval/epoch_batch/std_time_ms": timing_stats['std_batch_time']*1000,
                "eval/epoch_batch/avg_size": timing_stats['avg_batch_size']
            })
        model.train()
    global_train_time = time.time() - global_train_start
    pure_train_time = global_train_time - total_eval_time

    # prevent divide 0
    if pure_train_time > 0:
        train_samples_per_second = global_train_samples / pure_train_time
    else:
        train_samples_per_second = 0
    print(f"\n[Summary] Train samples/sec: {train_samples_per_second:.3f} ({global_train_samples} samples, {pure_train_time:.1f} sec, not counting eval)")
    print(f"[Summary] Total evaluation time: {total_eval_time:.1f} sec")
    if args.use_wandb:
        wandb.log({
            "train_samples_per_second": train_samples_per_second,
            "train_runtime": pure_train_time,
            "eval_runtime_total": total_eval_time
        })
    if args.mode in ("mag_tp","wanda_tp"):
        assert MAG_TP_K_MAP is not None and MAG_TP_LAYER_NAME_MAP is not None, \
            "mag_tp requires pre-captured budget; missing MAG_TP_K_MAP/LAYER_NAME_MAP"

        k_map = MAG_TP_K_MAP
        layer_name_map = MAG_TP_LAYER_NAME_MAP
        if args.mode == "wanda_tp":
            sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=args.sample_ratio)
            tracker_a = NeuronTracker(model, topk_ratio=1.0, device=device, verbose=False)
            wanda_all = tracker_a.get_wanda_indices(dataloader=sel_loader)   
            def _is_fc1(lname:str)->bool:
                return any(k in lname for k in ["gate_proj","fc1","lin1","wi","mlp_fc1",
                                                "intermediate_dense","mlp.fc1","intermediate.dense"])
            sel_indices = {}
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    lname = layer_name_map.get(m, None)
                    if lname and _is_fc1(lname):
                        k = k_map.get(lname, 0)
                        idx = wanda_all.get(lname, [])
                        if k>0 and len(idx)>0:
                            sel_indices[lname] = torch.as_tensor(idx[:k], dtype=torch.long)
            active_neurons = sel_indices
        else: 
            mag_indices = {}
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    lname = layer_name_map.get(m, None)
                    if lname is None or not any(key in lname for key in [
                        "gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense",
                        "mlp.fc1","intermediate.dense"
                    ]):
                        continue
                    k = k_map.get(lname, 0)
                    if k <= 0:
                        continue
                    # L1 per-output（行）打分
                    score = m.weight.abs().sum(dim=1)
                    topk = torch.topk(score, min(k, score.numel())).indices
                    mag_indices[lname] = topk
                    active_neurons = mag_indices
        nst = NeuroselectiveTransformer5(
            model=model,
            active_neurons=active_neurons,
            layer_name_map=layer_name_map,
            tune_pruned=False,
            device=device,
            verbose=True
        )
        model = nst.transform().to(device).to(torch.bfloat16)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[mag_tp][post] params={final_param_count:,}, trainable={final_trainable_params:,}")
        try:
            F_fwd_variant = flops_forward(model, _inp1_fwd, device=str(device))
            print(f"[FLOPs] per-image forward (mag_tp): {F_fwd_variant/1e9:.3f} GFLOPs")
        except Exception as e:
            print(f"[mag_tp][post] FLOPs check skipped: {e}")
        acc_pp, loss_pp, thpt_pp, stats_pp = evaluate_classification(
        model, eval_dataloader, device, args.modality,
        description="[post_prune][pre_recovery] eval",
        cola=args.dataset.lower()=="cola",
        f1=args.dataset.lower() in ["mrpc","qqp"],
        stsb=args.dataset.lower()=="stsb",
        )
        print(f"[post_prune][pre_recovery] acc={acc_pp:.4f} | loss={loss_pp:.4f}")
        print(f"[post_prune][pre_recovery] throughput(avg)={stats_pp['avg_throughput']:.1f} samples/sec")
        if args.use_wandb:
            wandb.log({
                "eval_post_prune_pre_recovery/accuracy": acc_pp,
                "eval_post_prune_pre_recovery/loss":     loss_pp,
                "eval_post_prune_pre_recovery/throughput_avg": stats_pp["avg_throughput"],
                "eval_post_prune_pre_recovery/batch/avg_time_ms": stats_pp["avg_batch_time"]*1000,
            })


        print("\n[mag_tp][recovery] start 1-epoch recovery fine-tuning on the pruned subnetwork...")
        try:
            del optimizer
        except Exception:
            pass
        import gc; gc.collect(); torch.cuda.empty_cache()

        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "out_proj", "pooler"]):
                for _, p in module.named_parameters(recurse=False):
                    p.requires_grad = True
                    trainable.append(p)
                print(f"[mag_tp][recovery] trainable head: {name}")
        for name, module in model.named_modules():
            if isinstance(module, NeuroselectiveLinear):
                for _, p in module.named_parameters():
                    p.requires_grad = True
                    trainable.append(p)
                print(f"[mag_tp][recovery] trainable MLP:  {name}")
        print(f"[mag_tp][recovery] #trainable params: {sum(p.numel() for p in trainable):,}")

        optimizer_rec = AdamW(trainable, lr=args.recovery_lr, weight_decay=args.wd)

        try:
            F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
            print(f"[FLOPs] per-image train (mag_tp recovery): {F_train_rec/1e9:.3f} GFLOPs")
        except Exception as e:
            F_train_rec = 0
            print(f"[FLOPs][recovery] per-image train estimation failed: {e}")
        train_size = len(train_dataloader.dataset)
        training_flops_rec = int(F_train_rec) * int(train_size) * 1  # 1 epoch
        if args.use_wandb:
            wandb.log({
                "flops/train_recovery": training_flops_rec,
                "flops/per_image_train_recovery": F_train_rec,
            })

        model.train()
        rec_epoch_loss = 0.0
        global_step_rec = 0
        pbar = tqdm(train_dataloader, desc="[mag_tp][recovery] Epoch 1/1")
        for step, batch in enumerate(pbar):
            optimizer_rec.zero_grad()
            if args.modality == "image":
                inputs, labels = batch
                inputs = inputs.to(device).to(torch.bfloat16)
                labels = labels.long().to(device) if num_classes > 1 else labels.float().to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = loss_fn(logits.float(), labels)
            else:  # text
                inputs, attention_mask, labels = batch[0], batch[1], batch[-1]
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.long().to(device) if num_classes > 1 else labels.float().to(device)
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = loss_fn(logits.float(), labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer_rec.step()
                optimizer_rec.zero_grad()
                global_step_rec += 1

            rec_epoch_loss += loss.item() * (1 if args.gradient_accumulation_steps == 1 else args.gradient_accumulation_steps)
            pbar.set_postfix(loss=rec_epoch_loss / (step + 1))

        model.eval()
        acc_rec, loss_rec, thpt_rec, stats_rec = evaluate_classification(
            model, eval_dataloader, device, args.modality,
            description="[mag_tp][recovery] eval",
            cola=args.dataset.lower() == "cola",
            f1=args.dataset.lower() in ["mrpc", "qqp"],
            stsb=args.dataset.lower() == "stsb",
        )
        print(f"[mag_tp][recovery] accuracy={acc_rec:.4f} | loss={loss_rec:.4f}")
        print(f"[mag_tp][recovery] throughput(avg)={stats_rec['avg_throughput']:.1f} samples/sec")
        if args.use_wandb:
            wandb.log({
                "recovery/epoch_loss": rec_epoch_loss / max(1, len(train_dataloader)),
                "recovery/accuracy": acc_rec,
                "recovery/loss": loss_rec,
                "recovery/throughput/avg": stats_rec['avg_throughput'],
                "recovery/batch/avg_time_ms": stats_rec['avg_batch_time'] * 1000,
            })
        try:
            base_train_flops = 0
            for _k in [
                "training_flops_total", "total_train_flops", "train_total_flops",
                "baseline_train_flops", "train_flops_total"
            ]:
                if _k in locals():
                    base_train_flops = locals()[_k]
                    break
            train_total_including_recovery = base_train_flops + training_flops_rec
            print(f"[FLOPs][total] Train incl. recovery ≈ {train_total_including_recovery/1e15:.3f} PFLOPs "
                  f"({train_total_including_recovery/1e12:.3f} TFLOPs)")
            if args.use_wandb:
                wandb.log({
                    "flops/train_total_including_recovery": train_total_including_recovery,
                    "flops/train_total_including_recovery_pflops": train_total_including_recovery/1e15,
                    "flops/train_total_including_recovery_tflops": train_total_including_recovery/1e12,
                })
        except Exception as e:
            print(f"[FLOPs] logging train_total_including_recovery failed: {e}")

    # Final evaluation
    final_accuracy, final_loss, final_throughput, final_stats = evaluate_classification(
        model,
        eval_dataloader,
        device,
        args.modality,
        description="Final evaluation",
        cola = args.dataset.lower() == "cola",
        f1 = args.dataset.lower() in ["mrpc", "qqp"],
        stsb = args.dataset.lower() == "stsb",
    )
    # Save unrestored model checkpoint
    unrestored_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_accuracy': final_accuracy,
        'final_loss': final_loss,
        'args': vars(args)
    }
    torch.save(unrestored_checkpoint, os.path.join(run_dir, "unrestored_model.pth"))
    print(f"Saved unrestored model checkpoint to {os.path.join(run_dir, 'unrestored_model.pth')}")

    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {final_accuracy:.4f} | Loss: {final_loss:.4f}")
    
    # Initialize variables for restored model
    restored_accuracy = None
    restored_loss = None
    restored_throughput = None
    restored_stats = None
    
    if args.mode in ["stt", "stt_lora"]:  # Only for neuron selection modes
        print("\n" + "="*60)
        print("EXPERIMENT: Evaluating with restored neurons")
        print("="*60)
        
        if args.mode == "stt_lora":
            print("STT_LoRA Mode: Merge first, then restore neurons")
            
            # Phase 1: Merge LoRA adapters first
            print("Phase 1: Merging LoRA adapters...")
            for name, module in model.named_modules():
                if isinstance(module, NSLoraLinear):
                    if not module.merge_weights:
                        print(f"  Merging adapters in {name}")
                        module.merge()
                    else:
                        print(f"  Adapters already merged in {name}")
            
            # Phase 2: Restore pruned neurons in NSLoraLinear modules
            print("Phase 2: Restoring pruned neurons...")
            for name, module in model.named_modules():
                if isinstance(module, NSLoraLinear):
                    print(f"  Restoring neurons in {name}")
                    module.ns_linear.restore_full_weights()
            
            print("NSLoRA merging and restoration complete!")
        
        restored_accuracy, restored_loss, restored_throughput, restored_stats = evaluate_with_restored_neurons(
            model, eval_dataloader, device, args.modality,
            description="Evaluation with restored neurons"
        )
        
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Pruned model accuracy:  {final_accuracy:.4f}")
        print(f"Restored model accuracy: {restored_accuracy:.4f}")
        print(f"Improvement:            {restored_accuracy - final_accuracy:+.4f}")
        print(f"Relative improvement:   {((restored_accuracy - final_accuracy) / final_accuracy * 100):+.2f}%")
        
        # Save restored model checkpoint
        restored_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'restored_accuracy': restored_accuracy,
            'restored_loss': restored_loss,
            'final_accuracy': final_accuracy,  # Keep original for reference
            'final_loss': final_loss,
            'args': vars(args)
        }
        torch.save(restored_checkpoint, os.path.join(run_dir, "restored_model.pth"))
        print(f"Saved restored model checkpoint to {os.path.join(run_dir, 'restored_model.pth')}")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "experiment/pruned_accuracy": final_accuracy,
                "experiment/restored_accuracy": restored_accuracy,
                "experiment/improvement_abs": restored_accuracy - final_accuracy,
                "experiment/improvement_rel": (restored_accuracy - final_accuracy) / final_accuracy * 100,
            })
    print(f"Throughput Statistics:")
    if 'f1' in final_stats: 
        print(f"F1: {final_stats['f1']:.4f}")
    if 'precision' in final_stats: 
        print(f"Precision: {final_stats['precision']:.4f}")
    if 'recall' in final_stats: 
        print(f"Recall: {final_stats['recall']:.4f}")
    if 'mcc' in final_stats: 
        print(f"MCC: {final_stats['mcc']:.4f}")
    if 'spearman' in final_stats: 
        print(f"Spearman: {final_stats['spearman']:.4f}")

    print(f"  - Average: {final_stats['avg_throughput']:.1f} samples/sec")
    print(f"  - Median (P50): {final_stats['p50_throughput']:.1f} samples/sec")
    print(f"  - P90: {final_stats['p90_throughput']:.1f} samples/sec")
    print(f"  - P95: {final_stats['p95_throughput']:.1f} samples/sec")
    print(f"  - Min: {final_stats['min_throughput']:.1f} samples/sec")
    print(f"  - Max: {final_stats['max_throughput']:.1f} samples/sec")
    print(f"Batch Statistics:")
    print(f"  - Average Time: {final_stats['avg_batch_time']*1000:.1f} ms")
    print(f"  - Std Dev Time: {final_stats['std_batch_time']*1000:.1f} ms")
    print(f"  - Average Size: {final_stats['avg_batch_size']:.1f} samples")
    print(f"Best accuracy: {best_accuracy:.4f}")


    results = {
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "final_throughput": final_stats['avg_throughput'],
        "throughput_p50": final_stats['p50_throughput'],
        "throughput_p90": final_stats['p90_throughput'],
        "throughput_p95": final_stats['p95_throughput'],
        "throughput_min": final_stats['min_throughput'],
        "throughput_max": final_stats['max_throughput'],
        "avg_batch_time_ms": final_stats['avg_batch_time']*1000,
        "std_batch_time_ms": final_stats['std_batch_time']*1000,
        "best_accuracy": best_accuracy,
        "run_name": run_name,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "mode": args.mode,
        "threshold": args.threshold if args.mode == "stt" else None,
        "use_abs_threshold": args.use_abs_threshold if args.mode == "stt" else None,
        "lora_r": args.lora_r if "lora" in args.mode else None,
        "lora_alpha": args.lora_alpha if "lora" in args.mode else None,
        # Add restored model metrics if available
        "restored_accuracy": restored_accuracy,
        "restored_loss": restored_loss,
        "restored_throughput": restored_stats['avg_throughput'] if restored_stats else None,
        "improvement_abs": restored_accuracy - final_accuracy if restored_accuracy is not None else None,
        "improvement_rel": ((restored_accuracy - final_accuracy) / final_accuracy * 100) if restored_accuracy is not None else None
    }

    # import json
    # with open(os.path.join(run_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=2)

    print(f"Training completed. Results saved to {run_dir}")
    if args.use_wandb:
        wandb_log = {
            "final/accuracy": final_accuracy,
            "final/loss": final_loss,
            "final/throughput/avg": final_stats['avg_throughput'],
            "final/throughput/p50": final_stats['p50_throughput'],
            "final/throughput/p90": final_stats['p90_throughput'],
            "final/throughput/p95": final_stats['p95_throughput'],
            "final/throughput/min": final_stats['min_throughput'],
            "final/throughput/max": final_stats['max_throughput'],
            "final/batch/avg_time_ms": final_stats['avg_batch_time']*1000,
            "final/batch/std_time_ms": final_stats['std_batch_time']*1000,
            "best/accuracy": best_accuracy
        }
        
        # Add restored model metrics if available
        if restored_accuracy is not None:
            wandb_log.update({
                "final/restored_accuracy": restored_accuracy,
                "final/restored_loss": restored_loss,
                "final/restored_throughput/avg": restored_stats['avg_throughput'],
                "final/improvement_abs": restored_accuracy - final_accuracy,
                "final/improvement_rel": ((restored_accuracy - final_accuracy) / final_accuracy * 100)
            })
        
        wandb.log(wandb_log)
        wandb.finish()

    return results


if __name__ == "__main__":
    main()