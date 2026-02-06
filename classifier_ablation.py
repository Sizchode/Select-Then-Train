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
from stt.mlps.stt_linear2 import STTLinear  # Use the selective linear layer from stt_linear2.py
from stt.stt_transformer import STTTransformer  # Use the pruner from stt_transformer.py
from stt.stt_tracker import STTTracker as NeuronTracker
from stt.ablation_tracker import AblationTracker
from stt.dataset import get_dataloader
from util.torch_flops import (
    flops_forward,          
    flops_train_step,    
    TEXT_MAXLEN,
    peek_vit_dataloader,
    prepare_flops_inputs_and_criterion
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
    VisionEncoderWithClassifier,
    calculate_jaccard_similarity,
)
from torch.utils.data import DataLoader
MAG_TP_K_MAP = None
MAG_TP_LAYER_NAME_MAP = None

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
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["lora", "baseline", "mag_tp", "wanda_tp",
                        "activation_mean_value", "activation_rate", "calibration"],
                        help="Run mode: 'activation_mean_value' for activation magnitude ablation, 'activation_rate' for activation frequency ablation, etc.")

    # Neuron selection options
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Activation threshold for neuron selection")
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
    
    # Prepare FLOPs calculation inputs and criterion
    F_fwd_base = None
    F_fwd_variant = None
    F_train_variant = None  
    selection_flops = 0     

    _inp1_fwd, _inp1_train, _criterion, _y1, _flops_dbg_once = prepare_flops_inputs_and_criterion(
        model=model,
        train_dataloader=train_dataloader,
        modality=args.modality,
        device=device,
        dtype=torch.bfloat16,
        flops_debug=True
    )
    
    if args.modality == "image":
        F_fwd_base = flops_forward(model, _inp1_fwd, device=str(device))
        print(f"[FLOPs] per-image forward (baseline): {F_fwd_base/1e9:.3f} GFLOPs")
    elif args.modality == "text":
        print(f"[DEBUG] About to call flops_forward with inputs: {list(_inp1_fwd.keys())}")
        F_fwd_base = flops_forward(model, _inp1_fwd, device=str(device))
        print(f"[FLOPs][BERT] per-example forward baseline (L={TEXT_MAXLEN}): {F_fwd_base/1e9:.3f} GFLOPs")
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
                    device=device,
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
    elif args.mode == "activation_mean_value":
        print("[ABLT] one-shot, sparsity-based selection; budget-matched to NS per layer")
        sel_loader = sample_active_set(non_shuffle_train_dataloader, float(args.sample_ratio))
        
        # First run NS tracker (tracker6) to get K-per-layer budget
        tracker_budget = NeuronTracker(
            model, threshold=0.01, topk_ratio=args.topk_ratio,
            device=device, verbose=False
        )
        active = tracker_budget.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}

        # Now use AblationTracker (ablation_tracker) with ONLY sparsity metric
        print("[ABLT] Using ablation_tracker with ONLY sparsity metric (fraction of samples > threshold)")
        tracker_a = AblationTracker(model, topk_ratio=args.topk_ratio, device=device, verbose=False)  
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


        nst = STTTransformer(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            device=device, verbose=True
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
            if isinstance(module, STTLinear):
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
            device=device, verbose=False
        )
        active = tracker_budget.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}

        # Now use AblationTracker (ablation_tracker) with activation rate metric
        print("[ABLT2] Using ablation_tracker with activation rate metric (fraction of samples > threshold)")
        tracker_a = AblationTracker(model, topk_ratio=args.topk_ratio, device=device, verbose=False)  
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


        nst = STTTransformer(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            device=device, verbose=True
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
            if isinstance(module, STTLinear):
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
                                device=device, verbose=False)
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
                            device=device, verbose=False)
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
        nst = STTTransformer(
            model=model,
            active_neurons=active_neurons,
            layer_name_map=layer_name_map,
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
            if isinstance(module, STTLinear):
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
        "threshold": args.threshold if args.mode in ["activation_mean_value", "activation_rate"] else None,
        "lora_r": args.lora_r if "lora" in args.mode else None,
        "lora_alpha": args.lora_alpha if "lora" in args.mode else None
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
        
        wandb.log(wandb_log)
        wandb.finish()

    return results


if __name__ == "__main__":
    main()