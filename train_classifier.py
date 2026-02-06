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
from stt.stt_lora import STTLoraLinear
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
    clear_cuda_cache_and_states,
    pad_all_nslinear_modules,
    bench_forward,
    bench_forward_image
)
import torch
from torch.utils.data import DataLoader
from torch import nn

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
                        choices=["stt", "stt_lora", "baseline", "magnitude_pruning", "wanda_adapt"],
                        help="Run mode: 'stt' for selection then train, 'stt_lora' for STT+LoRA, 'baseline' for full finetuning, 'magnitude_pruning' for magnitude-based pruning, 'wanda_adapt' for Wanda adaptive pruning")

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
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for evaluation (defaults to batch_size if not specified)")
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
    # Set eval_batch_size to batch_size if not specified
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
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
    # Recreate eval dataloaders with eval_batch_size if different from batch_size
    if args.eval_batch_size != args.batch_size:
        print(f"[Config] Using separate batch sizes: train={args.batch_size}, eval={args.eval_batch_size}")
        # Recreate eval_dataloader
        eval_dataloader = DataLoader(
            eval_dataloader.dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=eval_dataloader.collate_fn if hasattr(eval_dataloader, 'collate_fn') and eval_dataloader.collate_fn is not None else None
        )
        # Recreate val_dataloader if it exists
        if val_dataloader is not None:
            val_dataloader = DataLoader(
                val_dataloader.dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=val_dataloader.collate_fn if hasattr(val_dataloader, 'collate_fn') and val_dataloader.collate_fn is not None else None
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

    model = model.to(device).to(torch.float32)
    if args.modality == "image":
        _ = peek_vit_dataloader(train_dataloader, model=model, n_batches=2, prefix="[VIT-TRAIN]")
        _ = peek_vit_dataloader(eval_dataloader, model=model, n_batches=2, prefix="[VIT-TEST]")
    
    # Prepare FLOPs calculation
    F_fwd_base = None
    F_fwd_variant = None
    F_train_variant = None
    selection_flops = 0
    
    _inp1_fwd, _inp1_train, _criterion, _y1, _flops_dbg_once = prepare_flops_inputs_and_criterion(
        model=model,
        train_dataloader=train_dataloader,
        modality=args.modality,
        device=device,
        dtype=torch.float32,
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
    if args.mode == "stt":
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
            device=device,
            verbose=True
        )

        active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker.get_layer_name_map()
        # print(f"Active neurons selected: {len(active_neurons)}")
        
        # Visualization code (not core functionality, commented out)
        # model_clean = args.model_name.replace('/', '_').replace('-', '_')
        # dataset_clean = args.dataset.replace('/', '_').replace('-', '_')
        # print("\n--- Generating dual metric composition visualizations ---")
        # composition_stats = tracker.compute_dual_metric_composition(delta=args.threshold)
        # tracker.visualize_layer_composition(
        #     composition_stats, 
        #     model_name=f"{model_clean}_topk{args.topk_ratio}", 
        #     dataset_name=dataset_clean
        # )

        non_empty_layers = {k: v for k, v in active_neurons.items() if v.numel() > 0}
        print(f"[tracker] non-empty layers = {len(non_empty_layers)} / {len(active_neurons)}")
        if not non_empty_layers:
            raise RuntimeError("Tracker failed.")

        nstransformer = STTTransformer(
            model=model,  # Use the fresh or original model instance
            active_neurons=active_neurons,  
            layer_name_map=layer_name_map, 
            verbose=True,
            device=device,
            inference_time=False  
        )

        model = nstransformer.transform().to(device).to(torch.float32)
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

        trainable = []
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
            # if any(key in name for key in ["classifier"]):
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

        if sel_samples == 0:
            raise ValueError(f"Active dataloader is empty! sample_ratio={args.sample_ratio} resulted in 0 samples. "
                           f"Please increase sample_ratio or check dataset size.")

        first_batch = next(iter(active_dataloader))
        print("Doing STTLora now")
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
            tokenizer=tokenizer if args.modality == "text" else None,  
            threshold=args.threshold,                   
            topk_ratio=args.topk_ratio,             
            device=device,
            verbose=True
        )

        active_neurons = tracker.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker.get_layer_name_map()

        non_empty_layers = {k: v for k, v in active_neurons.items() if v.numel() > 0}
        print(f"[tracker] non-empty layers = {len(non_empty_layers)} / {len(active_neurons)}")
        if not non_empty_layers:
            raise RuntimeError("Tracker failed.")

        nstransformer = STTTransformer(
            model=model,  
            active_neurons=active_neurons,  
            layer_name_map=layer_name_map,  
            verbose=True,
            device=device,
            inference_time=False  
        )

        model = nstransformer.transform().to(device).to(torch.float32)
        print("Pruning complete")

        for name, module in model.named_modules():
            if isinstance(module, STTLinear):
                parent_path, attr = name.rsplit(".", 1)
                parent = model
                for p in parent_path.split("."):
                    parent = getattr(parent, p)
                ns_lora = STTLoraLinear(
                    stt_linear=module,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    merge_weights=False
                )
                ns_lora = ns_lora.to(device).to(torch.float32)

                setattr(parent, attr, ns_lora)
                print(f"  Wrapped {name}")

        for param in model.parameters():
            param.requires_grad = False
        for nm, mod in model.named_modules():
            if isinstance(mod, STTLoraLinear):
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
    elif args.mode == "magnitude_pruning":
        sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=float(args.sample_ratio))
        tracker = NeuronTracker(model, threshold=0.01, topk_ratio=args.topk_ratio,
                                device=device, verbose=False)
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
        nst = STTTransformer(model, active_neurons=mag_indices,
                                        layer_name_map=layer_name_map,
                                        device=device, verbose=True,
                                        inference_time=False  
                                        )
        model = nst.transform().to(device).to(torch.float32)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=args.lr, weight_decay=args.wd)
        F_fwd_variant   = flops_forward(model, _inp1_fwd, device=str(device))
        F_train_variant = flops_train_step(model, _inp1_train, device=str(device), criterion=_criterion)
        print(f"[FLOPs] per-image forward ({args.mode}): {F_fwd_variant/1e9:.3f} GFLOPs")
        print(f"[FLOPs] per-image train   ({args.mode}): {F_train_variant/1e9:.3f} GFLOPs") 
    elif args.mode == "wanda_adapt":  
        print("Doing Wanda adaptive pruning now")
        sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=float(args.sample_ratio))
        tracker_budget = NeuronTracker(
            model, threshold=0.01, topk_ratio=args.topk_ratio,
            device=device, verbose=False
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
            k = int(k_map.get(lname, 0))    
            if k <= 0:
                continue
            idx_t = torch.as_tensor(idx[:k], dtype=torch.long)
            sel_indices[lname] = idx_t
        nst = STTTransformer(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            device=device, verbose=True,
            inference_time=False  # Training: use scatter mode
        )
        model = nst.transform().to(device).to(torch.float32)
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

        sel_loader = sample_active_set(non_shuffle_train_dataloader, ratio=args.sample_ratio)
        tracker = NeuronTracker(model, threshold=0.01, topk_ratio=args.topk_ratio,
                            device=device, verbose=False)
        active = tracker.get_active_indices(dataloader=sel_loader)
        layer_name_map = tracker.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in active.items()}
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
    print(f"Selection : {selection_flops/1e9:.2f} GFLOPs")   
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
                inputs = inputs.to(device).to(torch.float32)
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
    is_stt_mode = args.mode == "stt"    
    if is_stt_mode:
        print("\n[Evaluation] inference throughput for pruning...")
        clear_cuda_cache_and_states(
            optimizer=optimizer if 'optimizer' in locals() else None,
            trainer=None,  # train_classifier doesn't use trainer object
            model=model,
            device=device,
            verbose=True
        )
        modified_count = 0
        for name, module in model.named_modules():
            if isinstance(module, STTLinear):
                is_gate_or_up = (
                    'gate_proj' in name or 'up_proj' in name or  # LLM
                    'fc1' in name or 'intermediate.dense' in name or 'intermediate_dense' in name  # ViT/BERT
                )
                is_down = (
                    'down_proj' in name or  # LLM
                    'fc2' in name or 'output.dense' in name or 'output_dense' in name  # ViT/BERT
                )
                
                if is_gate_or_up:
                    module.inference_time = True
                    modified_count += 1
                elif is_down:
                    module.inference_time = False  # Explicitly set to False
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Evaluation] Model switched to inference_time mode")
        print(f"[Evaluation] Modified {modified_count} gate/up/fc1/intermediate modules to inference_time=True")
        print(f"\n[Padding] Applying padding 128 for True Pruning mode...")
        pad_all_nslinear_modules(model, pad_to=128)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[Padding] Padding applied successfully.")
    else:
        print("\n[Evaluation] Using baseline mode (no pruning)")
    
    if eval_dataloader is not None:
        mode_desc = "True Pruning mode (with padding 128)" if is_stt_mode else "Baseline mode"
        print(f"\n[Benchmark] Testing {mode_desc}...")
        if args.modality == "image":
            throughput, latency_ms = bench_forward_image(model, eval_dataloader=eval_dataloader, iters=200, warmup=20, device=device)
        else:  # text
            throughput, latency_ms = bench_forward(model, eval_dataloader=eval_dataloader, iters=200, warmup=20, device=device)
        print(f"  Throughput: {throughput:.2f} samples/sec (using real data)")
        print(f"  Latency:    {latency_ms:.2f} ms/batch")
    else:
        print(f"\n[Benchmark] Skipping benchmark (no test data available)")

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
        "threshold": args.threshold if args.mode == "stt" else None,
        "lora_r": args.lora_r if "lora" in args.mode else None,
        "lora_alpha": args.lora_alpha if "lora" in args.mode else None,
    }

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