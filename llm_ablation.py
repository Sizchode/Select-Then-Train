import argparse
import logging
from datetime import datetime
import torch
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import AdamW
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
import os
from META import LLM_DATASET_PATHS as dataset_paths  # Assuming this exists
from stt.dataset import CLUTRR
from stt.dataset.genloader_2 import BoolQ, ARC
from stt.mlps.stt_linear2 import STTLinear
from stt.stt_transformer import STTTransformer
from stt.stt_tracker import STTTracker as NeuronTracker
from stt.ablation_tracker import AblationTracker
from peft import get_peft_model, LoraConfig
from stt.trainers import CustomSFTTrainerV2
from stt.stt_lora import STTLoraLinear  
from util.utils import (
    set_seed,
    setup_lora,
    calculate_jaccard_similarity,
    calculate_directed_coverage
)
from util.torch_flops import (
    estimate_flops_infer, estimate_flops_train,
)
import torch.nn as nn
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Hidden Dimension Pruning Training for LLMs')

    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--task', "--dataset", type=str,
                        choices=['clutrr', 'boolq', 'arc-e', 'arc-c'],
                        default="clutrr",
                        help='Task/dataset to evaluate on')
    parser.add_argument('--mode', type=str,
                        choices=["stt", "lora", "adalora", "loha", "lokr", "finetune", "baseline", "stt_lora","mag_pt", "mag_tp","wanda_p","transfer", "covplot",
                        "wanda_tp", "calibration","activation_mean_value", "activation_rate"],
                        default="stt_lora",  # Changed default to stt_lora
                        help="Training mode: 'stt' for hidden dim pruning, 'stt_lora' for STT+LoRA, 'activation_mean_value' for activation magnitude ablation, 'activation_rate' for activation frequency ablation, etc.")
    parser.add_argument('--apply_chat_template',
                        action='store_true',
                        help='Using chat template in the prompt; False by default')

    parser.add_argument('--lr', "--learning_rate", type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--recovery_lr', "--recovery)learning_rate", type=float, default=1e-5, help='recovery Learning rate')

    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--train_batch', "--batch_size", type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_epoch', "--num_epochs", type=int, default=10, help='Number of epochs to fine-tune')
    parser.add_argument('--train_size', type=int, default=0, help='Number of training examples to use (0 for all)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients')


    parser.add_argument('--active_threshold', type=float, default=0.01,
                        help='Activation threshold for finding active neurons')
    parser.add_argument('--active_thresholds', type=float, nargs=2, default=None,
                        help='Two activation thresholds for neuron selection (e.g., 0.01 0.05)')
    parser.add_argument('--use_abs_threshold', action='store_true',
                        help='Use absolute threshold values for activation')
    parser.add_argument('--active_sample_ratio', type=float, default=0.1,
                        help='Sample ratio of training data for activation tracking')
    parser.add_argument("--topk_ratio",type=float, default=0.30, metavar="R",
                        help="Fraction (0,1] of neurons kept per score when selecting active neurons")

# Removed unused parameters (aggregation_mode, hidden_dim_patterns, input_conn_patterns, output_conn_patterns)
    parser.add_argument('--tune_pruned', action='store_true', help='Make pruned layers trainable')

    parser.add_argument('--lora_r', type=int, default=16,
                        help='Rank of the LoRA updates')
    parser.add_argument('--lora_alpha', type=float, default=32,
                        help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='Dropout rate for LoRA layers')

    parser.add_argument('--output_dir', default='./outputs', type=str,
                        help='Output directory for checkpoints and outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='hidden-dim-pruning-llm', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Log to WandB every N steps')

# Removed tqa_fold_num parameter for single GPU setup

    parser.add_argument("--schedule", action="store_true", help="enable linear schedule")
    parser.add_argument("--tune_attn", action="store_true", help="tune attn proj during tracking (if tracker supports)")
    parser.add_argument('--dev_mode', action='store_true',
                        help='Use a held-out dev set from training set for validation (grid search); do NOT use real test set')
    args = parser.parse_args()

    set_seed(args.seed)
    mag_tp_k_map = {}
    mag_tp_layer_map = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_map = {
        'qwen': 'Qwen/Qwen2.5-1.5B',
        "qwen3":"Qwen/Qwen2.5-3B",
        "qwen7": "Qwen/Qwen2.5-7B",
    }

    args.model_name = model_map[args.model]

    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{args.model_name.split('/')[-1]}_{args.task}_{args.mode}_lr{args.lr}_e{args.num_epoch}_s{args.seed}_{timestamp}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )

    if args.active_thresholds is not None:
        if len(args.active_thresholds) != 2:
            raise ValueError("Please provide exactly two values for --active_thresholds.")
        args.active_threshold = args.active_thresholds
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16
        )
    print(model.dtype)   
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    orig_param_count = sum(p.numel() for p in model.parameters())
    orig_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loading dataset: {args.task}")
    if args.task == 'clutrr':
        data_loader = CLUTRR(
            split_dir=dataset_paths['clutrr'],
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'boolq':
        data_loader = BoolQ(
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'arc-e':
        data_loader = ARC(
            subset="easy",
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    elif args.task == 'arc-c':
        data_loader = ARC(
            subset="challenge",
            chat_template=args.apply_chat_template,
            model_name=args.model_name
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}. Supported tasks: clutrr, boolq, arc-e, arc-c")

    datasets = data_loader.load_data(train_size=args.train_size)

    if args.dev_mode:
        if args.task in ['arc-e', 'arc-c']:
            train_dataset = datasets['train']
            test_dataset = datasets['validation'] 
        else:  # clutrr, boolq
            dev_dataset, train_dataset = data_loader.get_dev_set(ratio=args.topk_ratio, return_rest=True)
            test_dataset = dev_dataset
    else:
        if args.task == 'boolq':
            train_dataset = datasets['train']
            test_dataset = datasets['test']
        elif args.task in ['arc-e', 'arc-c']:
            train_dataset = datasets['train']
            test_dataset = datasets['test']
        else:  # clutrr
            train_dataset = datasets['train']
            test_dataset = datasets['test']

    if args.task == 'boolq':
        response_template_with_context = " A:\n"
    elif args.task == 'clutrr':
        response_template_with_context = " 's\n"
    elif args.task in ['arc-e', 'arc-c']:
        response_template_with_context = "Answer:\n"
    else:
        response_template_with_context = "\n"  # Default fallback

    if args.apply_chat_template:
        response_template_with_context = \
            tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True).split('\n')[-1]  # Heuristic
        if not response_template_with_context: response_template_with_context = " [/INST]"  # Fallback

    offset = 1  # Adjust based on tokenizer behavior if needed
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[offset:]
    model = model.to(device)
    optimizer = None
    pruner_stats = None
    if args.mode == "calibration":
        print("--- [MODE: ANALYSIS] Running Sub-network Stability Analysis ---")
        
        # 1. Define the calibration set ratios to test
        ratios_to_test = [0.01, 0.02, 0.05, 0.1]
        results_by_ratio = {} 

        print(f"[*] Testing active set stability for ratios: {ratios_to_test}")

        # 2. Loop through each ratio to find the active neurons
        for ratio in ratios_to_test:
            print(f"\n----- Processing ratio: {ratio} -----")
            # Get a small subset of data for calibration
            sampled_subset = data_loader.get_active_set(ratio)
            sampled_subset = sampled_subset['text']
            active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
            print(f"[MEM] Before processing ratio {ratio}: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")               
            # Initialize the tracker to find active neurons
            tracker = NeuronTracker(
                model=model,
                tokenizer=tokenizer,
                threshold=args.active_threshold,
                topk_ratio=args.topk_ratio, 
                use_abs_threshold=args.use_abs_threshold,
                device=device,
                track_attention_proj=args.tune_attn,
                verbose=False # Keep the log clean during the loop
            )
            
            # Get and store the dictionary of active neurons for this ratio
            active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
            if active_indices_dict:
                num_layers = len(active_indices_dict) - 1 if '_attn_proj_layers' in active_indices_dict else len(active_indices_dict)
                print(f"[*] Found active indices for {num_layers} layers.")
                results_by_ratio[ratio] = active_indices_dict
                print(f"[MEM] After storing ratio {ratio}: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
            else:
                print("[!] Warning: No active neurons found for this ratio.")
        
        # 3. Calculate the pairwise similarity matrix
        print("\n[*] Calculating similarity matrix for the heatmap...")
        tested_ratios = sorted(results_by_ratio.keys())
        num_ratios = len(tested_ratios)
        similarity_matrix = np.zeros((num_ratios, num_ratios))

        for i in range(num_ratios):
            for j in range(num_ratios):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                r1 = tested_ratios[i]
                r2 = tested_ratios[j]
                dict1 = results_by_ratio[r1]
                dict2 = results_by_ratio[r2]
                
                similarity = calculate_jaccard_similarity(dict1, dict2)
                similarity_matrix[i, j] = similarity
        
        # 4. Generate and save the heatmap visualization
        print("[*] Generating and saving heatmap...")

        N = similarity_matrix.shape[0]

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

        base = f"stability_heatmap_{str(args.model).replace('/', '_')}_{args.task}"
        plt.savefig(f"{base}.pdf", bbox_inches="tight")
        plt.savefig(f"{base}.svg", bbox_inches="tight")
        print(f"[*] Heatmap saved to: {base}.pdf / {base}.svg")
    if args.mode == "stt" or args.mode == "stt_lora":
        print("--- Starting Neuron Activation Tracking ---")
        print("[FLOPs] Measuring activation tracking cost...")
        # 1. Data Preparation
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )
        print(f"[FLOPs] Selection (active set) ≈ {sel_stats['flops']/1e12:.3f} TFLOPs "
              f"[FLOPs] Selection (active set) ≈ {sel_stats['flops']/1e15:.3f} PFLOPs "
              f"(N={sel_stats['N']:,}, L=2048)")
        if args.use_wandb:
            wandb.log({
                "flops/selection_total": sel_stats["flops"],
                "flops/selection_total_tflops": sel_stats["flops"]/1e12,
                "flops/selection_N": sel_stats["N"],
            })

        # 2. NeuronTracker Initialization
        print("Initializing NeuronTracker...")
        tracker = NeuronTracker(
            model=model,  # Assuming 'model' is the loaded original model
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        # 3. Tracker Usage
        print("Running activation tracking...")
        layer_map = tracker.get_layer_name_map()
        active_indices_dict = None
        active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
        composition_stats = tracker.compute_dual_metric_composition(delta=args.active_threshold)
        tracker.visualize_layer_composition(
            composition_stats,
            model_name=f"{args.model_name}_topk{args.topk_ratio}",
            dataset_name=args.task
        )

        if active_indices_dict is None:
            active_indices_dict = {}
        print(f"Tracking complete. Found active indices for "
                f"{len(active_indices_dict) - 1 if active_indices_dict and '_attn_proj_layers' in active_indices_dict else (len(active_indices_dict) if active_indices_dict else 0)} layers.")

        if len(active_indices_dict) == 0:
            print("No active neurons found — skipping pruning.")
            model_to_prune = model
        else:
            print("Loading fresh model instance for pruning...")
            model_to_prune = model


        # 4. STTTransformer Initialization
        print("Initializing STTTransformer...")
        transformer = STTTransformer(
            model=model_to_prune,
            active_neurons=active_indices_dict,
            layer_name_map=layer_map,
            verbose=True,
            tune_pruned=False,
            device=device
        )
        print("Performing model transformation (pruning)...")
        pruned_model = transformer.transform()
        pruned_model = pruned_model.to(device)
        model = pruned_model
        print("Model variable now points to the pruned model.")

        # For stt_lora mode, apply STTLoraLinear to STTLinear layers
        if args.mode == "stt_lora":
            print("--- Applying STTLoraLinear to STTLinear layers ---")
            # Convert STTLinear layers to STTLoraLinear
            for name, module in model.named_modules():
                if isinstance(module, STTLinear):
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    attr_name = name.split('.')[-1]

                    # Get parent module
                    parent = model
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)

                    # Replace STTLinear with STTLoraLinear
                    ns_lora = STTLoraLinear(
                        stt_linear=module,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        merge_weights=False
                    )

                    # Set the attribute on the parent module
                    setattr(parent, attr_name, ns_lora)

            print("STTLoraLinear transformation complete.")
            _bp = next(p for p in model.parameters() if p.is_floating_point())  # [ADDED]
            for m in model.modules():  # [ADDED]
                if isinstance(m, STTLoraLinear):  # [ADDED]
                    m.lora_A.to(device=_bp.device, dtype=_bp.dtype)  # [ADDED]
                    m.lora_B.to(device=_bp.device, dtype=_bp.dtype)  # [ADDED]
                    m.scaling = torch.as_tensor(m.scaling, device=_bp.device, dtype=_bp.dtype)  # [ADDED]
            print("[NSLoRA] base:", _bp.dtype, "adapters:", {m.lora_A.weight.dtype for m in model.modules() if isinstance(m, STTLoraLinear)})  # [ADDED]
    elif args.mode == "activation_mean_value":
        print("[ABLT] one-shot, sparsity-based selection; budget-matched to NS per layer")
        # 1) Build a small, non-shuffled active set (reuse your existing loader & ratio)
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        # 2) First run NS tracker (tracker6) to get K-per-layer budget
        tracker_budget = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,               # this sets the per-layer K for the budget
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=False
        )
        ns_active = tracker_budget.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in (ns_active or {}).items()}  # per-layer K budget from NS

        # 3) Now use AblationNeuronTracker (ablation_tracker) with ONLY sparsity metric
        print("[ABLT] Using ablation_tracker with ONLY sparsity metric (fraction of samples > threshold)")
        ablation_tracker = AblationNeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,  # use the same ratio as NS
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=False
        )
        # Use sparsity-based selection with same threshold
        sparsity_active = ablation_tracker.get_active_indices(dataloader=active_dataloader)

        def _is_fc1(lname: str) -> bool:
            # mirror classifier-side layer filter
            keys = ["gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense"]
            lname_l = (lname or "").lower()
            return any(k in lname_l for k in keys)

        # 4) Trim sparsity picks to match the NS budget per layer
        sel_indices = {}
        for m in model.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            lname = layer_name_map.get(m, None)
            if lname is None or not _is_fc1(lname):
                continue
            sparsity_idx = sparsity_active.get(lname, None)
            k = int(k_map.get(lname, 0))
            if sparsity_idx is None or k <= 0:
                continue
            # Trim to match budget
            keep = min(k, len(sparsity_idx))
            sel_indices[lname] = sparsity_idx[:keep]

        # 5) Transform with STTTransformer and freeze everything except lm_head
        nst = STTTransformer(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            tune_pruned=False, device=device, verbose=True
        )
        model = nst.transform().to(device)
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
                print(f"[ABLT] trainable head: {name}")

        print("[ABLT] Wanda (budget-matched) model ready; proceed to train/eval as usual.")
    elif args.mode == "activation_rate":
        print("[ABLT2] one-shot, activation-rate-based selection; budget-matched to NS per layer")
        # 1) Build a small, non-shuffled active set (reuse your existing loader & ratio)
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        # 2) First run NS tracker (tracker6) to get K-per-layer budget
        tracker_budget = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,               # this sets the per-layer K for the budget
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=False
        )
        ns_active = tracker_budget.get_active_indices(dataloader=active_dataloader)
        layer_name_map = tracker_budget.get_layer_name_map()
        k_map = {ln: len(idx) for ln, idx in (ns_active or {}).items()}  # per-layer K budget from NS

        # 3) Now use AblationNeuronTracker (ablation_tracker) with activation rate (sparsity) metric
        print("[ABLT2] Using ablation_tracker with activation rate metric (fraction of samples > threshold)")
        ablation_tracker = AblationNeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,  # use the same ratio as NS
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=False
        )
        # Use activation-rate-based selection (sparsity) with same threshold
        sparsity_active = ablation_tracker.get_active_indices(dataloader=active_dataloader, use_activation_rate=True)

        def _is_fc1(lname: str) -> bool:
            # mirror classifier-side layer filter
            keys = ["gate_proj","fc1","lin1","wi","mlp_fc1","intermediate_dense","mlp.fc1","intermediate.dense"]
            lname_l = (lname or "").lower()
            return any(k in lname_l for k in keys)

        # 4) Trim sparsity picks to match the NS budget per layer
        sel_indices = {}
        for m in model.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            lname = layer_name_map.get(m, None)
            if lname is None or not _is_fc1(lname):
                continue
            sparsity_idx = sparsity_active.get(lname, None)
            k = int(k_map.get(lname, 0))
            if sparsity_idx is None or k <= 0:
                continue
            # Trim to match budget
            keep = min(k, len(sparsity_idx))
            sel_indices[lname] = sparsity_idx[:keep]

        # 5) Transform with STTTransformer and freeze everything except lm_head
        nst = STTTransformer(
            model, active_neurons=sel_indices,
            layer_name_map=layer_name_map,
            tune_pruned=False, device=device, verbose=True
        )
        model = nst.transform().to(device)
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            if any(key in name for key in ["lm_head", "classifier", "score", "pooler", "out_proj"]):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
                print(f"[ABLT2] trainable head: {name}")

        print("[ABLT2] Activation-rate-based (budget-matched) model ready; proceed to train/eval as usual.")
    elif args.mode == "transfer":
        src_name = args.source_task
        src_topk = args.source_ratio

        # print(f"[transfer] source={src_name}  ρ={src_topk:.2f}  sample_ratio={sel_ratio}")

        # 创建源任务的 data_loader
        if src_name == 'clutrr':
            data_loader = CLUTRR(
                split_dir=dataset_paths['clutrr'],
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'arc-e':
            data_loader = ARC(
                subset="easy",
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'arc-c':
            data_loader = ARC(
                subset="challenge",
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        elif src_name == 'boolq':
            data_loader = BoolQ(
                chat_template=args.apply_chat_template,
                model_name=args.model_name
            )
        else:
            raise ValueError(f"Unsupported source task: {src_name}")

        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        src_active_dl = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)
        tracker_src = NeuronTracker(
            model=model,
            tokenizer=tokenizer,                
            threshold=args.active_threshold,
            topk_ratio=args.source_ratio,                
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )
        active_src   = tracker_src.get_active_indices(dataloader=src_active_dl) or {}
        layer_map_src = tracker_src.get_layer_name_map()

        nst = STTTransformer(
            model=model,
            active_neurons=active_src,
            layer_name_map=layer_map_src,
            verbose=True,
            tune_pruned=False,
            device=device
        )
        model = nst.transform().to(device)
        print("[transfer] model pruned by SOURCE-selected subnetwork; downstream stays unchanged.")

        # 4) （可选）记录统计/开销（不影响训练逻辑）
        try:
            stats = nst.get_parameter_stats()
            print(f"[ns] overall_reduction = {stats['overall_model_reduction_perc']:.2f}%")
            sel_samples = len(src_active_dl.dataset)
            F_fwd_base  = flops_forward(model, _inp1_fwd, device=str(device))
            print(f"[FLOPs] source-selection forward ({sel_samples} ex): {sel_samples*F_fwd_base/1e9:.2f} GFLOPs")
        except Exception:
            pass
    
    elif args.mode == "mag_pt":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )
        # 2. NeuronTracker Initialization
        print("Initializing NeuronTracker...")
        tracker = NeuronTracker(
            model=model,  # Assuming 'model' is the loaded original model
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        # 3. Tracker Usage
        print("Running activation tracking...")
        layer_map = tracker.get_layer_name_map()
        active_indices_dict = None
        active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
        if active_indices_dict is None:
            active_indices_dict = {}
        print(f"Tracking complete. Found active indices for "
              f"{len(active_indices_dict) - 1 if active_indices_dict and '_attn_proj_layers' in active_indices_dict else (len(active_indices_dict) if active_indices_dict else 0)} layers.")

        k_map = {}
        for ln, idx in active_indices_dict.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
                k_map[ln] = int(len(idx))

        mag_indices = {}
        for m in model.modules():
            if isinstance(m, nn.Linear):
                lname = layer_map.get(m, None)
                if lname is None:
                    continue
                if not any(tag in lname for tag in ["gate_proj", "wi", "lin1"]):
                    continue
                k = k_map.get(lname, 0)
                if k <= 0:
                    continue
                # weight: [out_features, in_features] → L1 per output unit
                score = m.weight.detach().abs().sum(dim=1)
                keep = min(k, score.numel())
                topk_idx = torch.topk(score, keep, largest=True).indices
                mag_indices[lname] = topk_idx

        # 3) structural pruning into a neuroselective subnetwork
        nst = STTTransformer(
            model=model,
            active_neurons=mag_indices,
            layer_name_map=layer_map,
            tune_pruned=False,
            device=device,
            verbose=True
        )
        model = nst.transform().to(device)

        # 4) freeze all, then unfreeze pruned-MLP (STTLinear) + output head
        for p in model.parameters():
            p.requires_grad = False

        # heads typically named "lm_head" (plus a few common fallbacks)
        trainable_params_list = []
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, STTLinear) or ('lm_head' in name):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params_list.append(p)
        if not trainable_params_list:
            raise RuntimeError("[mag_pt] no STTLinear/lm_head found as trainables.")
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        # 5) accounting + fresh optimizer (downstream code expects `optimizer`)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[mag_pt] Final model parameters: {final_param_count:,}")
        print(f"[mag_pt] Final trainable parameters: {final_trainable_params:,}")
    elif args.mode == "wanda_p":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch)

        sel_stats = estimate_flops_infer(
            model=model,                
            data=active_dataloader,     
            modality="llm",
            tokenizer=tokenizer,              
            exclude_embeddings=True
        )

        print("Initializing NeuronTracker (budget)...")
        tracker_budget = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio,           
            use_abs_threshold=args.use_abs_threshold,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )

        print("Running activation tracking for budget...")
        layer_map = tracker_budget.get_layer_name_map()
        active_indices_dict = tracker_budget.get_active_indices(dataloader=active_dataloader) or {}
        print(f"[wanda_p] active layers (raw): {len(active_indices_dict)}")

        k_map = {}
        for ln, idx in active_indices_dict.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
                k_map[ln] = int(len(idx))

        print("Running Wanda ranking (L1  activation)...")
        tracker_w = NeuronTracker(
            model=model,
            tokenizer=tokenizer,
            topk_ratio=1.0,
            device=device,
            track_attention_proj=args.tune_attn,
            verbose=True
        )
        wanda_calib_batches = int(getattr(args, "wanda_calib_batches", 1))
        wanda_indices_all = tracker_w.get_wanda_indices(
            dataloader=active_dataloader,
            scan_batches=wanda_calib_batches
        )

        wanda_indices = {}
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = layer_map.get(m, None)
            if lname is None:
                continue
            if not any(tag in lname for tag in ["gate_proj", "wi", "lin1"]):
                continue  

            idx = wanda_indices_all.get(lname, None)
            if idx is None or len(idx) == 0:
                continue

            k = int(k_map.get(lname, 0))
            if k <= 0:
                continue

            keep = min(k, len(idx)) 
            topk_idx = torch.as_tensor(idx[:keep], dtype=torch.long)
            wanda_indices[lname] = topk_idx

        for ln, ids in wanda_indices.items():
            assert len(ids) == min(k_map.get(ln, 0), len(wanda_indices_all.get(ln, []))), \
                f"[wanda_p] budget mismatch @ {ln}: want={k_map.get(ln,0)} got={len(ids)}"

        nst = STTTransformer(
            model=model,
            active_neurons=wanda_indices,
            layer_name_map=layer_map,
            tune_pruned=False,
            device=device,
            verbose=True
        )
        model = nst.transform().to(device)

        for p in model.parameters():
            p.requires_grad = False

        trainable_params_list = []
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, STTLinear) or ('lm_head' in name):
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params_list.append(p)

        if not trainable_params_list:
            raise RuntimeError("[wanda_p] no STTLinear/lm_head found as trainables.")

        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[wanda_p] Final model parameters: {final_param_count:,}")
        print(f"[wanda_p] Final trainable parameters: {final_trainable_params:,}")
    elif args.mode == "wanda_tp":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)['text']
        active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch, shuffle=False)
        tracker = NeuronTracker(model=model, tokenizer=tokenizer,
                             threshold=args.active_threshold, topk_ratio=args.topk_ratio,
                             use_abs_threshold=args.use_abs_threshold, device=device,
                             track_attention_proj=args.tune_attn, verbose=True)
        layer_map = tracker.get_layer_name_map()
        active_indices = tracker.get_active_indices(dataloader=active_dataloader) or {}
        k_map = {ln: int(getattr(idx, "numel", lambda: len(idx))()) for ln, idx in active_indices.items()}
        mag_tp_layer_map = layer_map
        mag_tp_k_map = k_map
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    elif args.mode == "mag_tp":
        sampled_subset = data_loader.get_active_set(args.active_sample_ratio)
        sampled_subset = sampled_subset['text']
        active_dataloader = torch.utils.data.DataLoader(
             sampled_subset, batch_size=args.eval_batch, shuffle=False
         )
         # （可选）记录 selection FLOPs，接口与 mag_pt 一致
        _ = estimate_flops_infer(
             model=model, data=active_dataloader, modality="llm",
             tokenizer=tokenizer, exclude_embeddings=True
        )
         # 跟 mag_pt 相同的 tracker 配置，但这里只做 k-map
        tracker = NeuronTracker(
             model=model, tokenizer=tokenizer,
             threshold=args.active_threshold, topk_ratio=args.topk_ratio,
             use_abs_threshold=args.use_abs_threshold, device=device,
             track_attention_proj=args.tune_attn, verbose=True
        )
        layer_map = tracker.get_layer_name_map()
        active_indices = tracker.get_active_indices(dataloader=active_dataloader) or {}
        k_map = {}
        for ln, idx in active_indices.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
               k_map[ln] = int(len(idx))
        mag_tp_k_map = k_map
        mag_tp_layer_map = layer_map
        print(f"[mag_tp][pre] captured budget for {len(k_map)} layers, e.g. {list(k_map.items())[:3]}")
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.wd)

    elif args.mode =="lora":
        print(f"--- Setting up {args.mode} ---")
        lora_config_dict = {
            'type': args.mode.lower(),           
            'r': args.lora_r,
            'alpha': args.lora_alpha,            
            'dropout': args.lora_dropout,        
            'bias': 'none',
            'task_type': 'CAUSAL_LM',
        }
        model = setup_lora(model, lora_config_dict)  # Assumes setup_lora handles different PEFT types
        model = model.to(device)
        from peft.tuners.lora import LoraLayer
        base_dt = next(p for p in model.parameters() if p.is_floating_point()).dtype
        print("[PEFT] base:", base_dt, "adapters:", {next(iter(m.lora_A.values())).weight.dtype for m in model.modules() if isinstance(m, LoraLayer)})

    else:
        print(f"--- Running in {args.mode} mode (full finetuning) ---")
        model = model.to(device)
        print(model.dtype)   

    final_param_count = sum(p.numel() for p in model.parameters())
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.mode in ("stt", "transfer","activation_mean_value", "activation_rate"):
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, STTLinear) or 'lm_head' in name:
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params_list.append(param)
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)
    elif args.mode == "stt_lora":
        if isinstance(model.lm_head, torch.nn.Linear):
            head_cfg = LoraConfig(
                r=args.lora_r,          
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["lm_head"],   
                bias="none",
            )
            model = get_peft_model(model, head_cfg)        

        # For STTLoraLinear, we only train the LoRA parameters
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Enable training for STTLoraLinear parameters and lm_head
        for name, module in model.named_modules():
            if isinstance(module, STTLoraLinear) or 'lm_head' in name:
                for param_name, param in module.named_parameters():
                    if 'lora_A' in param_name or 'lora_B' in param_name:
                        param.requires_grad = True
                        trainable_params_list.append(param)
        if not trainable_params_list:
            raise RuntimeError("no lora is found, check")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trainable_params_list)
        print(f"[NS-LoRA] trainable: {trainable_params:,} / {total_params:,}  "
            f"({trainable_params / total_params * 100:.3f}%)")

        print("lora list:")
        for name, param in model.named_parameters():
            if param.requires_grad:          
                print(f"  {name:60s}  {tuple(param.shape)}")
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)
    else:
        if not use_deepspeed:
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.wd
            )

    print("\n--- Parameter Counts ---")
    print(f"Original Total Params:      {orig_param_count:,}")
    print(f"Original Trainable Params:  {orig_trainable_params:,}")
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_total_params     = sum(p.numel() for p in model.parameters())
    print(f"Final Total Params:         {final_total_params:,}")
    print(f"Final Trainable Params:     {final_trainable_params:,}")
    if orig_trainable_params > 0:
        reduction = (1 - final_trainable_params / orig_trainable_params) * 100
        print(f"Trainable Param Reduction:  {reduction:.2f}%")
    else:
        print("No original trainable parameters to compare.")
    print("-" * 26)

    if args.use_wandb:
        log_data = {
            'params/original_total': orig_param_count,
            'params/original_trainable': orig_trainable_params,
            'params/final_total': final_param_count,
            'params/final_trainable': final_trainable_params,
        }
        if orig_trainable_params > 0:
            log_data['params/trainable_reduction_pct'] = (1 - final_trainable_params / orig_trainable_params) * 100
        if pruner_stats:
            log_data.update({f'pruner/{k}': v for k, v in pruner_stats.items() if not isinstance(v, (dict, list))})
            if 'overall_model_reduction_perc' in pruner_stats:
                log_data['params/overall_reduction_pct'] = pruner_stats['overall_model_reduction_perc']

        wandb.log(log_data)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="no",
        lr_scheduler_type="linear" if args.schedule else "constant",
        load_best_model_at_end=False,
        max_length=256,
        dataset_text_field="text",
        bf16=True,
        warmup_ratio=0.1,      
        learning_rate=args.lr,
        weight_decay=args.wd,                 
        max_grad_norm=0.5,                               
        deepspeed=(args.deepspeed if use_deepspeed else None),
    )
    if args.task == "clutrr":
        eval_dataset = datasets['val']
    else:
        eval_dataset = test_dataset

    trainer = CustomSFTTrainerV2(
        task=args.task,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    if args.use_wandb:
        wandb.log({"status": "training_started"})
    
    try:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=False
        )
        tr_stats = estimate_flops_train(
            model=model, data=train_dataloader, modality="llm",
            epochs=args.num_epoch, tokenizer=tokenizer
        )
        print(f"[FLOPs] Train (planned) ≈ {tr_stats['flops']/1e15:.3f} PFLOPs "
              f"[FLOPs] Train (planned) ≈ {tr_stats['flops']/1e12:.3f} TFLOPs "
              f"(N={tr_stats['N']:,}, MAX_TOKEN=2048, epochs={args.num_epoch})")
        if args.use_wandb:
            wandb.log({
                "flops/train_total": tr_stats["flops"],
                "flops/train_total_tflops": tr_stats["flops"]/1e12,
                "flops/train_total_pflops": tr_stats["flops"]/1e15,
            })
    except Exception as e:
        logging.warning(f"[FLOPs] Train estimation failed: {e}")

    print("Starting training...")
    trainer.train()


    if args.use_wandb:
        wandb.log({"status": "training_finished", "evaluating": True})

    accuracy, predictions = trainer.test(
        fname=os.path.join(args.output_dir, run_name),
        task=args.task,
        eval_dataset=test_dataset,
        model_name=args.model_name,
        apply_chat_template=args.apply_chat_template,
    )
    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")

    # NEW: Evaluate with restored neurons
    if args.mode in ["stt", "stt_lora"]:  # Only for neuron selection modes
        print("\n" + "="*60)
        print("EXPERIMENT: Evaluating with restored neurons")
        print("="*60)
        
        # For STT_LoRA: Merge first, then restore (better approach)
        if args.mode == "stt_lora":
            print("Phase 1: Merging LoRA adapters...")
            merged_count = 0
            for name, module in model.named_modules():
                if isinstance(module, STTLoraLinear):
                    if not module.merge_weights:  # Only merge if not already merged
                        module.merge()
                        merged_count += 1
                        print(f"  Merged adapters for {name}")
            print(f"Total merged STTLoraLinear modules: {merged_count}")
            
            print("Phase 2: Restoring pruned neurons...")
            restored_count = 0
            for name, module in model.named_modules():
                if isinstance(module, STTLoraLinear):
                    # Restore the underlying STTLinear to full model
                    module.ns_linear.restore_full_weights()
                    restored_count += 1
                    print(f"  Restored neurons for {name}")
            print(f"Total restored STTLoraLinear modules: {restored_count}")
            print("Merging and restoration complete. Now proceeding with evaluation...")
        
        # Create a simple evaluation function for LLM
        def evaluate_llm_with_restored(model, eval_dataset, task, model_name, apply_chat_template):
            # Store original forward methods
            original_forwards = {}
            
            # Replace forward methods for STTLinear layers
            for name, module in model.named_modules():
                if isinstance(module, STTLinear):
                    original_forwards[name] = module.forward
                    module.forward = module.forward_full
                elif isinstance(module, STTLoraLinear):
                    # For STTLoraLinear, use the underlying STTLinear's forward_full
                    original_forwards[name] = module.forward
                    module.forward = module.ns_linear.forward_full
            
            try:
                # Run evaluation using the same trainer.test method
                restored_accuracy, restored_predictions = trainer.test(
                    fname=os.path.join(args.output_dir, run_name + "_restored"),
                    task=task,
                    eval_dataset=eval_dataset,
                    model_name=model_name,
                    apply_chat_template=apply_chat_template,
                )
                return restored_accuracy, restored_predictions
            finally:
                # Restore original forward methods
                for name, module in model.named_modules():
                    if (isinstance(module, STTLinear) or isinstance(module, STTLoraLinear)) and name in original_forwards:
                        module.forward = original_forwards[name]
        
        restored_accuracy, restored_predictions = evaluate_llm_with_restored(
            model, test_dataset, args.task, args.model_name, args.apply_chat_template
        )
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Pruned model accuracy:  {accuracy:.4f}")
        print(f"Restored model accuracy: {restored_accuracy:.4f}")
        print(f"Improvement:            {restored_accuracy - accuracy:+.4f}")
        print(f"Relative improvement:   {((restored_accuracy - accuracy) / accuracy * 100):+.2f}%")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "experiment/pruned_accuracy": accuracy,
                "experiment/restored_accuracy": restored_accuracy,
                "experiment/improvement_abs": restored_accuracy - accuracy,
                "experiment/improvement_rel": (restored_accuracy - accuracy) / accuracy * 100,
            })
    eval_dataloader = DataLoader(
            test_dataset, batch_size=args.eval_batch, shuffle=False
        )

    inf_unmerged = estimate_flops_infer(
        model=model,
        data=eval_dataloader,   
        modality="llm",
        tokenizer=tokenizer,   
        exclude_embeddings=True
    )

    # ---- Eval FLOPs ----
    total_eval_flops = inf_unmerged["flops"]
    N_eval = max(inf_unmerged["N"], 1)
    assert inf_unmerged["D"] == 2048 * inf_unmerged["N"], \
        "Eval D != N*2048 — check dataloader/iterator & tokenizer."
    print(f"[FLOPs] Inference (eval, unmerged) ≈ {total_eval_flops/1e15:.3f} PFLOPs "
          f"({total_eval_flops/1e12:.3f} TFLOPs); "
          f"per-example ≈ {(total_eval_flops/N_eval)/1e9:.3f} GFLOPs "
          f"(L=2048, N={inf_unmerged['N']:,})")
    if args.use_wandb:
        wandb.log({
            "flops/eval_total": total_eval_flops,
            "flops/eval_total_pflops": total_eval_flops/1e15,
            "flops/eval_total_tflops": total_eval_flops/1e12,
            "flops/eval_per_example_gflops": (total_eval_flops/N_eval)/1e9,
            "tokens/eval_D": inf_unmerged["D"],
            "tokens/eval_L": 2048,
            "size/eval_N": inf_unmerged["N"],
        })





    if args.use_wandb:
        wandb.log({
            "flops/infer_eval_unmerged_2ND": inf_unmerged["flops"],
            "tokens/eval_D": inf_unmerged["D"],
            "params/nonembed_postprune": inf_unmerged["N"],
        })

    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Num predictions: {len(predictions)}")
    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Num predictions: {len(predictions)}")
    
    if args.use_wandb:
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/num_predictions": len(predictions)
        })

    if args.mode in ("mag_tp", "wanda_tp"):
        k_map = mag_tp_k_map or {}
        layer_map = mag_tp_layer_map or {}
        if not k_map or not layer_map:
            raise RuntimeError("[mag_tp] Budget map not found; ensure pre-phase ran.")
        prune_indices = {}
        if args.mode == "wanda_tp":
          sampled_subset = data_loader.get_active_set(args.active_sample_ratio)['text']
          active_dataloader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.eval_batch, shuffle=False)
          tracker_w = NeuronTracker(model=model, tokenizer=tokenizer, topk_ratio=1.0,
                                   device=device, track_attention_proj=args.tune_attn, verbose=True)
          wanda_all = tracker_w.get_wanda_indices(
              dataloader=active_dataloader,
              scan_batches=int(getattr(args, "wanda_calib_batches", 1))
          )
          sel_indices = {}
          for m in model.modules():
              if isinstance(m, nn.Linear):
                  lname = layer_map.get(m, None)
                  if lname and any(t in lname for t in ["gate_proj","up_proj","wi","fc1","lin1"]):
                      k = int(k_map.get(lname, 0))
                      idx = wanda_all.get(lname, [])[:max(0, k)]
                      if k > 0 and len(idx) > 0:
                          prune_indices[lname] = torch.as_tensor(idx, dtype=torch.long)

        else:
            print(f"[mag_tp] Using budget for {len(k_map)} layers")
            print("[mag_tp] Computing magnitude-based pruning indices...")
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    lname = layer_map.get(m, None)
                    if lname is None or not any(tag in lname for tag in ["gate_proj","up_proj","wi","fc1","lin1"]):
                        continue
                    k = int(k_map.get(lname, 0))
                    if k <= 0:
                        continue
                    score = m.weight.detach().abs().sum(dim=1)  # [out_features]
                    keep = min(k, score.numel())
                    topk_idx = torch.topk(score, keep, largest=True).indices
                    prune_indices[lname] = topk_idx
                    print(f"[mag_tp]   {lname}: keeping {keep}/{score.numel()} neurons")
            
            print(f"[mag_tp] Computed indices for {len(prune_indices)} layers")
            
        # Step 3: Apply structural pruning
        print("[mag_tp] Applying structural pruning...")
        device_ = next(p for p in model.parameters() if p.is_floating_point()).device
        nst = STTTransformer(
            model=model, 
            active_neurons=prune_indices,
            layer_name_map=layer_map, 
            tune_pruned=False,
            device=device_, 
            verbose=True
        )
        pruned_model = nst.transform().to(device_)
        
        # Update model reference
        model = pruned_model
        trainer.model = model  # Update trainer's model reference
        
        # Freeze all parameters for one-shot evaluation
        for p in model.parameters():
            p.requires_grad = False
            
        print(f"[mag_tp] Pruned model params: {sum(p.numel() for p in model.parameters()):,}, "
              f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Step 4: One-shot evaluation on pruned model (before recovery)
        print("\n[mag_tp] === PHASE 2: One-shot Evaluation on Pruned Model ===")
        pruned_run_name = run_name + "_pruned_oneshot"
        accuracy_pruned, predictions_pruned = trainer.test(
            fname=os.path.join(args.output_dir, pruned_run_name),
            task=args.task,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            apply_chat_template=args.apply_chat_template,
        )
        
        # Measure pruned model inference FLOPs
        eval_dataloader_pruned = DataLoader(test_dataset, batch_size=args.eval_batch, shuffle=False)
        inf_pruned = estimate_flops_infer(
            model=model,
            data=eval_dataloader_pruned,
            modality="llm",
            tokenizer=tokenizer,
            exclude_embeddings=True
        )
        total_eval_flops_pruned = inf_pruned["flops"]
        N_eval_pruned = max(inf_pruned["N"], 1)
        
        print(f"[FLOPs][pruned-oneshot] Inference ≈ {total_eval_flops_pruned/1e15:.3f} PFLOPs "
              f"({total_eval_flops_pruned/1e12:.3f} TFLOPs); "
              f"per-example ≈ {(total_eval_flops_pruned/N_eval_pruned)/1e9:.3f} GFLOPs")
        print(f"[mag_tp][pruned-oneshot] Accuracy: {accuracy_pruned:.4f}, "
              f"Predictions: {len(predictions_pruned)}")
        
        if args.use_wandb:
            wandb.log({
                "eval_pruned_oneshot/accuracy": accuracy_pruned,
                "eval_pruned_oneshot/num_predictions": len(predictions_pruned),
                "flops/eval_pruned_oneshot_total": total_eval_flops_pruned,
                "flops/eval_pruned_oneshot_total_pflops": total_eval_flops_pruned/1e15,
                "flops/eval_pruned_oneshot_total_tflops": total_eval_flops_pruned/1e12,
                "flops/eval_pruned_oneshot_per_example_gflops": (total_eval_flops_pruned/N_eval_pruned)/1e9,
            })
        
        # ===== ENHANCED MEMORY CLEANUP BEFORE RECOVERY =====
        print("\n[mag_tp] === Memory Cleanup Before Recovery ===")
        
        # Clear evaluation objects
        try:
            del eval_dataloader_pruned, inf_pruned, predictions_pruned, accuracy_pruned
        except Exception as e:
            print(f"[mag_tp] Warning: cleanup of eval objects failed: {e}")
        
        # Clear original trainer state thoroughly
        try:
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                trainer.optimizer.state.clear()
                del trainer.optimizer
                trainer.optimizer = None
            
            if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None:
                del trainer.lr_scheduler
                trainer.lr_scheduler = None
                
            # Clear trainer caches
            if hasattr(trainer, 'state'):
                trainer.state = None
            if hasattr(trainer, 'train_dataloader'):
                trainer.train_dataloader = None
            if hasattr(trainer, 'eval_dataloader'):
                trainer.eval_dataloader = None
                
            del trainer
        except Exception as e:
            print(f"[mag_tp] Warning: trainer cleanup failed: {e}")
        
        # Clear original optimizer if exists
        try:
            if 'optimizer' in locals() and optimizer is not None:
                optimizer.state.clear()
                del optimizer
        except Exception as e:
            print(f"[mag_tp] Warning: optimizer cleanup failed: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"[mag_tp] GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # ===== RECOVERY TRAINING PHASE =====
        print("\n[mag_tp] === PHASE 3: Recovery Fine-tuning ===")
        
        # Enable gradients only for pruned layers and output head
        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        for name, module in model.named_modules():
            if isinstance(module, STTLinear) or ('lm_head' in name):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params_list.append(param)
        
        if not trainable_params_list:
            raise RuntimeError("[mag_tp] No trainable parameters found for recovery")
        
        print(f"[mag_tp][recovery] Trainable parameters: {sum(p.numel() for p in trainable_params_list):,}")
        
        # Create optimizer with memory-efficient settings
        optimizer_recovery = AdamW(
            trainable_params_list, 
            lr=args.recovery_lr,  # Use recovery_lr instead of lr
            weight_decay=args.wd,
            foreach=False  # Disable multi-tensor operations to save memory
        )

        # Configure training args with memory optimization
        recovery_args = SFTConfig(
            output_dir=os.path.join(args.output_dir, run_name + "_pruned_recovery"),
            num_train_epochs=1,
            per_device_train_batch_size=max(1, args.train_batch // 2),  # Reduce batch size
            per_device_eval_batch_size=args.eval_batch,
            seed=args.seed,
            eval_strategy="no",
            save_strategy="no",
            lr_scheduler_type="linear" if args.schedule else "constant",
            load_best_model_at_end=False,
            max_length=256,
            dataset_text_field="text",
            bf16=True,
            warmup_ratio=0.0,
            learning_rate=args.recovery_lr,
            weight_decay=args.wd,
            max_grad_norm=0.5,
            gradient_accumulation_steps=args.gradient_accumulation_steps * 2, 
            dataloader_pin_memory=False, 
            deepspeed=(args.deepspeed if use_deepspeed else None),
        )

        # Create recovery trainer
        recovery_trainer = CustomSFTTrainerV2(
            task=args.task,
            model=model,
            args=recovery_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer_recovery, None) if (not use_deepspeed) else (None, None)
        )

        # Optional: Estimate FLOPs for recovery training
        try:
            train_dataloader_rec = DataLoader(train_dataset, batch_size=recovery_args.per_device_train_batch_size, shuffle=False)
            tr_stats_rec = estimate_flops_train(
                model=model, data=train_dataloader_rec, modality="llm",
                epochs=1, tokenizer=tokenizer
            )
            print(f"[FLOPs][recovery] Train ≈ {tr_stats_rec['flops']/1e15:.3f} PFLOPs "
                f"({tr_stats_rec['flops']/1e12:.3f} TFLOPs); "
                f"N={tr_stats_rec['N']:,}, MAX_TOKEN=2048, epochs=1")
            if args.use_wandb:
                wandb.log({
                    "flops/train_recovery_total": tr_stats_rec["flops"],
                    "flops/train_recovery_total_pflops": tr_stats_rec["flops"]/1e15,
                    "flops/train_recovery_total_tflops": tr_stats_rec["flops"]/1e12,
                })
        except Exception as e:
            logging.warning(f"[FLOPs][recovery] Train estimation failed: {e}")

        # Start recovery training
        if args.use_wandb:
            wandb.log({"status": "recovery_training_started"})
        
        recovery_trainer.train()
        
        if args.use_wandb:
            wandb.log({"status": "recovery_training_finished"})

        pruned_rec_run_name = run_name + "_pruned_recovery"
        accuracy_pruned_rec, predictions_pruned_rec = recovery_trainer.test(
            fname=os.path.join(args.output_dir, pruned_rec_run_name),
            task=args.task,
            eval_dataset=test_dataset,
            model_name=args.model_name,
            apply_chat_template=args.apply_chat_template,
        )
        eval_dataloader_pruned_rec = DataLoader(test_dataset, batch_size=args.eval_batch, shuffle=False)
        inf_unmerged_pruned_rec = estimate_flops_infer(
            model=model, data=eval_dataloader_pruned_rec, modality="llm",
            tokenizer=tokenizer, exclude_embeddings=True
        )
        total_eval_flops_pruned_rec = inf_unmerged_pruned_rec["flops"]
        N_eval_pruned_rec = max(inf_unmerged_pruned_rec["N"], 1)
        print(f"[FLOPs][pruned+recovery] Inference (eval, unmerged) ≈ "
              f"{total_eval_flops_pruned_rec/1e15:.3f} PFLOPs "
              f"({total_eval_flops_pruned_rec/1e12:.3f} TFLOPs); "
              f"per-example ≈ {(total_eval_flops_pruned_rec/N_eval_pruned_rec)/1e9:.3f} GFLOPs "
              f"(L=2048, N={inf_unmerged_pruned_rec['N']:,})")
        print(f"[mag_tp][recovery] accuracy={accuracy_pruned_rec:.4f}, "
              f"num_predictions={len(predictions_pruned_rec)}")
        if args.use_wandb:
            wandb.log({
                "eval_pruned_recovery/accuracy": accuracy_pruned_rec,
                "eval_pruned_recovery/num_predictions": len(predictions_pruned_rec),
                "flops/eval_pruned_recovery_total": total_eval_flops_pruned_rec,
                "flops/eval_pruned_recovery_total_pflops": total_eval_flops_pruned_rec/1e15,
                "flops/eval_pruned_recovery_total_tflops": total_eval_flops_pruned_rec/1e12,
                "flops/eval_pruned_recovery_per_example_gflops": (total_eval_flops_pruned_rec/N_eval_pruned_rec)/1e9,
                "tokens/eval_pruned_recovery_D": inf_unmerged_pruned_rec["D"],
                "tokens/eval_pruned_recovery_L": 2048,
                "size/eval_pruned_recovery_N": inf_unmerged_pruned_rec["N"],
            })

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
