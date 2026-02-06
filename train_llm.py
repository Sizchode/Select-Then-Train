import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
import wandb
import numpy as np
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
from stt.wanda_adapt_tracker import WandaAdaptTracker
from peft import get_peft_model, LoraConfig
from stt.trainers import CustomSFTTrainerV2
from stt.stt_lora import STTLoraLinear
from util.utils import (
    set_seed,
    bench_forward,
    pad_all_nslinear_modules,
    clear_cuda_cache_and_states,
)
from util.torch_flops import (
    estimate_flops_infer, estimate_flops_train,
)
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Hidden Dimension Pruning Training for LLMs')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--task', "--dataset", type=str,
                        choices=['clutrr', 'boolq', 'arc-e', 'arc-c'],
                        default="clutrr",
                        help='Task/dataset to evaluate on')
    parser.add_argument('--mode', type=str,
                        choices=["baseline", "magnitude_pruning", "wanda_adapt", "stt", "stt_lora"],
                        default="stt_lora", 
                        help="Training mode: 'stt' for neuron selection, 'stt_lora' for STT+LoRA, 'magnitude_pruning' for magnitude-based pruning, 'wanda_adapt' for Wanda adaptive pruning")
    parser.add_argument('--apply_chat_template',
                        action='store_true',
                        help='Using chat template in the prompt; False by default')
    parser.add_argument('--lr', "--learning_rate", type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--train_batch', "--batch_size", type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_epoch', "--num_epochs", type=int, default=10, help='Number of epochs to fine-tune')
    parser.add_argument('--train_size', type=int, default=0, help='Number of training examples to use (0 for all)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--active_threshold', type=float, default=0.01,
                        help='Activation threshold for finding active neurons')
    parser.add_argument('--active_sample_ratio', type=float, default=0.1,
                        help='Sample ratio of training data for activation tracking')
    parser.add_argument("--topk_ratio",type=float, default=0.30, metavar="R",
                        help="Fraction (0,1] of neurons kept per score when selecting active neurons")
    parser.add_argument('--lora_r', type=int, default=16,
                        help='Rank of the LoRA updates')
    parser.add_argument('--lora_alpha', type=float, default=32,
                        help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='Dropout rate for LoRA layers')
    parser.add_argument('--output_dir', default='outputs', type=str,  # Example path
                        help='Output directory for checkpoints and outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='hidden-dim-pruning-llm', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Log to WandB every N steps')
    parser.add_argument("--schedule", action="store_true", help="enable linear schedule")
    parser.add_argument("--tune_attn", action="store_true", help="tune attn proj during tracking (if tracker supports)")
    parser.add_argument('--dev_mode', action='store_true',
                        help='Use a held-out dev set from training set for validation (grid search); do NOT use real test set')
    args = parser.parse_args()
    set_seed(args.seed)
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
        response_template_with_context = "\n"  

    if args.apply_chat_template:
        response_template_with_context = \
            tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True).split('\n')[-1]  
        if not response_template_with_context: response_template_with_context = " [/INST]"  

    offset = 1  
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[offset:]
    model = model.to(device)
    optimizer = None
    pruner_stats = None
    if args.mode == "stt" or args.mode == "stt_lora":
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
        print("Initializing STT Tracker...")
        tracker = NeuronTracker(
            model=model,  
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            device=device,
            verbose=True
        )

        # 3. Tracker Usage
        print("Running activation tracking...")
        layer_map = tracker.get_layer_name_map()
        active_indices_dict = None
        active_indices_dict = tracker.get_active_indices(dataloader=active_dataloader)
        # composition_stats = tracker.compute_dual_metric_composition(delta=args.active_threshold)
        # tracker.visualize_layer_composition(
        #     composition_stats,
        #     model_name=f"{args.model_name}_topk{args.topk_ratio}",
        #     dataset_name=args.task
        # )
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
            device=device,
            inference_time=False  
        )
        print("Performing model transformation (pruning)...")
        pruned_model = transformer.transform()
        pruned_model = pruned_model.to(device)
        model = pruned_model
        print("Model variable now points to the pruned model.")

        if args.mode == "stt_lora":
            print("--- Applying STTLoraLinear to STTLinear layers ---")
            for name, module in model.named_modules():
                if isinstance(module, STTLinear):
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    attr_name = name.split('.')[-1]
                    # Get parent module
                    parent = model
                    if parent_name:
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)
                    ns_lora = STTLoraLinear(
                        stt_linear=module,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        merge_weights=False
                    )
                    setattr(parent, attr_name, ns_lora)

            print("STTLoraLinear transformation complete.")
            _bp = next(p for p in model.parameters() if p.is_floating_point())  
            for m in model.modules():  
                if isinstance(m, STTLoraLinear):  
                    m.lora_A.to(device=_bp.device, dtype=_bp.dtype)  
                    m.lora_B.to(device=_bp.device, dtype=_bp.dtype) 
                    m.scaling = torch.as_tensor(m.scaling, device=_bp.device, dtype=_bp.dtype)  
            print("[NSLoRA] base:", _bp.dtype, "adapters:", {m.lora_A.weight.dtype for m in model.modules() if isinstance(m, STTLoraLinear)})  
    elif args.mode == "magnitude_pruning":
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
            model=model, 
            tokenizer=tokenizer,
            threshold=args.active_threshold,
            topk_ratio=args.topk_ratio, 
            device=device,
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
                score = m.weight.detach().abs().sum(dim=1)
                keep = min(k, score.numel())
                topk_idx = torch.topk(score, keep, largest=True).indices
                mag_indices[lname] = topk_idx

        nst = STTTransformer(
            model=model,
            active_neurons=mag_indices,
            layer_name_map=layer_map,
            device=device,
            verbose=True,
            inference_time=False  
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
            raise RuntimeError("[magnitude_pruning] no STTLinear/lm_head found as trainables.")
        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        # 5) accounting + fresh optimizer (downstream code expects `optimizer`)
        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[magnitude_pruning] Final model parameters: {final_param_count:,}")
        print(f"[magnitude_pruning] Final trainable parameters: {final_trainable_params:,}")
    elif args.mode == "wanda_adapt":
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
            device=device,
            verbose=True
        )

        print("Running activation tracking for budget...")
        layer_map = tracker_budget.get_layer_name_map()
        active_indices_dict = tracker_budget.get_active_indices(dataloader=active_dataloader) or {}
        print(f"[wanda_adapt] active layers (raw): {len(active_indices_dict)}")

        k_map = {}
        for ln, idx in active_indices_dict.items():
            try:
                k_map[ln] = int(idx.numel())
            except Exception:
                k_map[ln] = int(len(idx))

        print("Running Wanda ranking (L1  activation)...")
        tracker_w = WandaAdaptTracker(
            model=model,
            tokenizer=tokenizer,
            topk_ratio=1.0,
            device=device,
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
                f"[wanda_adapt] budget mismatch @ {ln}: want={k_map.get(ln,0)} got={len(ids)}"

        nst = STTTransformer(
            model=model,
            active_neurons=wanda_indices,
            layer_name_map=layer_map,
            device=device,
            verbose=True,
            inference_time=False
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
            raise RuntimeError("[wanda_adapt] no STTLinear/lm_head found as trainables.")

        optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.wd)

        final_param_count = sum(p.numel() for p in model.parameters())
        final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[wanda_adapt] Final model parameters: {final_param_count:,}")
        print(f"[wanda_adapt] Final trainable parameters: {final_trainable_params:,}")
    else:
        print(f"--- Running in {args.mode} mode (full finetuning) ---")
        model = model.to(device)
        print(model.dtype)   
    final_param_count = sum(p.numel() for p in model.parameters())
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.mode in ("stt"):
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
        if isinstance(model.lm_head, nn.Linear):
            head_cfg = LoraConfig(
                r=args.lora_r,          
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["lm_head"],   
                bias="none",
            )
            model = get_peft_model(model, head_cfg)        

        trainable_params_list = []
        for name, param in model.named_parameters():
            param.requires_grad = False

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

    # Only test stt and baseline modes
    is_stt_mode = args.mode == "stt"
    
    if is_stt_mode:
        print("\n[Evaluation] inference throughput for pruning...")
        clear_cuda_cache_and_states(
            optimizer=optimizer,
            trainer=trainer,
            model=model,
            device=device,
            verbose=True
        )
        modified_count = 0
        for name, module in model.named_modules():
            if isinstance(module, STTLinear):
                if 'gate_proj' in name or 'up_proj' in name:
                    module.inference_time = True
                    modified_count += 1
                elif 'down_proj' in name:
                    module.inference_time = False  
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Evaluation] Model switched to inference_time mode (gate/up only, down_proj remains False)")
        print(f"[Evaluation] Modified {modified_count} gate/up_proj modules to inference_time=True")
        
        print(f"\n[Padding] Applying padding 128 for True Pruning mode...")
        pad_all_nslinear_modules(model, pad_to=128)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[Padding] Padding applied successfully.")
        
    else:
        print("\n[Evaluation] Using baseline mode (no pruning)")

    def tokenize_collate_fn(batch):
        """Collate function that tokenizes text field from raw dataset"""
        texts = []
        for item in batch:
            if isinstance(item, dict):
                if 'prompt' in item:
                    texts.append(str(item['prompt']))
                elif 'text' in item:
                    texts.append(str(item['text']))
                else:
                    for key, value in item.items():
                        if isinstance(value, str):
                            texts.append(value)
                            break
                    else:
                        texts.append(str(item))
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
        
        tokenized = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        return tokenized
    
    eval_dataloader_bench = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch, 
        shuffle=False,
        collate_fn=tokenize_collate_fn
    ) if 'test_dataset' in locals() else None
    
    if eval_dataloader_bench is not None:
        mode_desc = "True Pruning mode (with padding 128)" if is_stt_mode else "Baseline mode"
        print(f"\n[Benchmark] Testing {mode_desc}...")
        throughput, latency_ms = bench_forward(model, eval_dataloader=eval_dataloader_bench, iters=200, warmup=20, device=device)
        print(f"  Throughput: {throughput:.2f} samples/sec (using real data)")
        print(f"  Latency:    {latency_ms:.2f} ms/batch")
    else:
        print(f"\n[Benchmark] Skipping benchmark (no test data available)")
    accuracy, predictions = trainer.test(
        fname=os.path.join(args.output_dir, run_name),
        task=args.task,
        eval_dataset=test_dataset,
        model_name=args.model_name,
        apply_chat_template=args.apply_chat_template,
    )
    print("\n[Evaluation Metrics]")
    print(f"  Accuracy: {accuracy:.4f}")
    unrestored_checkpoint = {
        'model_state_dict': model.state_dict(),
        'final_accuracy': accuracy,
        'args': vars(args)
    }
    torch.save(unrestored_checkpoint, os.path.join(args.output_dir, run_name + "_unrestored.pth"))
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
    if args.use_wandb:
        wandb.finish()
if __name__ == '__main__':
    main()
