import argparse
import os
import random
from peft import TaskType
import torch
from trl import DataCollatorForCompletionOnlyLM, SFTConfig
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import defaultdict

from stt.mlps.stt_linear2 import STTLinear
from stt.stt_lora import STTLoraLinear
# from diet.tracker2 import NeuronTracker

# Import from peft for LoRA
from peft import get_peft_model, LoraConfig, AdaLoraConfig
import re
import logging


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_model_layers(model):
    """Print layer structure of the model for debugging."""
    print("\nModel Layer Structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")
    print("\n")


def adapt_neuron_indices_for_transformer(tracker_indices, model):
    """
    Adapts indices from STTTracker format to STTTransformer format.

    Args:
        tracker_indices: Dictionary of indices from NeuronTracker
        model: The model being transformed

    Returns:
        Dictionary with keys properly formatted for STTTransformer
    """
    transformer_indices = {}

    # Get all linear layer names from the model
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)

    # Convert NeuronTracker keys to transformer keys
    for tracker_key, indices in tracker_indices.items():
        # Check if this is a layer_X_mlp_Y format key
        parts = tracker_key.split('_')
        if len(parts) >= 4 and parts[0] == 'layer' and 'mlp' in parts:
            layer_idx = parts[1]
            proj_type = parts[3]  # The projection type (e.g., gate_proj, intermediate.dense)

            # Find matching layer in the model
            matches = []
            for layer_name in linear_layers:
                # Check if the layer has both the index and projection type
                if f".{layer_idx}." in layer_name and proj_type in layer_name:
                    matches.append(layer_name)
                # Alternative format with brackets
                elif f"[{layer_idx}]" in layer_name and proj_type in layer_name:
                    matches.append(layer_name)

            if matches:
                # If multiple matches, take the shortest (most specific) one
                matches.sort(key=len)
                model_layer_name = matches[0]
                transformer_key = f"{model_layer_name}_out"  # Typically we want to prune outputs
                transformer_indices[transformer_key] = indices
                logging.info(f"Mapped {tracker_key} to {transformer_key}")
            else:
                # If no exact match, try a more flexible approach
                for layer_name in linear_layers:
                    if proj_type in layer_name:
                        # Check if there's a number in the layer name that matches our layer index
                        if re.search(r'\.{}\.'.format(layer_idx), layer_name) or \
                                re.search(r'\[{}\]'.format(layer_idx), layer_name):
                            transformer_key = f"{layer_name}_out"
                            transformer_indices[transformer_key] = indices
                            logging.info(f"Flexibly mapped {tracker_key} to {transformer_key}")
                            break
        else:
            # For non-standard keys, try direct mapping if possible
            for layer_name in linear_layers:
                if tracker_key in layer_name:
                    transformer_key = f"{layer_name}_out"
                    transformer_indices[transformer_key] = indices
                    logging.info(f"Direct mapped {tracker_key} to {transformer_key}")
                    break

    logging.info(f"Adapted {len(transformer_indices)}/{len(tracker_indices)} indices for transformer")
    return transformer_indices


class VisionEncoderWithClassifier(nn.Module):
    """Wrapper for vision encoders with classification head."""

    def __init__(self, vision_encoder, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        if hasattr(vision_encoder.config, 'hidden_size'):
            hidden_size = vision_encoder.config.hidden_size
        elif hasattr(vision_encoder.config, 'projection_dim'):
            hidden_size = vision_encoder.config.projection_dim
        elif hasattr(vision_encoder.config, 'vision_config') and hasattr(vision_encoder.config.vision_config,
                                                                         'hidden_size'):
            hidden_size = vision_encoder.config.vision_config.hidden_size
        else:
            hidden_size = 768

        # Add classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, **kwargs):
        if hasattr(self.vision_encoder, 'get_image_features'):
            features = self.vision_encoder.get_image_features(pixel_values)
        else:
            outputs = self.vision_encoder(pixel_values, **kwargs)
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = outputs[0][:, 0]

        logits = self.classifier(features)

        return type('obj', (object,), {'logits': logits})


def evaluate_classification(model, eval_dataloader, device, modality="image", description="Evaluating",
                            cola=False, f1=False, stsb=False):
    """Evaluate the model on a validation or test set.

    Args:
        model: Model to evaluate.
        eval_dataloader: DataLoader with evaluation data.
        device: Device to use.
        modality: Data modality ("image" or "text").
        description: Description for the progress bar.
        cola: Flag indicating whether to use the CoLA evaluation metric (Matthews correlation coefficient).
        f1: Flag indicating whether to report F1 score (weighted average). Ignored if stsb or cola is True.
        stsb: Flag indicating whether to use Spearman correlation for the STS-B dataset (regression-based evaluation).

    Returns:
        tuple: (metric, avg_loss, throughput, stats)
            - metric:
                * For STS-B (stsb=True): Spearman correlation.
                * For CoLA (cola=True): Matthews correlation coefficient.
                * For F1 (f1=True): Weighted F1 score.
                * Otherwise: Accuracy.
            - avg_loss: Average loss per batch.
            - throughput: Average samples processed per second.
            - stats: Dictionary with detailed timing statistics.
    """

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_time = 0
    total_samples = 0

    # Determine whether to accumulate predictions and labels based on flags.
    # Note: stsb takes precedence over both cola and f1.
    use_accumulated_metrics = stsb or cola or f1
    all_preds = [] if use_accumulated_metrics else None
    all_labels = [] if use_accumulated_metrics else None

    # Track detailed timing stats
    timing_stats = {
        'batch_times': [],
        'samples_per_batch': [],
        'batch_throughputs': []
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=description):
            # Create CUDA events for timing (if using a CUDA device)
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)

            if modality == "image":
                # Handle image modality
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                # Time the forward pass
                batch_start.record()
                outputs = model(images)
                batch_end.record()

                # Extract logits whether outputs has a logits attribute or is a raw tensor.
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            elif modality == "text":
                # Handle text modality (assumes batch is a tuple with inputs, attention mask, and labels)
                inputs, attention_mask, labels = batch[0], batch[1], batch[-1]
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)

                # Time the forward pass
                batch_start.record()
                outputs = model(inputs, attention_mask=attention_mask)
                batch_end.record()

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            else:
                raise ValueError("Unsupported modality type. Choose 'text' or 'image'.")

            # Synchronize CUDA and measure time
            torch.cuda.synchronize()
            batch_time = batch_start.elapsed_time(batch_end) / 1000.0  # Convert ms to seconds

            # Compute loss and collect predictions/labels
            if stsb:
                # For STS-B, treat the task as regression.
                # Assume model outputs one value per example; squeeze extra dims if necessary.
                preds = logits.squeeze()
                # Compute MSE loss; labels should be floats.
                loss = nn.MSELoss()(preds, labels.float())
                total_loss += loss.item()

                # Accumulate predictions and ground truth
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
            else:
                # For classification tasks.
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()

                predictions = logits.argmax(dim=1)

                if cola or f1:
                    all_preds.extend(predictions.detach().cpu().numpy().tolist())
                    all_labels.extend(labels.detach().cpu().numpy().tolist())
                else:
                    correct += (predictions == labels).sum().item()

            total += batch_size

            # Track throughput timing
            total_time += batch_time
            total_samples += batch_size

            # Record detailed timing stats for this batch
            timing_stats['batch_times'].append(batch_time)
            timing_stats['samples_per_batch'].append(batch_size)
            timing_stats['batch_throughputs'].append(batch_size / batch_time if batch_time > 0 else 0)

    # Compute the metric based on flags:
    # Priority: stsb > cola > f1 > accuracy.
    if stsb:
        metric, _ = spearmanr(all_labels, all_preds)
    elif cola:
        metric = matthews_corrcoef(all_labels, all_preds)
    elif f1:
        metric = f1_score(all_labels, all_preds, average='weighted')
    else:
        metric = correct / total

    avg_loss = total_loss / len(eval_dataloader)
    throughput = total_samples / total_time if total_time > 0 else 0

    # Compile detailed timing statistics.
    stats = {
        'total_time': total_time,
        'total_samples': total_samples,
        'avg_batch_time': np.mean(timing_stats['batch_times']),
        'std_batch_time': np.std(timing_stats['batch_times']),
        'min_batch_time': np.min(timing_stats['batch_times']),
        'max_batch_time': np.max(timing_stats['batch_times']),
        'avg_batch_size': np.mean(timing_stats['samples_per_batch']),
        'avg_throughput': throughput,
        'p50_throughput': np.median(timing_stats['batch_throughputs']),
        'p90_throughput': np.percentile(timing_stats['batch_throughputs'], 90),
        'p95_throughput': np.percentile(timing_stats['batch_throughputs'], 95),
        'min_throughput': np.min(timing_stats['batch_throughputs']),
        'max_throughput': np.max(timing_stats['batch_throughputs'])
    }

    return metric, avg_loss, throughput, stats


def sample_active_set(dataloader, ratio=0.1, samples_per_class=None):
    """
    Sample a subset of the data for neuron activation analysis.

    Args:
        dataloader: PyTorch DataLoader instance
        ratio: Fraction of data to sample
        samples_per_class: Number of samples per class (if None, determined by ratio)

    Returns:
        A new DataLoader with the sampled data
    """
    # Initialize dictionary to store indices by class    
    rng = random.Random(42)   
    indices_by_class = defaultdict(list)

    print("Analyzing dataset to find samples per class...")
    dataset = dataloader.dataset
    indices_by_class = defaultdict(list)

    print("Analyzing dataset to find samples per class...")

    scan_loader = DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        collate_fn=getattr(dataloader, 'collate_fn', None)
    )

    # Iterate through the dataset to find examples of each class
    for idx, batch in enumerate(tqdm(scan_loader)):
        # Handle different batch structures
        if isinstance(batch, (list, tuple)):
            # Common case: (inputs, labels) or (inputs, attention_mask, labels)
            if len(batch) == 2:  # (inputs, labels)
                labels = batch[1]
            elif len(batch) == 3:  # (inputs, attention_mask, labels)
                labels = batch[2]
            elif len(batch) == 4:  # (inputs, attention_mask, token_type_ids, labels)
                labels = batch[3]
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        else:
            labels = batch['labels']

        if isinstance(labels, torch.Tensor):
            batch_labels = labels.cpu().numpy()
        else:
            batch_labels = labels

        batch_size = len(batch_labels)
        for i in range(batch_size):
            label = batch_labels[i]
            sample_idx = idx * scan_loader.batch_size + i
            indices_by_class[label].append(sample_idx)

    if samples_per_class is None:
        # Calculate samples per class based on ratio
        total_samples = sum(len(indices) for indices in indices_by_class.values())
        num_classes = len(indices_by_class)
        samples_per_class = int(total_samples * ratio / num_classes)

    sampled_indices = []
    print(f"Found {len(indices_by_class)} classes")
    for label, indices in indices_by_class.items():
        n_samples = min(samples_per_class, len(indices))
        print(f"Class {label}: Found {len(indices)} samples, taking {n_samples}")
        sampled_indices.extend(rng.sample(indices, n_samples))

    subset_dataset = Subset(dataloader.dataset, sampled_indices)

    sampled_dataloader = DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        collate_fn=dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else None
    )

    print(f"Created active sampled dataloader with {len(subset_dataset)} total samples")
    return sampled_dataloader



def setup_stt_lora(model, lora_config):
    """
    Apply STTLoraLinear wrapper to STTLinear layers.

    Args:
        model: Model to adapt
        lora_config: Configuration with r, alpha, etc.

    Returns:
        Model with STTLoraLinear wrappers applied
    """
    from stt.stt_lora import STTLoraLinear

    # Wrap STTLinear layers with STTLoraLinear
    for name, module in model.named_modules():
        if isinstance(module, STTLinear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            # Create STTLoraLinear wrapper
            ns_lora = STTLoraLinear(
                stt_linear=module,
                r=lora_config['r'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
            )
            setattr(parent, child_name, ns_lora)
            print(f"Applied STTLoraLinear to {name}")

    return model


def setup_lora(model, lora_config, attn=False, ns=False):
    """
    Apply LoRA to the model.

    Args:
        model: Model to adapt
        lora_config: Configuration with r, alpha, etc.
        attn: Whether to apply LoRA to attention layers
        ns: Whether to apply STTLoraLinear to STTLinear layers

    Returns:
        Model with LoRA applied
    """
    # First apply STTLoraLinear if requested
    if ns:
        model = setup_stt_lora(model, lora_config)

    # Get target modules for regular LoRA
    exact_target_modules = []
    for name, module in model.named_modules():
        if not attn and "attn" in name or "attention" in name:
            continue
        if isinstance(module, nn.Linear):
            if ns and isinstance(module, (STTLinear, STTLoraLinear)):
                continue
            exact_target_modules.append(name)

    exact_target_modules = sorted(set(exact_target_modules))
    print("Exact target modules for LoRA:", exact_target_modules)

    if lora_config["type"] == "lora":
        config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=exact_target_modules,
            task_type = TaskType.CAUSAL_LM  
        )
    elif lora_config["type"] == "adalora":
        config = AdaLoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            deltaT=10,
            target_modules=exact_target_modules
        )
    elif lora_config["type"] == "loha":
        config = LoHaConfig(
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_modules=exact_target_modules
        )
    elif lora_config["type"] == "lokr":
        config = LoKrConfig(
            r=lora_config["r"],
            alpha=lora_config["alpha"],
            dropout=lora_config["dropout"],
            target_modules=exact_target_modules
        )
    else:
        raise ValueError(f"Unknown LoRA type: {lora_config['type']}")
    # Apply LoRA
    model = get_peft_model(model, config)
    model.config.return_dict = True  
    # Print LoRA-applied modules
    print("\nLoRA applied to the following modules:")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):  # LoRA-modified layers have lora_A parameter
            print(f"{name}")

    return model


def clear_cuda_cache_and_states(optimizer=None, trainer=None, model=None, device="cuda", verbose=True):
    """
    Clear CUDA cache and optimizer/trainer states before mode switch.
    
    Args:
        optimizer: Optional optimizer to clear state
        trainer: Optional trainer object to clear optimizer/lr_scheduler states
        model: Optional model to clear gradient buffers
        device: Device to check ("cuda" or "cpu")
        verbose: If True, print cleanup messages
    
    Returns:
        None
    """
    if device != "cuda":
        return
    
    if verbose:
        print("[Evaluation] Clearing CUDA cache and trainer states before mode switch...")
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Clear optimizer state if exists
    if optimizer is not None:
        try:
            optimizer.state.clear()
            if verbose:
                print("[Evaluation] Cleared optimizer state")
        except Exception as e:
            if verbose:
                print(f"[Evaluation] Warning: failed to clear optimizer state: {e}")
    
    # Clear trainer's optimizer and lr_scheduler states (but keep trainer for test())
    if trainer is not None:
        try:
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                trainer.optimizer.state.clear()
                if verbose:
                    print("[Evaluation] Cleared trainer.optimizer state")
            if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None:
                # Clear scheduler state if it has one
                if hasattr(trainer.lr_scheduler, "state_dict"):
                    try:
                        trainer.lr_scheduler.state_dict().clear()
                    except:
                        pass
                if verbose:
                    print("[Evaluation] Cleared trainer.lr_scheduler state")
        except Exception as e:
            if verbose:
                print(f"[Evaluation] Warning: failed to clear trainer states: {e}")
    
    # Clear model's gradient buffers if any
    if model is not None:
        try:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
        except:
            pass
    
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    if verbose:
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[Evaluation] CUDA memory after cleanup: {allocated_gb:.2f} GB")


@torch.no_grad()
def bench_forward(model, eval_dataloader=None, seq_len=512, batch=4, iters=200, warmup=20, device="cuda"):
    """
    Micro-benchmark to measure pure model forward pass speed for LLM/text models.
    Uses real data from eval_dataloader if provided, otherwise uses synthetic data.
    This measures only GPU compute time, excluding dataloader/generation overhead.
    
    Args:
        model: The model to benchmark
        eval_dataloader: Optional DataLoader with evaluation data (preferred for paper)
        seq_len: Sequence length for input tokens (used only if eval_dataloader is None)
        batch: Batch size (used only if eval_dataloader is None)
        iters: Number of iterations to run
        warmup: Number of warmup iterations
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        (throughput, latency_ms)
    """
    import time
    model.eval()
    
    if eval_dataloader is not None:
        # Use real data from dataloader
        dataloader_iter = iter(eval_dataloader)
        batch_samples = []
        try:
            for _ in range(min(warmup + iters, 100)):
                batch = next(dataloader_iter)
                batch_samples.append(batch)
        except StopIteration:
            dataloader_iter = iter(eval_dataloader)
            for _ in range(min(warmup + iters - len(batch_samples), 100)):
                try:
                    batch = next(dataloader_iter)
                    batch_samples.append(batch)
                except StopIteration:
                    break
        
        if not batch_samples:
            raise ValueError("No data available in eval_dataloader for benchmarking")
        
        def get_batch_input(batch):
            """Extract input_ids and attention_mask from batch"""
            # Handle BatchEncoding from transformers (acts like a dict)
            if hasattr(batch, "input_ids") and hasattr(batch, "attention_mask"):
                return batch["input_ids"].to(device), batch["attention_mask"].to(device)
            elif isinstance(batch, dict):
                return batch["input_ids"].to(device), batch["attention_mask"].to(device)
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    return batch[0].to(device), batch[1].to(device)
                else:
                    return batch[0].to(device), torch.ones_like(batch[0]).to(device)
            raise ValueError(f"Unknown batch format: {type(batch)}")
        
        # Warmup using real data
        for i in range(warmup):
            batch = batch_samples[i % len(batch_samples)]
            input_ids, attention_mask = get_batch_input(batch)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark using real data
        t0 = time.time()
        for i in range(iters):
            batch = batch_samples[(warmup + i) % len(batch_samples)]
            input_ids, attention_mask = get_batch_input(batch)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        
        sec = t1 - t0
        avg_batch_size = sum(get_batch_input(batch)[0].shape[0] for batch in batch_samples[:iters]) / iters
        throughput = iters * avg_batch_size / sec
        latency_ms = (sec / iters) * 1000
        
    else:
        # Fallback to synthetic data (for backward compatibility)
        tok = torch.randint(0, model.config.vocab_size, (batch, seq_len), device=device)
        attn = torch.ones_like(tok)

        # warmup
        for _ in range(warmup):
            _ = model(input_ids=tok, attention_mask=attn, use_cache=False).logits
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = model(input_ids=tok, attention_mask=attn, use_cache=False).logits
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        sec = t1 - t0
        throughput = iters * batch / sec
        latency_ms = (sec / iters) * 1000
    
    return throughput, latency_ms


@torch.no_grad()
def bench_forward_image(model, eval_dataloader, iters=200, warmup=20, device="cuda"):
    """
    Micro-benchmark to measure pure model forward pass speed for image models (ViT/BERT vision).
    Uses real data from eval_dataloader for more accurate and paper-appropriate benchmarking.
    This measures only GPU compute time, excluding dataloader overhead.
    
    Args:
        model: The model to benchmark (should have forward method accepting pixel_values or images)
        eval_dataloader: DataLoader with evaluation data to use for benchmarking
        iters: Number of iterations to run
        warmup: Number of warmup iterations
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        (throughput, latency_ms)
    """
    import time
    model.eval()
    
    # Get real data samples from dataloader
    dataloader_iter = iter(eval_dataloader)
    batch_samples = []
    try:
        for _ in range(min(warmup + iters, 100)):  # Get enough batches for warmup + iters
            batch = next(dataloader_iter)
            batch_samples.append(batch)
    except StopIteration:
        # If dataloader is shorter than needed, cycle through available batches
        dataloader_iter = iter(eval_dataloader)
        for _ in range(min(warmup + iters - len(batch_samples), 100)):
            try:
                batch = next(dataloader_iter)
                batch_samples.append(batch)
            except StopIteration:
                break
    
    if not batch_samples:
        raise ValueError("No data available in eval_dataloader for benchmarking")
    
    def get_batch_input(batch):
        """Extract pixel_values/images from batch"""
        if isinstance(batch, dict):
            if "pixel_values" in batch:
                return batch["pixel_values"].to(device)
            elif "images" in batch:
                return batch["images"].to(device)
        elif isinstance(batch, (list, tuple)):
            # (images, labels) or (images, ...)
            return batch[0].to(device)
        raise ValueError(f"Unknown batch format: {type(batch)}")
    
    # Check and log actual image size from data
    sample_pixel_values = get_batch_input(batch_samples[0])
    if sample_pixel_values.dim() == 4:
        B, C, H, W = sample_pixel_values.shape
        print(f"[Benchmark] Using real data with image size: {H}x{W} (batch={B}, channels={C})")
    
    # Warmup using real data
    for i in range(warmup):
        batch = batch_samples[i % len(batch_samples)]
        pixel_values = get_batch_input(batch)
        outputs = model(pixel_values=pixel_values)
        _ = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark using real data
    t0 = time.time()
    for i in range(iters):
        batch = batch_samples[(warmup + i) % len(batch_samples)]
        pixel_values = get_batch_input(batch)
        outputs = model(pixel_values=pixel_values)
        _ = outputs.logits if hasattr(outputs, 'logits') else outputs
    
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    sec = t1 - t0
    # Calculate average batch size from samples
    avg_batch_size = sum(get_batch_input(batch).shape[0] for batch in batch_samples[:iters]) / iters
    throughput = iters * avg_batch_size / sec
    latency_ms = (sec / iters) * 1000
    
    return throughput, latency_ms


def pad_all_nslinear_modules(model, pad_to: int):
    """
    Pad all STTLinear modules in MLP layers to multiples of pad_to.
    This modifies modules in-place.
    
    Args:
        model: The model to modify
        pad_to: Pad dimensions to multiples of this value (e.g., 128, 256)
    
    Returns:
        Number of modules padded
    """
    padded_count = 0
    
    # Detect model type and get layers
    if hasattr(model, 'vision_encoder'):
        # ViT/BERT model: VisionEncoderWithClassifier
        vision_model = model.vision_encoder
        if hasattr(vision_model, 'vision_model'):
            # CLIP/SigLIP structure
            encoder = vision_model.vision_model.encoder
        elif hasattr(vision_model, 'encoder'):
            # Direct encoder structure
            encoder = vision_model.encoder
        else:
            encoder = vision_model
        
        if hasattr(encoder, 'layer'):
            layers = encoder.layer  # BERT structure
        elif hasattr(encoder, 'layers'):
            layers = encoder.layers  # ViT structure
        else:
            print(f"[Padding] Warning: Could not find layers in encoder. Encoder has: {dir(encoder)}")
            return 0
        
        # ViT/BERT MLP layer names
        mlp_layer_names = ["fc1", "fc2"]  # ViT uses fc1 and fc2
        
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLM model: has model.model.layers structure
        layers = model.model.layers
        mlp_layer_names = ["gate_proj", "up_proj", "down_proj"]  # LLM uses gate/up/down
        
    else:
        # Try to find layers directly
        if hasattr(model, 'layers'):
            layers = model.layers
            mlp_layer_names = ["gate_proj", "up_proj", "down_proj"]  # Default to LLM names
        else:
            print(f"[Padding] Warning: Could not determine model structure. Model has: {dir(model)}")
            return 0
    
    # Iterate through layers and pad STTLinear modules
    for layer_idx, layer in enumerate(layers):
        # Get MLP module
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
        elif hasattr(layer, 'intermediate'):  # BERT structure
            mlp = layer
            mlp_layer_names = ["intermediate.dense", "output.dense"]
        else:
            continue
        
        for name in mlp_layer_names:
            # Handle nested names like "intermediate.dense"
            if '.' in name:
                parts = name.split('.')
                m = layer
                for part in parts:
                    if hasattr(m, part):
                        m = getattr(m, part)
                    else:
                        m = None
                        break
            else:
                m = getattr(mlp, name, None)
            
            if m is None:
                continue
            
            if isinstance(m, torch.nn.Linear):
                continue  # Skip standard Linear
            
            # Unwrap if needed (e.g., STTLoraLinear)
            base = getattr(m, "stt_linear", None)
            if base is not None:
                m = base
            
            # Only pad STTLinear modules
            if isinstance(m, STTLinear):
                # Debug info for first layer only
                if layer_idx == 0:
                    w_shape = tuple(m.linear.weight.shape)
                    print(f"[Padding] Padding L{layer_idx}.{name}: weight_shape={w_shape} -> pad_to={pad_to}")
                
                m.pad_weights(pad_to=pad_to)
                padded_count += 1
    
    print(f"[Padding] Padded {padded_count} STTLinear modules to multiples of {pad_to}")
    return padded_count


def benchmark_nslinear_padding(model, device="cuda", pad_values=[None, 128, 256]):
    """
    Benchmark STTLinear with different padding strategies.
    
    Args:
        model: The model to benchmark (should have STTLinear modules)
        device: Device to run on
        pad_values: List of padding values to test (None = no padding, 128, 256, etc.)
    
    Returns:
        List of tuples: (name, throughput, latency_ms)
    """
    # Save STTLinear modules info for restoration
    nslinear_modules_info = {}
    for name, module in model.named_modules():
        if isinstance(module, STTLinear):
            # Unwrap if needed
            base = getattr(module, "stt_linear", None)
            if base is not None:
                module = base
            
            if isinstance(module, STTLinear):
                nslinear_modules_info[name] = {
                    'module': module,
                    'original_in_features': module.original_in_features,
                    'original_out_features': module.original_out_features,
                    'in_indices': module.in_indices.clone() if module.in_indices is not None else None,
                    'out_indices': module.out_indices.clone() if module.out_indices is not None else None,
                    'inference_time': module.inference_time,
                    'weight': module.linear.weight.clone(),
                    'bias': module.linear.bias.clone() if module.linear.bias is not None else None,
                }
    
    # Helper function to restore STTLinear state
    def restore_nslinear_state():
        """Restore model to STTLinear state by recreating modules"""
        for name, info in nslinear_modules_info.items():
            # Get parent module and attribute name
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            # Recreate STTLinear module
            ns_linear = STTLinear(
                in_features=info['original_in_features'],
                out_features=info['original_out_features'],
                in_indices=info['in_indices'],
                out_indices=info['out_indices'],
                bias=(info['bias'] is not None),
                device=info['weight'].device,
                dtype=info['weight'].dtype,
                inference_time=info['inference_time']
            )
            
            # Restore weights
            with torch.no_grad():
                ns_linear.linear.weight.copy_(info['weight'])
                if info['bias'] is not None:
                    ns_linear.linear.bias.copy_(info['bias'])
            
            # Replace the module
            setattr(parent, attr_name, ns_linear)
    
    results = []
    
    # Test each padding strategy
    for pad_to in pad_values:
        # Restore to original state
        restore_nslinear_state()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Apply padding if needed
        if pad_to is not None:
            pad_all_nslinear_modules(model, pad_to=pad_to)
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            name = f"STTLinear (pad to {pad_to})"
        else:
            name = "STTLinear (no padding)"
        
        # Calculate parameter count after padding (for inference parameter tracking)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n[Benchmark] {name}")
        if pad_to is not None:
            print(f"  Inference Parameters (after padding): {total_params:,}")
            print(f"  Trainable Parameters:                  {trainable_params:,}")
        
        # Run benchmark
        throughput, latency_ms = bench_forward(
            model, seq_len=512, batch=4, iters=200, warmup=20, 
            device=device
        )
        results.append((name, throughput, latency_ms))
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Latency:    {latency_ms:.2f} ms/batch")
    
    # Restore to original state for final use
    restore_nslinear_state()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print(f"[Benchmark Summary]")
        print(f"{'='*80}")
        baseline_throughput = results[0][1]
        
        print(f"{'Method':<40} {'Throughput':>15} {'Latency':>15} {'Speedup':>10}")
        print(f"{'-'*80}")
        for name, tp, lat in results:
            speedup = ((tp / baseline_throughput - 1) * 100) if baseline_throughput > 0 else 0
            speedup_str = f"{speedup:+.2f}%"
            print(f"{name:<40} {tp:>15.2f} {lat:>15.2f} {speedup_str:>10}")
        print(f"{'='*80}")
    
    return results


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_jaccard_similarity(dict1, dict2):
    """Calculates the average Jaccard similarity between two active_indices dictionaries."""
    all_keys = set(dict1.keys()) & set(dict2.keys())
    if '_attn_proj_layers' in all_keys:
        all_keys.remove('_attn_proj_layers')
        
    similarities = []
    
    for key in all_keys:
        set1 = set(dict1[key].cpu().numpy())
        set2 = set(dict2[key].cpu().numpy())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            similarities.append(1.0)
        else:
            similarities.append(intersection / union)
            
    if not similarities:
        return 0.0
        
    return sum(similarities) / len(similarities)

def calculate_directed_coverage(dictA, dictB, weighted=True):
    """Cov(A|B): layerwise intersection-over-A, then aggregate.
       Args:
         dictA, dictB: {layer_name: tensor/list/ndarray of indices}
         weighted: True -> weight by |A_layer|; False -> unweighted mean
    """
    keys = set(dictA.keys()) & set(dictB.keys())
    if '_attn_proj_layers' in keys:
        keys.remove('_attn_proj_layers')

    num, den, count = 0.0, 0.0, 0
    for k in keys:
        # to set[int]
        A = dictA[k]
        B = dictB[k]
        try:
            import torch
            if hasattr(A, "detach"): A = A.detach().cpu().numpy()
            if hasattr(B, "detach"): B = B.detach().cpu().numpy()
        except Exception:
            pass
        import numpy as np
        A = set(int(i) for i in np.array(A).ravel().tolist())
        B = set(int(i) for i in np.array(B).ravel().tolist())

        kA = len(A)
        inter = len(A & B)

        if kA == 0:
            cov_l = 1.0   
            w = 0.0 if weighted else 1.0
        else:
            cov_l = inter / kA
            w = float(kA) if weighted else 1.0

        num += cov_l * w
        den += w
        count += 1

    if count == 0:
        return 1.0  
    if den == 0:
        return 1.0
    return float(num / den)
