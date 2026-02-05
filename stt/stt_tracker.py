import torch
import torch.nn as nn
from collections import defaultdict
import inspect
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
MLP_FIRST_LAYER_PATTERNS = [
    # —— GPT / Falcon / Llama —— #
    "gate_proj", "wi", "lin1",
    # —— ViT / CLIP-ViT —— #
    "fc1", "mlp.fc1",
    # —— BERT / RoBERTa / DeBERTa —— #
    "intermediate.dense",
]


class NeuronTracker6:
    def __init__(self, model: nn.Module, tokenizer=None, threshold=0.01, topk_ratio: float = 0.10, use_abs_threshold=True, device="cuda", track_attention_proj=False, verbose=False, use_abs_for_rate=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.use_abs_threshold = use_abs_threshold
        self.track_attention_proj = track_attention_proj
        self.topk_ratio = topk_ratio
        self.verbose = verbose
        self.use_abs_for_rate = use_abs_for_rate  
        self.layer_name_map = {}
        for name, module in model.named_modules():
            clean_name = name.replace(".", "_").replace("-", "_")
            self.layer_name_map[module] = clean_name
            if any(g in name for g in ["gate_proj", "fc1", "w1", "wi_0"]):
                print(f"[MATCH] raw_name={name}  clean_name={clean_name}")
        self.stats = defaultdict(lambda: {
            "sum_activation": None,   
            "sum_sparsity": None,     
            "count": 0                
        })
        self.hooks = []
# Removed Wanda-related attributes (moved to wanda_adapt_tracker.py)

    def _hook_fn(self, module, input, output):
        name = self.layer_name_map[module]
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() >= 3:
            flat = output.detach().reshape(-1, output.size(-1))
        else:
            flat = output.detach().reshape(-1, output.size(-1))
        flat = flat.to(torch.float32).cpu()
        
        # Use original signed values (not abs) for both magnitude and rate
        # - Magnitude: sum of original values (signed)
        # - Rate: count of original values > threshold (signed)
        
        # CHANGED: Update streaming statistics instead of appending
        st = self.stats[name]
        if st["sum_activation"] is None:
            D = flat.shape[1]
            st["sum_activation"] = torch.zeros(D)
            st["sum_sparsity"] = torch.zeros(D)
        
        # Magnitude: sum of original signed values
        st["sum_activation"] += flat.sum(dim=0)
        # Rate: count of original signed values > threshold
        st["sum_sparsity"] += (flat >= self.threshold).sum(dim=0)
        st["count"] += flat.shape[0]

    def _attach_hooks(self):
        matched = []                      
        for m in self.model.modules():
            lname = self.layer_name_map[m].lower()
            if isinstance(m, nn.Linear) and any(x in lname for x in ["gate_proj", "fc1", "lin1", "wi", "mlp_fc1","intermediate_dense"]):
                self.hooks.append(m.register_forward_hook(self._hook_fn))
                matched.append(lname)     
        print(f"[tracker] hooks attached on {len(matched)} Linear layers")
        if matched:
            print("[tracker] e.g. ", ", ".join(matched[:5]), "...")
        else:
            print("[tracker][WARN] No layers matched MLP patterns!")

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def _filter_kwargs(self, model: nn.Module, kw: dict) -> dict:
        sig = inspect.signature(model.forward)
        valid = sig.parameters.keys()
        return {k: v for k, v in kw.items() if k in valid}


    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def collect(self, dataloader):
        self._attach_hooks()
        self.model.eval()
        run_device = self._model_device()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    print("using the llm branch in tracker now")
                    batch = {k: v.to(run_device) for k, v in batch.items()}
                    self.model(**batch)
                elif isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):

                    if batch[0].dim() == 4:          
                        imgs = batch[0].to(self.device)
                        print("using the image branch in tracker now")
                        try:                        
                            self.model(pixel_values=imgs)
                        except TypeError:            
                            self.model(imgs)
                    else:                            
                        print("using the text-tuple branch in tracker now")
                        ids  = batch[0].to(run_device)
                        mask = batch[1].to(run_device) if len(batch) > 1 else None
                        toks = batch[2].to(run_device) if len(batch) > 2 else None

                        kw = {"input_ids": ids}
                        if mask is not None:
                            kw["attention_mask"] = mask
                        if toks is not None:
                            kw["token_type_ids"] = toks
                        kw = self._filter_kwargs(self.model, kw)      
                        self.model(**kw)


                elif isinstance(batch, (list, tuple)) and all(isinstance(item, str) for item in batch):
                    if self.tokenizer is None:
                        raise ValueError("Tokenizer required for text input")
                    encoded = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                    encoded = {k: v.to(run_device) for k, v in encoded.items()}
                    self.model(**encoded)
        self._remove_hooks()
        
    def compute_layerwise_active_indices(self, delta=0.01):
        union_per_layer = {}  
        for layer, st in self.stats.items():  
            if st["count"] == 0:
                continue
            mean = st["sum_activation"] / st["count"]
            sparsity = st["sum_sparsity"] / st["count"]

            D = mean.size(0)
            k = max(1, int(self.topk_ratio * D))
            k = min(k, D)                            
            top_mean_idx = torch.topk(mean, k).indices
            top_sp_idx   = torch.topk(sparsity, k).indices
            union_idx = torch.unique(torch.cat([top_mean_idx, top_sp_idx], dim=0))
            union_per_layer[layer] = union_idx.cpu()
        return union_per_layer 

    def get_active_indices(self, dataloader):
        self.collect(dataloader)
        return self.compute_layerwise_active_indices()

    def get_layer_name_map(self):
        return self.layer_name_map

    def _is_mlp_fc1_name(self, lname: str) -> bool:
        lname = lname.lower()
        patterns = set([p.replace(".", "_") for p in MLP_FIRST_LAYER_PATTERNS])
        patterns.update(["mlp_fc1", "intermediate_dense", "wi_0", "w1", "gate_proj", "fc1", "lin1", "wi"])
        return any(p in lname for p in patterns)

    def compute_dual_metric_composition(self, delta=0.01):
        """
        Compute dual metric composition for layer-wise visualization.
        Returns dict with layer names and their composition stats.
        """
        composition_stats = {}
        
        # CHANGED: Use stats instead of neuron_activations
        for layer, st in self.stats.items():
            if st["count"] == 0:
                continue
            
            # Compute mean and sparsity from streaming stats
            mean = st["sum_activation"] / st["count"]
            sparsity = st["sum_sparsity"] / st["count"]
            
            D = mean.size(0)
            k = max(1, int(self.topk_ratio * D))
            k = min(k, D)
            
            top_mean_idx = torch.topk(mean, k).indices
            top_sp_idx = torch.topk(sparsity, k).indices
            
            mean_set = set(top_mean_idx.cpu().numpy())
            sp_set = set(top_sp_idx.cpu().numpy())
            
            both_set = mean_set & sp_set
            mean_only_set = mean_set - both_set  
            sp_only_set = sp_set - both_set     
            union_size = len(mean_set | sp_set)
            
            composition_stats[layer] = {
                'total_neurons': D,
                'selected_k': k,
                'mean_only_count': len(mean_only_set),
                'both_count': len(both_set),
                'rate_only_count': len(sp_only_set),
                'mean_only_share': len(mean_only_set) / union_size if union_size > 0 else 0,
                'both_share': len(both_set) / union_size if union_size > 0 else 0,
                'rate_only_share': len(sp_only_set) / union_size if union_size > 0 else 0,
                'union_size': union_size
            }
            
        return composition_stats

    def visualize_layer_composition(self, composition_stats, model_name="model", dataset_name="dataset",
                                    early_only: bool=False, early_layers: int=10, save_suffix: Optional[str]=None):
        """
        Create difference rate curve visualization across layers.
        Saves both SVG and PDF formats.
        """
        if not composition_stats:
            print("[WARNING] No composition stats to visualize")
            return
            
        def extract_layer_number(layer_name):
            import re
            match = re.search(r'(?:layer_|_L)(\d+)', layer_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
            match = re.search(r'(\d+)', layer_name)
            if match:
                return int(match.group(1))
            return 0  
        
        sorted_layers = sorted(composition_stats.keys(), key=extract_layer_number)
        if early_only:
            sorted_layers = [L for L in sorted_layers if extract_layer_number(L) < early_layers]
            if not sorted_layers:
                print(f"[WARNING] No layers < {early_layers} found; falling back to all layers.")
                sorted_layers = sorted(composition_stats.keys(), key=extract_layer_number)
        
        layer_numbers = []
        overlap_rates = []
        
        for layer in sorted_layers:
            stats = composition_stats[layer]
            layer_num = extract_layer_number(layer)
            layer_numbers.append(layer_num)
            # Overlap rate = both_count / union_size
            overlap_rate = stats['both_count'] / stats['union_size'] if stats['union_size'] > 0 else 0
            overlap_rates.append(overlap_rate)
        
        # Ensure X-axis is self-adaptive: map to 1, 2, 3, ..., N
        if layer_numbers:
            min_layer = min(layer_numbers)
            max_layer = max(layer_numbers)
            # Create mapping from original layer numbers to sequential 1, 2, 3, ...
            layer_mapping = {orig: i+1 for i, orig in enumerate(sorted(set(layer_numbers)))}
            # Update layer_numbers to be sequential
            layer_numbers = [layer_mapping[num] for num in layer_numbers]
            print(f"[PLOT] Layer range: {min_layer}-{max_layer} mapped to 1-{len(layer_mapping)}")
        
        # Calculate difference rate instead of overlap rate
        difference_rates = []
        for layer in sorted_layers:
            stats = composition_stats[layer]
            # Difference rate = (mean_only + rate_only) / union_size = 1 - overlap_rate
            difference_rate = (stats['mean_only_count'] + stats['rate_only_count']) / stats['union_size'] if stats['union_size'] > 0 else 0
            difference_rates.append(difference_rate)
        
        # Set matplotlib parameters to match results_fig style
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 14,
            'figure.titlesize': 18,
            'font.family': 'serif',
            'axes.linewidth': 1.0,
            'grid.alpha': 0.3
        })
        
        # Line width and marker size to match results_fig style
        LW, MS = 2.5, 7.0
        
        fig_width = max(12, len(layer_numbers) * 0.4)  
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        def get_model_display_name(model_name):
            if 'bert' in model_name.lower():
                if 'base' in model_name.lower():
                    return 'BERT-BASE'
                elif 'large' in model_name.lower():
                    return 'BERT-LARGE'
                else:
                    return 'BERT'
            elif 'vit' in model_name.lower() or 'clip' in model_name.lower():
                if 'base' in model_name.lower():
                    return 'CLIP-BASE'
                elif 'large' in model_name.lower():
                    return 'CLIP-LARGE'
                else:
                    return 'ViT'
            elif 'qwen' in model_name.lower():
                    return 'QWEN'

            else:
                return model_name.replace('_', '-').upper()
        
        model_display = get_model_display_name(model_name)
        dataset_display = dataset_name.upper()
        suffix = save_suffix if save_suffix is not None else ("_early{}".format(early_layers) if early_only else "")
        
        # Plot difference rate curve matching bert_base_tradeoff style
        ax.plot(layer_numbers, difference_rates, 'o-', linewidth=LW, markersize=MS, 
                color='#1f77b4', alpha=0.8, zorder=3)
        
        ax.set_xlabel('Number of MLP Layer', fontweight='bold')
        ax.set_ylabel('Difference Rate', fontweight='bold')
        
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        
        # Remove top and right spines to match bert_base_tradeoff style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add dataset label in bottom-right corner (no bbox, simpler style)
        ax.text(
            0.97, 0.05, dataset_display,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=12, fontweight='bold',
        )
        
        plt.tight_layout()
        

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"difference_rate_{model_name.replace('/', '_')}_{dataset_name}_{timestamp}"
        svg_path = f"{base_filename}.svg"
        pdf_path = f"{base_filename}.pdf"
        
        plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"[VISUALIZATION] Saved difference rate charts: {svg_path}, {pdf_path}")
        
        print(f"\n[DIFFERENCE RATE SUMMARY] {model_name} on {dataset_name}")
        print(f"Total MLP layers analyzed: {len(sorted_layers)}")
        print(f"Average difference rate: {np.mean(difference_rates):.3f}")
        print(f"Difference rate range: {np.min(difference_rates):.3f} - {np.max(difference_rates):.3f}")
        for i, layer in enumerate(sorted_layers):
            stats = composition_stats[layer]
            print(f"  Layer {layer_numbers[i]}: difference rate = {difference_rates[i]:.3f} "
                  f"({stats['mean_only_count'] + stats['rate_only_count']}/{stats['union_size']} neurons)")
