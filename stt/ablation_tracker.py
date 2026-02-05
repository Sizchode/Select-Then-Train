import torch
import torch.nn as nn
from collections import defaultdict
import inspect
import torch.distributed as dist

MLP_FIRST_LAYER_PATTERNS = [
    # —— GPT / Falcon / Llama —— #
    "gate_proj", "wi", "lin1",

    # —— ViT / CLIP-ViT —— #
    "fc1", "mlp.fc1",

    # —— BERT / RoBERTa / DeBERTa —— #
    "intermediate.dense",
]


class AblationTracker:
    def __init__(self, model: nn.Module, tokenizer=None, threshold=0.01, topk_ratio: float = 0.10, use_abs_threshold=True, device="cuda", track_attention_proj=False, verbose=False, use_abs_for_rate=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.use_abs_threshold = use_abs_threshold
        self.track_attention_proj = track_attention_proj
        self.topk_ratio = topk_ratio
        self.verbose = verbose
        self.use_abs_for_rate = use_abs_for_rate  # If True, use abs for rate calculation (default: False to use signed values)
        self.layer_name_map = {}
        for name, module in model.named_modules():
            clean_name = name.replace(".", "_").replace("-", "_")
            self.layer_name_map[module] = clean_name
            if any(g in name for g in ["gate_proj", "fc1", "w1", "wi_0"]):
                print(f"[MATCH] raw_name={name}  clean_name={clean_name}")

        # CHANGED: Use streaming statistics instead of storing full activations
        self.stats = defaultdict(lambda: {
            "sum_activation": None,   # torch tensor (D,)
            "sum_sparsity": None,     # torch tensor (D,)
            "count": 0                # total tokens
        })
        self.hooks = []

        self._wanda_sum_abs = defaultdict(lambda: None) 
        self._wanda_count   = defaultdict(int)           
        self._wanda_hooks   = []                         

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
        
    def compute_layerwise_active_indices(self, use_activation_rate=False, delta=0.01):
        """
        Compute active indices based on either activation magnitude or activation rate.
        
        Args:
            use_activation_rate: If True, use activation rate (sparsity) for selection.
                                If False, use activation magnitude (mean) for selection.
            delta: Unused, kept for compatibility.
        
        Returns:
            Dictionary mapping layer names to active neuron indices.
        """
        active_per_layer = {}  
        # CHANGED: Use stats instead of neuron_activations
        for layer, st in self.stats.items():  
            if st["count"] == 0:
                continue

            D = st["sum_activation"].size(0)
            k = max(1, int(self.topk_ratio * D))
            k = min(k, D)                            
            
            if use_activation_rate:
                # Use activation rate (sparsity): fraction of samples > threshold
                # Higher rate = more frequently active = more important
                activation_rate = st["sum_sparsity"] / st["count"]
                top_idx = torch.topk(activation_rate, k).indices
            else:
                # Use activation magnitude (mean): average activation value
                # Higher mean = higher average activation = more important
                mean = st["sum_activation"] / st["count"]
                top_idx = torch.topk(mean, k).indices
            
            active_per_layer[layer] = top_idx.cpu()
        return active_per_layer 

    def get_active_indices(self, dataloader, use_activation_rate=False):
        """
        Get active indices based on either activation magnitude or activation rate.
        
        Args:
            dataloader: DataLoader to collect activations from.
            use_activation_rate: If True, use activation rate (sparsity) for selection.
                                If False, use activation magnitude (mean) for selection.
        
        Returns:
            Dictionary mapping layer names to active neuron indices.
        """
        self.collect(dataloader)
        return self.compute_layerwise_active_indices(use_activation_rate=use_activation_rate)

    def compute_layerwise_active_indices_with_budget(self, k_map, use_activation_rate=False):
        """
        Compute active indices based on budget from k_map (like mag_pt/wanda_p).
        
        Args:
            k_map: Dictionary mapping layer names to number of neurons to keep.
            use_activation_rate: If True, use activation rate (sparsity) for selection.
                                If False, use activation magnitude (mean) for selection.
        
        Returns:
            Dictionary mapping layer names to active neuron indices.
        """
        active_per_layer = {}
        for layer, st in self.stats.items():
            if st["count"] == 0:
                continue
            
            k = k_map.get(layer, 0)
            if k <= 0:
                continue
            
            D = st["sum_activation"].size(0)
            keep = min(k, D)
            
            if use_activation_rate:
                activation_rate = st["sum_sparsity"] / st["count"]
                top_idx = torch.topk(activation_rate, keep, largest=True).indices
            else:
                mean = st["sum_activation"] / st["count"]
                top_idx = torch.topk(mean, keep, largest=True).indices
            
            active_per_layer[layer] = top_idx.cpu()
        return active_per_layer

    def get_active_indices_with_budget(self, dataloader, k_map, use_activation_rate=False):
        """
        Collect activations and get active indices based on budget from k_map.
        
        Args:
            dataloader: DataLoader to collect activations from.
            k_map: Dictionary mapping layer names to number of neurons to keep.
            use_activation_rate: If True, use activation rate (sparsity) for selection.
                                If False, use activation magnitude (mean) for selection.
        
        Returns:
            Dictionary mapping layer names to active neuron indices.
        """
        self.collect(dataloader)
        return self.compute_layerwise_active_indices_with_budget(k_map, use_activation_rate=use_activation_rate)

    def get_layer_name_map(self):
        return self.layer_name_map

    def _is_mlp_fc1_name(self, lname: str) -> bool:
        lname = lname.lower()
        patterns = set([p.replace(".", "_") for p in MLP_FIRST_LAYER_PATTERNS])
        patterns.update(["mlp_fc1", "intermediate_dense", "wi_0", "w1", "gate_proj", "fc1", "lin1", "wi"])
        return any(p in lname for p in patterns)
