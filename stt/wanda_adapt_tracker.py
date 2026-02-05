import torch
import torch.nn as nn
from collections import defaultdict
import inspect
from typing import Optional

MLP_FIRST_LAYER_PATTERNS = [
    # —— GPT / Falcon / Llama —— #
    "gate_proj", "wi", "lin1",
    # —— ViT / CLIP-ViT —— #
    "fc1", "mlp.fc1",
    # —— BERT / RoBERTa / DeBERTa —— #
    "intermediate.dense",
]


class WandaAdaptTracker:
    def __init__(self, model: nn.Module, tokenizer=None, topk_ratio: float = 0.10, device="cuda", track_attention_proj=False, verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.topk_ratio = topk_ratio
        self.track_attention_proj = track_attention_proj
        self.verbose = verbose
        
        # Build layer name mapping
        self.layer_name_map = {}
        for name, module in model.named_modules():
            clean_name = name.replace(".", "_").replace("-", "_")
            self.layer_name_map[module] = clean_name
            if any(g in name for g in ["gate_proj", "fc1", "w1", "wi_0"]):
                if self.verbose:
                    print(f"[WANDA MATCH] raw_name={name}  clean_name={clean_name}")

        # Wanda-specific statistics
        self._wanda_sum_abs = defaultdict(lambda: None)  
        self._wanda_count   = defaultdict(int)           
        self._wanda_hooks   = []

    def _filter_kwargs(self, model: nn.Module, kw: dict) -> dict:
        sig = inspect.signature(model.forward)
        valid = sig.parameters.keys()
        return {k: v for k, v in kw.items() if k in valid}

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _is_mlp_fc1_name(self, lname: str) -> bool:
        lname = lname.lower()
        patterns = set([p.replace(".", "_") for p in MLP_FIRST_LAYER_PATTERNS])
        patterns.update(["mlp_fc1", "intermediate_dense", "wi_0", "w1", "gate_proj", "fc1", "lin1", "wi"])
        return any(p in lname for p in patterns)

    def _wanda_hook_fn_factory(self, lname: str):
        def _hook(mod, inp, out):
            with torch.no_grad():
                x = inp[0].detach()
                if x.dim() == 2:           
                    x = x.unsqueeze(1)
                sa = x.abs().sum(dim=(0, 1))  
                if self._wanda_sum_abs[lname] is None:
                    self._wanda_sum_abs[lname] = sa.clone()
                else:
                    self._wanda_sum_abs[lname] += sa
                self._wanda_count[lname] += x.shape[0] * x.shape[1]
        return _hook

    def _wanda_attach_hooks(self):
        matched = []
        for m in self.model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = self.layer_name_map.get(m, None)
            if lname is None or not self._is_mlp_fc1_name(lname):
                continue
            h = m.register_forward_hook(self._wanda_hook_fn_factory(lname))
            self._wanda_hooks.append(h)
            matched.append(lname)
        if self.verbose:
            print(f"[wanda] hooks attached on {len(matched)} fc1 layers")
            if matched:
                print("[wanda] e.g.", ", ".join(matched[:6]), "...")

    def _wanda_remove_hooks(self):
        for h in self._wanda_hooks:
            h.remove()
        self._wanda_hooks = []

    @torch.no_grad()
    def wanda_scan(self, dataloader, scan_batches: int = 4):
        self._wanda_sum_abs.clear(); self._wanda_count.clear()
        self._wanda_attach_hooks()

        self.model.eval()
        run_device = self._model_device()

        it = iter(dataloader)
        for i in range(int(scan_batches)):
            try:
                batch = next(it)
            except StopIteration:
                break

            if isinstance(batch, dict):
                batch = {k: (v.to(run_device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                self.model(**self._filter_kwargs(self.model.forward, batch))

            elif isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
                if batch[0].dim() == 4:
                    imgs = batch[0].to(self.device)
                    try:
                        self.model(pixel_values=imgs)   
                    except TypeError:
                        self.model(imgs)                 
                else:
                    ids  = batch[0].to(run_device)
                    mask = batch[1].to(run_device) if len(batch) > 1 and torch.is_tensor(batch[1]) else None
                    toks = batch[2].to(run_device) if len(batch) > 2 and torch.is_tensor(batch[2]) else None
                    kw = {"input_ids": ids}
                    if mask is not None: kw["attention_mask"] = mask
                    if toks is not None: kw["token_type_ids"] = toks
                    self.model(**self._filter_kwargs(self.model.forward, kw))

            elif isinstance(batch, (list, tuple)) and all(isinstance(item, str) for item in batch):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer required for text input")
                encoded = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                encoded = {k: v.to(run_device) for k, v in encoded.items()}
                self.model(**encoded)

            else:
                if torch.is_tensor(batch):
                    self.model(batch.to(run_device))
                else:
                    try:
                        self.model(batch)
                    except Exception:
                        pass

        self._wanda_remove_hooks()

    @torch.no_grad()
    def wanda_select_indices(self):
        wanda_indices = {}
        for m in self.model.modules():
            if not isinstance(m, nn.Linear):
                continue
            lname = self.layer_name_map.get(m, None)
            if lname is None or not self._is_mlp_fc1_name(lname):
                continue

            in_features  = m.weight.shape[1]
            out_features = m.weight.shape[0]

            x_norm = self._wanda_sum_abs.get(lname, None)
            if x_norm is None:
                x_norm = torch.zeros(in_features, device=m.weight.device, dtype=m.weight.dtype)
            else:
                x_norm = x_norm.to(device=m.weight.device, dtype=m.weight.dtype)

            score = torch.matmul(m.weight.abs(), x_norm)  
            k = max(1, int(self.topk_ratio * out_features))
            k = min(k, out_features)
            idx = torch.topk(score, k, largest=True, sorted=False).indices
            wanda_indices[lname] = idx.detach().cpu().numpy()

        return wanda_indices

    def get_wanda_indices(self, dataloader, scan_batches: int = 4):
        """
        Main interface for Wanda-based neuron selection.
        
        Args:
            dataloader: Data loader for calibration
            scan_batches: Number of batches to use for calibration
            
        Returns:
            Dict mapping layer names to selected neuron indices
        """
        self.wanda_scan(dataloader, scan_batches=scan_batches)
        return self.wanda_select_indices()

    def get_layer_name_map(self):
        return self.layer_name_map
