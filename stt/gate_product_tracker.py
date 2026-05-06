import inspect
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateProductTracker:
    """
    LLM-only tracker for ranking MLP neurons by activated-gate times up branch.

    The tracked score is computed from the pre-down-proj hidden state:
        act(gate_proj(x)) * up_proj(x)

    Selection is budget-matched to STT by passing a per-layer k_map built from
    tracker6. Returned keys use the gate layer names so downstream pruning can
    reuse the existing NeuroselectiveTransformer5 path unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        threshold: float = 0.01,
        topk_ratio: float = 0.10,
        use_abs_threshold: bool = True,
        device: str = "cuda",
        track_attention_proj: bool = False,
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.topk_ratio = topk_ratio
        self.use_abs_threshold = use_abs_threshold
        self.device = device
        self.track_attention_proj = track_attention_proj
        self.verbose = verbose

        self.layer_name_map = {}
        self.block_to_gate_name = {}
        self.module_role = {}
        for name, module in model.named_modules():
            clean_name = name.replace(".", "_").replace("-", "_")
            self.layer_name_map[module] = clean_name

            lname = clean_name.lower()
            if "gate_proj" in lname:
                block_key = lname.rsplit("_gate_proj", 1)[0]
                self.block_to_gate_name[block_key] = clean_name
                self.module_role[module] = ("gate", block_key)
            elif "up_proj" in lname:
                block_key = lname.rsplit("_up_proj", 1)[0]
                self.module_role[module] = ("up", block_key)

        self.stats = defaultdict(
            lambda: {"sum_activation": None, "sum_sparsity": None, "count": 0}
        )
        self.hooks = []
        self._pending_gate = {}
        self._pending_up = {}
        self.hidden_act = self._resolve_hidden_activation()

    def _resolve_hidden_activation(self):
        hidden_act = getattr(getattr(self.model, "config", None), "hidden_act", "silu")
        hidden_act = str(hidden_act).lower()
        if hidden_act in {"silu", "swish", "swiglu"}:
            return F.silu
        if hidden_act in {"gelu", "gelu_new", "gelu_pytorch_tanh"}:
            return F.gelu
        if hidden_act == "relu":
            return F.relu
        if self.verbose:
            print(f"[gateprod] Unknown hidden_act={hidden_act}; defaulting to SiLU")
        return F.silu

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _filter_kwargs(self, model: nn.Module, kw: dict) -> dict:
        target = model.forward if hasattr(model, "forward") else model
        sig = inspect.signature(target)
        valid = sig.parameters.keys()
        return {k: v for k, v in kw.items() if k in valid}

    def _update_stats(self, gate_layer_name: str, product: torch.Tensor):
        if isinstance(product, tuple):
            product = product[0]
        flat = product.detach().reshape(-1, product.size(-1)).to(torch.float32).cpu()

        st = self.stats[gate_layer_name]
        if st["sum_activation"] is None:
            dim = flat.shape[1]
            st["sum_activation"] = torch.zeros(dim)
            st["sum_sparsity"] = torch.zeros(dim)

        st["sum_activation"] += flat.sum(dim=0)
        st["sum_sparsity"] += (flat >= self.threshold).sum(dim=0)
        st["count"] += flat.shape[0]

    def _maybe_flush_pair(self, block_key: str):
        gate = self._pending_gate.get(block_key)
        up = self._pending_up.get(block_key)
        gate_layer_name = self.block_to_gate_name.get(block_key)
        if gate is None or up is None or gate_layer_name is None:
            return

        product = self.hidden_act(gate) * up
        self._update_stats(gate_layer_name, product)
        del self._pending_gate[block_key]
        del self._pending_up[block_key]

    def _hook_fn(self, module, inputs, output):
        role_info = self.module_role.get(module)
        if role_info is None:
            return

        role, block_key = role_info
        if isinstance(output, tuple):
            output = output[0]

        detached = output.detach()
        if role == "gate":
            self._pending_gate[block_key] = detached
        else:
            self._pending_up[block_key] = detached
        self._maybe_flush_pair(block_key)

    def _attach_hooks(self):
        matched = []
        for module, role_info in self.module_role.items():
            self.hooks.append(module.register_forward_hook(self._hook_fn))
            matched.append(f"{role_info[0]}:{self.layer_name_map[module].lower()}")
        print(f"[gateprod] hooks attached on {len(matched)} Linear layers")
        if matched:
            print("[gateprod] e.g.", ", ".join(matched[:6]), "...")
        else:
            print("[gateprod][WARN] No gate/up_proj layers matched")

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._pending_gate.clear()
        self._pending_up.clear()

    def collect(self, dataloader):
        self._attach_hooks()
        self.model.eval()
        run_device = self._model_device()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {
                        k: (v.to(run_device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                    self.model(**self._filter_kwargs(self.model, batch))
                elif isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
                    if batch[0].dim() == 4:
                        imgs = batch[0].to(self.device)
                        try:
                            self.model(pixel_values=imgs)
                        except TypeError:
                            self.model(imgs)
                    else:
                        ids = batch[0].to(run_device)
                        mask = (
                            batch[1].to(run_device)
                            if len(batch) > 1 and torch.is_tensor(batch[1])
                            else None
                        )
                        toks = (
                            batch[2].to(run_device)
                            if len(batch) > 2 and torch.is_tensor(batch[2])
                            else None
                        )
                        kwargs = {"input_ids": ids}
                        if mask is not None:
                            kwargs["attention_mask"] = mask
                        if toks is not None:
                            kwargs["token_type_ids"] = toks
                        self.model(**self._filter_kwargs(self.model, kwargs))
                elif isinstance(batch, (list, tuple)) and all(isinstance(x, str) for x in batch):
                    if self.tokenizer is None:
                        raise ValueError("Tokenizer required for text input")
                    encoded = self.tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True
                    )
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

        self._remove_hooks()

    def compute_layerwise_active_indices_with_budget(self, k_map):
        active_per_layer = {}
        for layer, st in self.stats.items():
            if st["count"] == 0:
                continue
            k = int(k_map.get(layer, 0))
            if k <= 0:
                continue

            mean = st["sum_activation"] / st["count"]
            keep = min(k, mean.size(0))
            top_idx = torch.topk(mean, keep, largest=True).indices
            active_per_layer[layer] = top_idx.cpu()
        return active_per_layer

    def get_active_indices_with_budget(self, dataloader, k_map):
        self.collect(dataloader)
        return self.compute_layerwise_active_indices_with_budget(k_map)

    def get_layer_name_map(self):
        return self.layer_name_map


# Backward-compatible alias; logic is identical.
NeuronTracker8 = GateProductTracker
