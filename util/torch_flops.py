
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Iterable
from contextlib import nullcontext
import torch
from torch.utils.flop_counter import FlopCounterMode
TEXT_MAXLEN, LLM_MAXLEN = 512, 2048
def _autocast_ctx(device="cuda", dtype=torch.float16):
    return torch.autocast(device_type=("cuda" if str(device).startswith("cuda") else "cpu"),
                          dtype=dtype) if dtype else nullcontext()

def _to_device(x, device):
    if torch.is_tensor(x): 
        return x.to(device)
    if isinstance(x, dict):   
        return {k: _to_device(v, device) for k,v in x.items()}
    if isinstance(x, (list,tuple)): 
        return type(x)(_to_device(v, device) for v in x)
    return x

def extract_inputs_from_batch(modality: str, batch: Union[Dict, tuple, list],
                              device: str = "cuda",
                              hf_vision_key: str = "pixel_values") -> Dict[str, torch.Tensor]:
    """
    Return a dict inputs suitable for HF models:
      - image: {'pixel_values': (1,3,H,W)}
      - text : {'input_ids': (1,L), 'attention_mask': (1,L), optional 'labels': (1, ...)}
    For timm/image-only models you may pass the tensor directly to flops_*.
    """
    if modality == "image":
        if isinstance(batch, dict) and hf_vision_key in batch:
            x = batch[hf_vision_key]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]) and batch[0].dim() == 4:
            x = batch[0]
        else:
            raise RuntimeError("Unsupported image loader structure.")
        x1 = x[:1]
        return {"pixel_values": _to_device(x1, device)}
    elif modality in ("text", "llm"):
        pad_id = None
        if isinstance(batch, dict) and "input_ids" in batch:
            pad_id = int(batch["input_ids"][0,0].item())
        elif isinstance(batch, (list,tuple)) and len(batch) >= 1 and torch.is_tensor(batch[0]):
            pad_id = int(batch[0][0,0].item())
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full((1, TEXT_MAXLEN), pad_id, device=device, dtype=torch.long)
        attn_mask = torch.ones((1, TEXT_MAXLEN), device=device, dtype=torch.long)

        out = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": input_ids.clone()}
        return out

@torch.no_grad()
def flops_forward(model: torch.nn.Module,
                  inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                  device: str = "cuda",
                  autocast_dtype: Optional[torch.dtype] = torch.float16,
                  display: bool = False) -> int:
    """Per-example forward FLOPs (inference)."""
    assert torch.cuda.is_available(), "FlopCounterMode requires CUDA"
    model.eval().to(device)
    inputs = _to_device(inputs, device)
    with FlopCounterMode(display=display) as m, _autocast_ctx(device, autocast_dtype):
        _ = model(**inputs) if isinstance(inputs, dict) else model(inputs)
    return int(m.get_total_flops())

def flops_train_step(model: torch.nn.Module,
                     inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                     device: str = "cuda",
                     autocast_dtype: Optional[torch.dtype] = torch.float16,
                     loss_key: str = "loss",
                     criterion: Optional[Any] = None,
                     display: bool = False) -> int:
    """Per-example training FLOPs (forward+backward)."""
    assert torch.cuda.is_available(), "FlopCounterMode requires CUDA"
    model.train().to(device)
    inputs = _to_device(inputs, device)
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    with FlopCounterMode(display=display) as m, _autocast_ctx(device, autocast_dtype):
        out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        if criterion is not None:
            loss = criterion(out, inputs)
        elif isinstance(out, dict) and (loss_key in out):
            loss = out[loss_key]
        else:
            raise ValueError("Provide criterion or ensure model returns {'loss': ...}.")
        loss.backward()
    return int(m.get_total_flops())

@dataclass
class TotalsSpec:
    train_size: int
    test_size: int
    epochs: int = 1
    sel_ratio: float = 0.0  

def summarize_totals(stats: Dict[str, int]) -> str:
    def to_gflops(x: int) -> float: return x / 1e9
    def to_tflops(x: int) -> float: return x / 1e12
    def to_pflops(x: int) -> float: return x / 1e15

    lines = []
    if "selection_flops" in stats:
        lines.append(f"[Selection] {to_tflops(stats['selection_flops']):.2f} TFLOPs")
    lines.append(f"[Training ] {to_pflops(stats['training_flops']):.2f} PFLOPs")
    lines.append(f"[Inference] {to_gflops(stats['inference_flops']):.2f} GFLOPs")
    lines.append(f"[Total    ] {to_pflops(stats['total_flops']):.2f} PFLOPs")
    if "per_example_fwd_flops" in stats:
        lines.append(f"[Per-Ex FWD ] {to_gflops(stats['per_example_fwd_flops']):.3f} GFLOPs")
    if "per_example_train_flops" in stats:
        lines.append(f"[Per-Ex TRAIN] {to_gflops(stats['per_example_train_flops']):.3f} GFLOPs")
    return "\n".join(lines)

def aggregate_totals_flops(
    F_fwd_base: int,
    F_fwd_variant: int,
    F_train_variant: int,
    spec: TotalsSpec,
    include_selection: bool,
) -> Dict[str, int]:
    sel = F_fwd_base * int(math.floor(spec.sel_ratio * spec.train_size)) if include_selection else 0
    train = F_train_variant * spec.train_size * spec.epochs
    infer = F_fwd_variant * spec.test_size
    return {
        "selection_flops": sel,
        "training_flops": train,
        "inference_flops": infer,
        "total_flops": sel + train + infer,
        "per_example_fwd_flops": F_fwd_variant,
        "per_example_train_flops": F_train_variant,
    }


def estimate_flops_infer(
    model: torch.nn.Module,
    data: Iterable,
    modality: str,
    device: str = "cuda",
    tokenizer: Optional[Any] = None,
    sample_batches: int = 2,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
    exclude_embeddings: bool = False,  
) -> Dict[str, int]:
    assert torch.cuda.is_available(), "FlopCounterMode requires CUDA"
    try:
        N_full = len(getattr(data, "dataset", data))
    except Exception:
        if hasattr(data, "batch_size") and hasattr(data, "__len__"):
            N_full = int(data.batch_size) * int(len(data))
        else:
            N_full = 0
    it = iter(data)
    if modality in ("text", "llm"):
        try:
            batch = next(it)
        except StopIteration:
            return {"flops": 0, "N": 0, "D": 0}
        inputs = extract_inputs_from_batch(modality, batch, device=device)
        per_ex = flops_forward(model, inputs, device=device, autocast_dtype=autocast_dtype)
        total = int(per_ex) * int(N_full)
        D = LLM_MAXLEN * int(N_full)
        return {"flops": total, "N": int(N_full), "D": D}
    else:
        assert sample_batches >= 1
        per_ex_list: List[int] = []
        for _ in range(sample_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = extract_inputs_from_batch(modality, batch, device=device)
            fwd = flops_forward(model, inputs, device=device, autocast_dtype=autocast_dtype)
            per_ex_list.append(int(fwd))
        if not per_ex_list:
            return {"flops": 0, "N": 0, "D": 0}
        mean_per_ex = sum(per_ex_list) // len(per_ex_list)
        total = int(mean_per_ex) * int(N_full)
        return {"flops": total, "N": int(N_full), "D": 0}

def estimate_flops_train(
    model: torch.nn.Module,
    data: Iterable,
    modality: str,
    epochs: int = 1,
    device: str = "cuda",
    tokenizer: Optional[Any] = None,
    sample_batches: int = 2,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
    loss_key: str = "loss",
    criterion: Optional[Any] = None,
) -> Dict[str, int]:
    """
    Estimate dataset-level training FLOPs (forward+backward) by sampling
    the first batch and scaling to N * epochs.
    """
    assert torch.cuda.is_available(), "FlopCounterMode requires CUDA"
    it = iter(data)
    if modality in ("text", "llm"):
        try:
            batch = next(it)
        except StopIteration:
            return {"flops": 0, "N": 0}
        inputs = extract_inputs_from_batch(modality, batch, device=device)
        per_ex = flops_train_step(model, inputs, device=device, autocast_dtype=autocast_dtype,
                                  loss_key=loss_key, criterion=criterion)
        try:
            N_full = len(getattr(data, "dataset", data))
        except Exception:
            if hasattr(data, "batch_size") and hasattr(data, "__len__"):
                N_full = int(data.batch_size) * int(len(data))
            else:
                N_full = 1
        total = int(per_ex) * int(N_full) * int(max(epochs, 1))
        return {"flops": total, "N": int(N_full)}
    else:
        assert sample_batches >= 1
        per_ex_list: List[int] = []
        for _ in range(sample_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = extract_inputs_from_batch(modality, batch, device=device)
            tr = flops_train_step(model, inputs, device=device, autocast_dtype=autocast_dtype,
                                  loss_key=loss_key, criterion=criterion)
            per_ex_list.append(int(tr))
        if not per_ex_list:
            return {"flops": 0, "N": 0}
        mean_per_ex = sum(per_ex_list) // len(per_ex_list)
        try:
            N_full = len(getattr(data, "dataset", data))
        except Exception:
            if hasattr(data, "batch_size") and hasattr(data, "__len__"):
                N_full = int(data.batch_size) * int(len(data))
            else:
                N_full = 1
        total = int(mean_per_ex) * int(N_full) * int(max(epochs, 1))
        return {"flops": total, "N": int(N_full)}


def infer_vit_seq_len(model: Optional[torch.nn.Module] = None,
                      H: Optional[int] = None,
                      W: Optional[int] = None,
                      add_cls: bool = True,
                      default_image_size: int = 224,
                      default_patch: int = 16) -> int:
    """
    Infer the sequence length (number of tokens) for a ViT model given image dimensions.
    
    Args:
        model: Optional model to extract image_size and patch_size from config
        H: Image height (if None, will try to infer from model config)
        W: Image width (if None, will try to infer from model config)
        add_cls: Whether to add CLS token (default: True)
        default_image_size: Default image size if not found in model config
        default_patch: Default patch size if not found in model config
    
    Returns:
        Sequence length (number of tokens)
    """
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


def peek_vit_dataloader(dl,
                        model: Optional[torch.nn.Module] = None,
                        n_batches: int = 2,
                        prefix: str = "[VIT]") -> int:
    """
    Peek at a ViT dataloader to inspect batch structure and infer sequence lengths.
    
    Args:
        dl: DataLoader to inspect
        model: Optional model to use for sequence length inference
        n_batches: Number of batches to peek at
        prefix: Prefix for print statements
    
    Returns:
        Total number of tokens across peeked batches
    """
    
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


def _pick_tensor(x):
    """
    Helper function to extract a tensor from model output (dict, object, list, or tensor).
    Used by FLOPs calculation criteria.
    """
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


def prepare_flops_inputs_and_criterion(model, train_dataloader, modality, device, dtype=torch.float32, flops_debug=True):
    """
    Prepare inputs and criterion for FLOPs calculation for classification models.
    
    Args:
        model: The model to prepare inputs for
        train_dataloader: Training dataloader to extract sample batch
        modality: "image" or "text"
        device: Device to place tensors on
        dtype: Data type for tensors (default: torch.float32)
        flops_debug: Whether to print debug information
    
    Returns:
        Tuple of (_inp1_fwd, _inp1_train, _criterion, _y1, flops_dbg_once)
        - _inp1_fwd: Input dict for forward pass FLOPs calculation
        - _inp1_train: Input dict for training step FLOPs calculation
        - _criterion: Criterion function for loss calculation
        - _y1: Optional label tensor (for image modality)
        - flops_dbg_once: Debug state dict
    """
    import torch.nn.functional as F
    
    flops_dbg_once = {"done": False}
    _y1 = None
    
    if modality == "image":
        _batch0 = next(iter(train_dataloader))
        if isinstance(_batch0, (list, tuple)) and len(_batch0) >= 1 and torch.is_tensor(_batch0[0]):
            _x1 = _batch0[0][:1].to(device).to(dtype)
            _y1 = _batch0[1][:1].to(device) if (len(_batch0) > 1 and torch.is_tensor(_batch0[1]) and _batch0[1].dim() == 1) else None
        elif isinstance(_batch0, dict) and "pixel_values" in _batch0 and torch.is_tensor(_batch0["pixel_values"]):
            _x1 = _batch0["pixel_values"][:1].to(device).to(dtype)
            _y1 = _batch0["labels"][:1].to(device) if ("labels" in _batch0 and torch.is_tensor(_batch0["labels"])) else None
        else:
            raise RuntimeError("[FLOPs] Unsupported image batch structure.")
        
        try:
            _ = model(pixel_values=_x1)
            _inp1_fwd = {"pixel_values": _x1}
            _inp1_train = {"pixel_values": _x1}
            if flops_debug and not flops_dbg_once["done"]:
                print(f"[FLOPs][debug] call path = keyword; x1.shape={tuple(_x1.shape)}, dtype={_x1.dtype}, device={_x1.device}")
        except TypeError:
            _inp1_fwd = _x1
            _inp1_train = _x1
            if flops_debug and not flops_dbg_once["done"]:
                print(f"[FLOPs][debug] call path = positional; x1.shape={tuple(_x1.shape)}, dtype={_x1.dtype}, device={_x1.device}")
        
        def _vit_criterion(out, _inputs_unused):
            t, tag = _pick_tensor(out)
            if t is None:
                raise RuntimeError(f"[FLOPs] cannot extract tensor from model output; type(out)={type(out)}")
            if flops_debug and not flops_dbg_once["done"]:
                lab = (_y1 is not None and torch.is_tensor(_y1))
                print(f"[FLOPs][debug] out.type={type(out).__name__}; pick='{tag}', shape={tuple(t.shape)}, dtype={t.dtype}; labels={lab}")
                flops_dbg_once["done"] = True
            
            if (_y1 is not None and torch.is_tensor(_y1)
                and t.dim() >= 2 and t.size(0) == _y1.size(0)):
                return F.cross_entropy(t, _y1)
            return t.float().mean()
        
        _criterion = _vit_criterion
        
    elif modality == "text":
        L = TEXT_MAXLEN
        _inp1_fwd = {
            "input_ids": torch.ones(1, L, dtype=torch.long, device=str(device)),
            "attention_mask": torch.ones(1, L, dtype=torch.long, device=str(device)),
        }
        _inp1_train = dict(_inp1_fwd)
        _inp1_extracted = _inp1_fwd
        _lab = _inp1_extracted.get("labels", None)
        if _lab is None:
            _norm_lab = torch.zeros(1, dtype=torch.long, device=str(device))
        elif torch.is_tensor(_lab):
            _lab = _lab.to(device)
            if _lab.dim() == 0:
                _norm_lab = _lab.view(1).to(torch.long)
            elif _lab.dim() == 1:
                _norm_lab = _lab[:1].to(torch.long)
            else:
                _norm_lab = _lab[:1, 0].to(torch.long)
        else:
            _norm_lab = torch.zeros(1, dtype=torch.long, device=str(device))
            _inp1_train = dict(_inp1_fwd)
            _inp1_train["loss_labels"] = _norm_lab
        
        if flops_debug:
            print(f"[DEBUG] Forward input keys (no labels): {list(_inp1_fwd.keys())}")
            print(f"[DEBUG] Training input keys (with labels): {list(_inp1_train.keys())}")
        
        def _bert_criterion(out, inputs_dict):
            t, tag = _pick_tensor(out)
            if t is None:
                raise RuntimeError(f"[FLOPs] cannot extract tensor from model output; type(out)={type(out)}")
            if flops_debug and not flops_dbg_once["done"]:
                print(f"[FLOPs][debug][BERT] out.pick='{tag}', shape={tuple(t.shape)} (TEXT_MAXLEN={TEXT_MAXLEN})")
                flops_dbg_once["done"] = True
            return t.float().mean()
        
        _criterion = _bert_criterion
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    
    return _inp1_fwd, _inp1_train, _criterion, _y1, flops_dbg_once
