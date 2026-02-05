
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
