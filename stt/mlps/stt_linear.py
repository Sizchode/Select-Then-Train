import torch
from torch import nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class STTLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            in_indices: Optional[torch.Tensor] = None,
            out_indices: Optional[torch.Tensor] = None,
            tune: bool = True,
            bias: bool = True,
            pass_through: bool = False,
            eps: float = 1e-9,
    ):
        super().__init__()

        self.pass_through = pass_through

        # For pass_through mode, we no longer require in_features == out_features
        # since we'll still use the linear transformation but handle inactive neurons specially

        if in_indices is not None:
            if not isinstance(in_indices, torch.Tensor) or in_indices.ndim != 1:
                raise ValueError("in_indices must be a 1D Tensor or None")
            in_indices = in_indices.to(torch.int64)
        self.register_buffer('in_indices', in_indices)

        if out_indices is not None:
            if not isinstance(out_indices, torch.Tensor) or out_indices.ndim != 1:
                raise ValueError("out_indices must be a 1D Tensor or None")
            out_indices = out_indices.to(torch.int64)
        self.register_buffer('out_indices', out_indices)

        self.original_in_features = in_features
        self.original_out_features = out_features

        self.eps = eps

        # Always create a linear layer, even in pass_through mode
        self.in_features = len(self.in_indices) if self.in_indices is not None else in_features
        self.out_features = len(self.out_indices) if self.out_indices is not None else out_features
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.linear.requires_grad_(tune)

        self.tune = tune

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _in_indices = self.in_indices
        _out_indices = self.out_indices

        # Early exit for empty indices
        if (_in_indices is not None and len(_in_indices) == 0) or (_out_indices is not None and len(_out_indices) == 0):
            return x

        # Simple case: no indices specified
        if _in_indices is None and _out_indices is None:
            if not (self.linear.in_features == self.original_in_features and
                    self.linear.out_features == self.original_out_features):
                pass
            return self.linear(x)

        # Reshape input tensor
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        try:
            x_flat = x.view(-1, self.original_in_features)
        except RuntimeError:
            x_flat = x.reshape(-1, self.original_in_features)

        # Select input features based on in_indices
        if _in_indices is not None:
            x_selected = torch.index_select(x_flat, 1, _in_indices)
        else:
            x_selected = x_flat

        # Forward through linear layer
        output_selected = self.linear(x_selected)

        if _out_indices is not None:
            batch_prod = output_selected.shape[0]

            # Initialize output with very small values (or zeros)
            if self.pass_through and _in_indices is not None and _out_indices is not None:
                # For pass-through: initialize with the original input for values we want to preserve
                # We can only do direct pass-through when in_features == out_features
                if self.original_in_features == self.original_out_features:
                    output_flat = x_flat.clone()
                else:
                    output_flat = torch.ones(
                        batch_prod, self.original_out_features,
                        dtype=output_selected.dtype, device=output_selected.device
                    ) * torch.sum(x_flat, dim=1, keepdim=True)
            else:
                output_flat = torch.ones(
                    batch_prod, self.original_out_features,
                    dtype=output_selected.dtype, device=output_selected.device
                ) * self.eps

            output_flat[:, _out_indices] = output_selected
        else:
            if output_selected.shape[-1] != self.original_out_features:
                raise RuntimeError(f"Output dimension mismatch: Expected {self.original_out_features}, "
                                f"got {output_selected.shape[-1]} from linear layer. "
                                f"Layer config: in_idx={_in_indices is not None}, out_idx={_out_indices is not None}")
            output_flat = output_selected

        if batch_dims:
            final_shape = batch_dims + (self.original_out_features,)
            try:
                output = output_flat.view(final_shape)
            except RuntimeError:
                output = output_flat.reshape(final_shape)
        else:
            output = output_flat.squeeze(0)

        return output

    def get_stats(self):
        orig_in = self.original_in_features
        orig_out = self.original_out_features

        sel_in = self.in_features
        sel_out = self.out_features

        orig_params_w = orig_in * orig_out
        sel_params_w = sel_in * sel_out

        orig_params_b = orig_out if self.linear.bias is not None else 0
        sel_params_b = sel_out if self.linear.bias is not None else 0

        orig_params_total = orig_params_w + orig_params_b
        sel_params_total = sel_params_w + sel_params_b

        param_reduction = 1.0 - (sel_params_total / orig_params_total) if orig_params_total > 0 else 0.0
        input_reduction = 1.0 - (sel_in / orig_in) if orig_in > 0 else 0.0
        output_reduction = 1.0 - (sel_out / orig_out) if orig_out > 0 else 0.0

        return {
            "original_in_features": orig_in,
            "original_out_features": orig_out,
            "selected_in_features": sel_in,
            "selected_out_features": sel_out,
            "input_reduction%": input_reduction * 100,
            "output_reduction%": output_reduction * 100,
            "parameter_reduction%": param_reduction * 100,
            "selected_parameters": sel_params_total,
            "original_parameters": orig_params_total,
            "has_bias": self.linear.bias is not None,
            "pass_through": self.pass_through,
        }

    def extra_repr(self) -> str:
        stats = self.get_stats()
        return (f'orig=({stats["original_in_features"]}->{stats["original_out_features"]}), '
                f'sel=({stats["selected_in_features"]}->{stats["selected_out_features"]}), '
                f'bias={stats["has_bias"]}, tune={self.tune}, pass_through={self.pass_through}')

    @classmethod
    def from_linear(cls, original_module: nn.Linear, in_indices=None, out_indices=None, tune=True, bias=None,
                    pass_through=False):
        if not isinstance(original_module, nn.Linear):
            raise TypeError("original_module must be an instance of nn.Linear")

        orig_out_features, orig_in_features = original_module.weight.shape

        has_bias = bias if bias is not None else (hasattr(original_module, 'bias') and original_module.bias is not None)

        instance = cls(
            in_features=orig_in_features,
            out_features=orig_out_features,
            in_indices=in_indices,
            out_indices=out_indices,
            tune=tune,
            bias=has_bias,
            pass_through=pass_through
        )

        with torch.no_grad():
            if in_indices is not None and out_indices is not None:
                selected_weight = original_module.weight[out_indices][:, in_indices]
            elif in_indices is not None:
                selected_weight = original_module.weight[:, in_indices]
            elif out_indices is not None:
                selected_weight = original_module.weight[out_indices]
            else:
                selected_weight = original_module.weight.clone()

            if selected_weight.shape != instance.linear.weight.shape:
                raise RuntimeError(f"Shape mismatch during weight copy: "
                                  f"Selected weight {selected_weight.shape}, "
                                  f"Target linear layer weight {instance.linear.weight.shape}. "
                                  f"In indices: {in_indices.shape if in_indices is not None else None}, "
                                  f"Out indices: {out_indices.shape if out_indices is not None else None}")

            instance.linear.weight.copy_(selected_weight)

            if has_bias and hasattr(original_module, 'bias') and original_module.bias is not None:
                if out_indices is not None:
                    selected_bias = original_module.bias[out_indices]
                else:
                    selected_bias = original_module.bias.clone()
                instance.linear.bias.copy_(selected_bias)

        return instance