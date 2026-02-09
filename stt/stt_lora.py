import math

import torch
import torch.nn as nn

from .mlps.stt_linear import STTLinear


class STTLoraLinear(nn.Module):
    """
    LoRA implementation for STTLinear layer.
    This implementation adapts LoRA to work with neuron-selective layers by:
    1. Applying LoRA only to the selected input and output features
    2. Maintaining the original pruning pattern while adding low-rank updates
    """

    def __init__(
            self,
            stt_linear: STTLinear,
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            merge_weights: bool = False,
    ):
        """
        Initialize LoRA for a STTLinear layer.

        Args:
            stt_linear: The STTLinear layer to apply LoRA to
            r: LoRA rank
            lora_alpha: LoRA alpha for scaling
            lora_dropout: Dropout probability for LoRA layers
            merge_weights: If True, LoRA weights are merged with the base weights
        """
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        # Store the original STTLinear layer
        self.stt_linear = stt_linear

        # Get dimensions from stt_linear
        self.in_features = self.stt_linear.active_in_features
        self.out_features = self.stt_linear.active_out_features
        self.original_in_features = self.stt_linear.original_in_features
        self.original_out_features = self.stt_linear.original_out_features

        # LoRA components working on the selected features only
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.scaling = self.lora_alpha / self.r

        # Optional dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        self.reset_lora_parameters()

        # Freeze the original weights
        for param in self.stt_linear.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self):
        """Initialize LoRA parameters using kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining the base STTLinear with LoRA.

        Args:
            x: Input tensor of shape (..., original_in_features)

        Returns:
            Output tensor of shape (..., original_out_features)
        """
        # Get original shape for later reshaping
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        if len(batch_dims) > 0:
            x_flat = x.reshape(-1, self.original_in_features)
        else:
            x_flat = x.unsqueeze(0)  # Add batch dimension if none

        # Select input features if needed
        if self.stt_linear.in_indices is not None:
            x_selected = torch.index_select(x_flat, 1, self.stt_linear.in_indices)
        else:
            x_selected = x_flat

        # Apply LoRA path
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x_selected))) * self.scaling

        # Apply base layer
        base_output = self.stt_linear.linear(x_selected)

        # Combine outputs
        combined_output = base_output + lora_output

        # Handle output feature scattering if needed
        if self.stt_linear.out_indices is not None:
            # Original implementation (restored - buffer optimization didn't help):
            output_flat = torch.zeros(
                x_flat.shape[0],
                self.original_out_features,
                device=combined_output.device,
                dtype=combined_output.dtype
            )
            output_flat[:, self.stt_linear.out_indices] = combined_output
        else:
            output_flat = combined_output

        # Restore original batch dimensions
        if len(batch_dims) > 0:
            output = output_flat.reshape(batch_dims + (self.original_out_features,))
        else:
            output = output_flat.squeeze(0)  # Remove batch dimension if added

        return output

    def merge(self):
        """Merge LoRA weights into the base layer."""
        if not self.merge_weights:
            if self.r > 0:
                with torch.no_grad():
                    # Compute the LoRA weights
                    lora_weights = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                    # Add to the base weights
                    self.stt_linear.linear.weight.data += lora_weights
            self.merge_weights = True

    def unmerge(self):
        """Unmerge LoRA weights from the base layer."""
        if self.merge_weights:
            if self.r > 0:
                with torch.no_grad():
                    # Subtract the LoRA weights
                    lora_weights = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
                    self.stt_linear.linear.weight.data -= lora_weights
            self.merge_weights = False

    @property
    def weight(self):
        """Return the effective weight including LoRA contribution if needed."""
        if self.merge_weights:
            return self.stt_linear.linear.weight
        else:
            return self.stt_linear.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"original_in={self.original_in_features}, original_out={self.original_out_features}, "
                f"active_in={self.in_features}, active_out={self.out_features}, "
                f"r={self.r}, alpha={self.lora_alpha}, "
                f"bias={self.stt_linear.linear.bias is not None})")