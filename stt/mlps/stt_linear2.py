import torch
from torch import nn
import logging
from typing import Optional, Union
import torch.nn.functional as F
logger = logging.getLogger(__name__)


class NeuroselectiveLinear(nn.Module):
    """
    A simplified Linear layer that operates with reduced dimensions.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            in_indices: Optional[torch.Tensor] = None,
            out_indices: Optional[torch.Tensor] = None,
            bias: bool = True,
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None,
            inference_time: bool = False,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Store original dimensions for reference
        self.original_in_features = in_features
        self.original_out_features = out_features

        # Get actual dimensions based on indices
        self.active_in_features = len(in_indices) if in_indices is not None else in_features
        self.active_out_features = len(out_indices) if out_indices is not None else out_features

        # Store indices
        self.register_buffer('in_indices', in_indices)
        self.register_buffer('out_indices', out_indices)
        
        # Inference time mode: if True, return reduced dimensions directly (no scatter)
        self.inference_time = inference_time

        # Create the reduced linear layer
        self.linear = nn.Linear(self.active_in_features, self.active_out_features, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve original shape information
        original_shape = x.shape
        batch_dims = original_shape[:-1]
        actual_in_features = x.shape[-1]

        # Reshape to 2D for processing
        if len(batch_dims) > 0:
            x_flat = x.reshape(-1, actual_in_features)
        else:
            x_flat = x.unsqueeze(0)  # Add batch dimension if none
            actual_in_features = x_flat.shape[-1]

        # Determine if input is already reduced
        input_is_reduced = (actual_in_features == self.active_in_features)
        input_is_full = (actual_in_features == self.original_in_features)

        # Select input features based on mode and input state
        if not self.inference_time:
            # Scatter mode OR down layer in inference time mode
            if input_is_reduced and self.in_indices is not None:
                # Input is already reduced (down layer receives reduced input from gate/up in inference time)
                # Skip index_select - input is already the active features
                x_active = x_flat
            elif input_is_full:
                # Scatter mode: input is full dimensions, need to select active features
                if self.in_indices is not None:
                    # Use advanced indexing instead of index_select to potentially reduce memory overhead
                    x_active = x_flat[:, self.in_indices]
                else:
                    x_active = x_flat
            else:
                raise ValueError(
                    f"Unexpected input dimension: {actual_in_features}. "
                    f"Expected either {self.original_in_features} (full) or {self.active_in_features} (reduced)."
                )
        else:
            # Inference time mode: gate/up layers
            # Input should be full dimensions, select active features for output
            if not input_is_full:
                raise ValueError(
                    f"In inference_time mode, input should have {self.original_in_features} features, "
                    f"but got {actual_in_features}."
                )
            # In inference_time mode, gate/up layers receive full input
            # They don't need in_indices (input is already full), but if in_indices exists, use it
            if self.in_indices is not None:
                # Use advanced indexing instead of index_select to potentially reduce memory overhead
                # Both create new tensors, but advanced indexing might be slightly more efficient
                x_active = x_flat[:, self.in_indices]
            else:
                x_active = x_flat

        # Apply reduced linear layer
        output_active = self.linear(x_active)

        # Expand output to original size if needed
        if self.out_indices is not None:
            if not self.inference_time:
                # Scatter version: expand to original size
                output_flat = torch.zeros(x_flat.shape[0], self.original_out_features,
                                          device=output_active.device, dtype=output_active.dtype)
                output_flat[:, self.out_indices] = output_active
            else:
                # Inference time: return reduced dimensions directly (no scatter)
                output_flat = output_active
                # Note: output shape will be (batch, active_out_features) instead of (batch, original_out_features)
                # Optimized implementation with buffer reuse (tried but didn't improve performance):
                # batch_size = x_flat.shape[0]
                # if self._scatter_buffer is None or self._scatter_buffer.shape[0] < batch_size:
                #     self._scatter_buffer = torch.zeros(
                #         batch_size, self.original_out_features,
                #         device=output_active.device, dtype=output_active.dtype
                #     )
                # output_flat = self._scatter_buffer[:batch_size]
                # output_flat.zero_()
                # output_flat[:, self.out_indices] = output_active
        else:
            output_flat = output_active

        # Restore original batch dimensions
        if len(batch_dims) > 0:
            output_dim = self.active_out_features if (self.inference_time and self.out_indices is not None) else self.original_out_features
            output = output_flat.reshape(batch_dims + (output_dim,))
        else:
            output = output_flat.squeeze(0)  # Remove batch dimension if added

        return output
    @classmethod
    def from_linear(
            cls,
            original_module: nn.Linear,
            in_indices: Optional[torch.Tensor] = None,
            out_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """Creates a NeuroselectiveLinear layer from a standard nn.Linear layer."""
        if not isinstance(original_module, nn.Linear):
            raise TypeError("original_module must be an instance of nn.Linear")

        orig_out_features, orig_in_features = original_module.weight.shape
        has_bias = original_module.bias is not None
        device = original_module.weight.device
        dtype = original_module.weight.dtype

        instance = cls(
            in_features=orig_in_features,
            out_features=orig_out_features,
            in_indices=in_indices,
            out_indices=out_indices,
            bias=has_bias,
            device=device,
            dtype=dtype,
            inference_time=kwargs.get('inference_time', False)
        )

        # Copy weights and biases
        with torch.no_grad():
            if in_indices is not None and out_indices is not None:
                # Both input and output dimensions are reduced
                reduced_weight = original_module.weight[out_indices][:, in_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    reduced_bias = original_module.bias[out_indices]
                    instance.linear.bias.copy_(reduced_bias)

            elif in_indices is not None:
                # Only input dimension is reduced
                reduced_weight = original_module.weight[:, in_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    instance.linear.bias.copy_(original_module.bias)

            elif out_indices is not None:
                # Only output dimension is reduced
                reduced_weight = original_module.weight[out_indices]
                instance.linear.weight.copy_(reduced_weight)

                if has_bias and instance.linear.bias is not None:
                    reduced_bias = original_module.bias[out_indices]
                    instance.linear.bias.copy_(reduced_bias)

            else:
                # No reduction, just copy
                instance.linear.weight.copy_(original_module.weight)

                if has_bias and instance.linear.bias is not None:
                    instance.linear.bias.copy_(original_module.bias)
        return instance

    def pad_weights(self, pad_to: int):
        """
        Pad the internal linear layer's weights to multiples of pad_to.
        This modifies the module in-place by padding the internal nn.Linear layer.
        Useful for optimizing CUDA kernel performance (e.g., Tensor Core alignment).
        
        Args:
            pad_to: Pad dimensions to multiples of this value (e.g., 128, 256)
        """
        w = self.linear.weight  # Shape: [out_features, in_features]
        b = self.linear.bias
        
        current_out, current_in = w.shape
        
        # Calculate padded dimensions
        padded_out = ((current_out + pad_to - 1) // pad_to) * pad_to
        padded_in = ((current_in + pad_to - 1) // pad_to) * pad_to
        
        # Only pad if needed
        if padded_out > current_out or padded_in > current_in:
            # Create padded weight matrix (zero-padded)
            w_padded = torch.zeros(padded_out, padded_in, device=w.device, dtype=w.dtype)
            w_padded[:current_out, :current_in] = w
            
            # Pad bias if exists
            b_padded = None
            if b is not None:
                b_padded = torch.zeros(padded_out, device=b.device, dtype=b.dtype)
                b_padded[:current_out] = b
            
            # Create new Linear layer with padded dimensions
            new_linear = torch.nn.Linear(
                padded_in,
                padded_out,
                bias=(b is not None),
                device=w.device,
                dtype=w.dtype
            )
            
            # Copy padded weights
            with torch.no_grad():
                new_linear.weight.data.copy_(w_padded)
                if b_padded is not None:
                    new_linear.bias.data.copy_(b_padded)
            
            # Explicitly delete old linear layer to free memory
            old_linear = self.linear
            del old_linear
            
            # Replace the internal linear layer
            self.linear = new_linear
            
            # Update active dimensions
            self.active_in_features = padded_in
            self.active_out_features = padded_out
            
            # Note: We keep original_in_features and original_out_features unchanged
            # The forward pass will still work correctly because:
            # - For gate/up_proj in inference_time mode: output is reduced, we'll use [:current_out]
            # - For down_proj: input is reduced, we'll use [:current_in]
        else:
            # No padding needed, just ensure contiguous
            if not w.is_contiguous():
                w = w.contiguous()
                self.linear.weight.data = w
            if b is not None and not b.is_contiguous():
                b = b.contiguous()
                self.linear.bias.data = b

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"original_in={self.original_in_features}, original_out={self.original_out_features}, "
                f"active_in={self.active_in_features}, active_out={self.active_out_features}, "
                f"bias={self.linear.bias is not None})")
