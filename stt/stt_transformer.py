import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List, Any, Union

logger = logging.getLogger(__name__)

# From our STTLinear implementation
from .mlps.stt_linear2 import STTLinear

# Common MLP submodule names
MLP_SUBMODULE_NAMES = {
    "gate": ["gate_proj", "fc1", "w1", "wi_0"],
    "up": ["up_proj", "w2", "wi_1"],
    "down": ["down_proj", "fc2", "wo"]
}


class STTTransformer:
    def __init__(
            self,
            model: nn.Module,
            active_neurons: Dict[str, Union[torch.Tensor, List[int]]],
            layer_name_map: Optional[Dict[nn.Module, str]] = None,
            device: Optional[Union[str, torch.device]] = None,
            verbose: bool = False,
            inference_time: bool = False,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.layer_name_map = layer_name_map
        self.inference_time = inference_time

        # Convert active neuron lists to tensors
        self.active_neurons = {}
        for k, v in active_neurons.items():
            if isinstance(v, list):
                self.active_neurons[k] = torch.tensor(v, dtype=torch.long, device=self.device)
            elif isinstance(v, torch.Tensor):
                self.active_neurons[k] = v.to(self.device).long()

        self.original_params = sum(p.numel() for p in model.parameters())
        self.transformed_mlp_blocks = {}
        self.parameter_stats = {}

    def _replace_module(self, parent_module: nn.Module, child_name: str, new_module: nn.Module) -> bool:
        """Replace a module in its parent."""
        try:
            setattr(parent_module, child_name, new_module)
            return True
        except Exception as e:
            logger.error(f"Failed to replace module '{child_name}': {e}")
            return False

    def _find_mlp_blocks(self) -> Dict[str, nn.Module]:
        """Find MLP blocks in the model."""
        mlp_blocks = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module) and \
                    any(key in name.lower() for key in ['mlp', 'ffn', 'feed_forward']) and \
                    any(isinstance(child, nn.Linear) for child in module.children()):
                # Check if this is not a child of an already found MLP block
                is_child = any(name.startswith(parent_name + ".") for parent_name in mlp_blocks)
                if not is_child:
                    mlp_blocks[name] = module

        logger.info(f"Found {len(mlp_blocks)} MLP blocks")
        return mlp_blocks

    def _find_mlp_submodules(self, mlp_block: nn.Module):
        """Find gate, up, and down projection layers in an MLP block."""
        gate_name, up_name, down_name = None, None, None
        block_children = dict(mlp_block.named_children())

        # Find first match for each role
        for potential_gate_name in MLP_SUBMODULE_NAMES["gate"]:
            if potential_gate_name in block_children and isinstance(block_children[potential_gate_name], nn.Linear):
                gate_name = potential_gate_name
                break

        for potential_up_name in MLP_SUBMODULE_NAMES["up"]:
            if potential_up_name in block_children and isinstance(block_children[potential_up_name], nn.Linear):
                up_name = potential_up_name
                break

        for potential_down_name in MLP_SUBMODULE_NAMES["down"]:
            if potential_down_name in block_children and isinstance(block_children[potential_down_name], nn.Linear):
                down_name = potential_down_name
                break

        return gate_name, up_name, down_name

    def _get_key_for_module(self, module: nn.Module) -> Optional[str]:
        """Find the corresponding key in active_neurons for a module."""
        if self.layer_name_map:
            original_name = self.layer_name_map.get(module)
            if original_name:
                # Try to find a matching key
                clean_name = original_name.replace(".", "_").replace("-", "_")
                if clean_name in self.active_neurons:
                    return clean_name

        return None

    def transform(self) -> nn.Module:
        """Transform the model by replacing linear layers with NS linear layers."""
        logger.info("Starting intermediate dimension pruning")
        num_blocks_processed = 0
        num_blocks_pruned = 0

        mlp_blocks = self._find_mlp_blocks()

        if not mlp_blocks:
            logger.warning("No MLP blocks found")
            return self.model

        for block_name, block_module in mlp_blocks.items():
            num_blocks_processed += 1

            try:
                # Find submodules
                gate_name, up_name, down_name = self._find_mlp_submodules(block_module)

                # Only require gate_name and down_name (up_name can be None for ViT/BERT)
                if not (gate_name and down_name):
                    logger.warning(f"Skipping block '{block_name}': Could not identify all required submodules")
                    continue

                # Get module instances
                gate_layer = getattr(block_module, gate_name)
                down_layer = getattr(block_module, down_name)
                # up_layer is only needed if up_name is not None (LLM has up layer, ViT/BERT don't)
                up_layer = getattr(block_module, up_name) if up_name is not None else None

                # Find active neurons
                gate_key = self._get_key_for_module(gate_layer)

                if not gate_key or gate_key not in self.active_neurons:
                    logger.warning(f"Skipping block '{block_name}': No active neurons found for gate layer")
                    continue

                intermediate_indices = self.active_neurons[gate_key]

                if intermediate_indices.numel() == 0:
                    logger.warning(f"Skipping block '{block_name}': Gate layer has 0 active neurons")
                    continue

                pruned_intermediate_dim = len(intermediate_indices)
                original_intermediate_dim = gate_layer.out_features

                logger.info(f"Pruning block '{block_name}': {original_intermediate_dim} -> {pruned_intermediate_dim}")

                # Create new pruned layers
                factory_kwargs = {'device': gate_layer.weight.device, 'dtype': gate_layer.weight.dtype}

                new_gate_layer = STTLinear.from_linear(
                    original_module=gate_layer, 
                    in_indices=None,  # Keep all inputs
                    out_indices=intermediate_indices,  # Prune outputs
                    inference_time=self.inference_time,  # For gate/up, use inference time if enabled
                    **factory_kwargs
                 )

                # Only create up_layer if up_name is not None (LLM has up layer, ViT/BERT don't)
                new_up_layer = None
                if up_name is not None:
                    new_up_layer = STTLinear.from_linear(
                        original_module=up_layer,
                        in_indices=None,  # Keep all inputs
                        out_indices=intermediate_indices,  # Prune outputs
                        inference_time=self.inference_time,  # For gate/up, use inference time if enabled
                        **factory_kwargs
                     )

                new_down_layer = STTLinear.from_linear(
                    original_module=down_layer,
                    in_indices=intermediate_indices,  # Prune inputs
                    out_indices=None,  # Keep all outputs
                    inference_time=False,  # Down layer always outputs full dimensions, no need for inference time
                    **factory_kwargs
                    )

                # Replace modules
                replace_success = True
                if not self._replace_module(block_module, gate_name, new_gate_layer):
                    replace_success = False
                # Only replace up_layer if up_name is not None (LLM has up layer, ViT/BERT don't)
                if up_name is not None:
                    if not self._replace_module(block_module, up_name, new_up_layer):
                        replace_success = False
                if not self._replace_module(block_module, down_name, new_down_layer):
                    replace_success = False

                if replace_success:
                    self.transformed_mlp_blocks[block_name] = {
                        "original_intermediate_dim": original_intermediate_dim,
                        "pruned_intermediate_dim": pruned_intermediate_dim,
                        "active_neuron_key": gate_key
                    }
                    num_blocks_pruned += 1

            except Exception as e:
                logger.error(f"Error processing block '{block_name}': {e}")

        # Calculate statistics
        final_total_params = sum(p.numel() for p in self.model.parameters())
        final_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        absolute_reduction = self.original_params - final_total_params
        overall_reduction_perc = absolute_reduction / self.original_params if self.original_params > 0 else 0

        logger.info(f"MLP blocks processed: {num_blocks_processed}")
        logger.info(f"MLP blocks pruned: {num_blocks_pruned}")
        logger.info(f"Original params: {self.original_params:,}")
        logger.info(f"Final params: {final_total_params:,}")
        logger.info(f"Reduction: {overall_reduction_perc:.2%}")

        self.parameter_stats = {
            "initial_total_params": self.original_params,
            "final_total_params": final_total_params,
            "final_trainable_params": final_trainable_params,
            "absolute_model_reduction": absolute_reduction,
            "overall_model_reduction_perc": overall_reduction_perc * 100,
            "mlp_blocks_processed": num_blocks_processed,
            "mlp_blocks_pruned": num_blocks_pruned
        }

        return self.model

    def get_parameter_stats(self):
        return self.parameter_stats

    def get_transformed_mlp_blocks(self):
        return self.transformed_mlp_blocks
    def record_original_active_indices(self) -> None:
        """Record active neuron indices in terms of the original weight matrix, without modifying model."""
        logger.info("Recording original intermediate indices only (no pruning)")
        num_blocks_processed = 0
        mlp_blocks = self._find_mlp_blocks()

        if not mlp_blocks:
            logger.warning("No MLP blocks found")
            return

        for block_name, block_module in mlp_blocks.items():
            num_blocks_processed += 1
            try:
                gate_name, up_name, down_name = self._find_mlp_submodules(block_module)

                if not (gate_name and up_name and down_name):
                    logger.warning(f"Skipping block '{block_name}': Could not identify all required submodules")
                    continue

                gate_layer = getattr(block_module, gate_name)
                gate_key = self._get_key_for_module(gate_layer)

                if not gate_key or gate_key not in self.active_neurons:
                    logger.warning(f"Skipping block '{block_name}': No active neurons found for gate layer")
                    continue

                intermediate_indices = self.active_neurons[gate_key]

                if intermediate_indices.numel() == 0:
                    logger.warning(f"Skipping block '{block_name}': Gate layer has 0 active neurons")
                    continue

                self.transformed_mlp_blocks[block_name] = {
                    "active_indices_orig": intermediate_indices.cpu().tolist(),
                    "original_intermediate_dim": gate_layer.out_features,
                    "pruned_intermediate_dim": len(intermediate_indices),
                    "active_neuron_key": gate_key
                }

            except Exception as e:
                logger.error(f"Error processing block '{block_name}': {e}")

        logger.info(f"Finished recording for {num_blocks_processed} MLP blocks")
