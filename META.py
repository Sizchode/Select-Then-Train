LLM_DATASET_PATHS = {
    'clutrr': "./datasets/clutrr",
}

def print_model_layers(model, sort=True, include_params=False, filter_pattern=None):
    """
    Print all layer (module) names in a PyTorch model for debugging purposes.
    Args:
        model: The PyTorch model to examine
        sort: Whether to sort the layer names alphabetically (default: True)
        include_params: Whether to include parameter shapes (default: False)
        filter_pattern: Optional regex pattern to filter layer names (default: None)
    Returns:
        List of all layer names
    """
    import re

    # Collect all layer names and optional parameter info
    layer_info = []

    # Get all named modules
    for name, module in model.named_modules():
        # Skip the root module
        if name == '':
            continue

        # Apply filter if provided
        if filter_pattern and not re.search(filter_pattern, name):
            continue

        if include_params:
            # Count parameters in this module
            param_count = sum(p.numel() for p in module.parameters(directly_only=True))
            param_info = []

            # Get shapes of direct parameters
            for param_name, param in module.named_parameters(recurse=False):
                param_info.append(f"{param_name}: {tuple(param.shape)}")

            # Format the info
            if param_info:
                params_str = f" - Params: {param_count:,} - Shapes: {', '.join(param_info)}"
            else:
                params_str = f" - Params: {param_count:,}"

            layer_info.append((name, f"{name} ({module.__class__.__name__}){params_str}"))
        else:
            layer_info.append((name, f"{name} ({module.__class__.__name__})"))

    # Sort if requested
    if sort:
        layer_info.sort(key=lambda x: x[0])

    # Print all layer info
    print(f"Model: {model.__class__.__name__}")
    # print(f"Total layers: {len(layer_info)}")
    # print("-" * 80)
    #
    # for _, info in layer_info:
    #     print(info)
    #
    # print("-" * 80)

    # Return just the layer names
