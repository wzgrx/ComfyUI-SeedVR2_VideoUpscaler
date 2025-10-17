"""
Model Management Module for SeedVR2

This module handles all model-related operations including:
- Model configuration and path resolution
- Model loading with format detection (SafeTensors, PyTorch)
- DiT and VAE model setup and inference configuration
- State dict management with native FP8 support
- Universal compatibility wrappers

Key Features:
- Dynamic import path resolution for different ComfyUI environments
- Native FP8 model support with optimal performance
- Automatic compatibility mode for model architectures
- Memory-efficient model loading and configuration
"""

import os
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, Union, Callable

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Import Debug for type annotations
from ..utils.debug import Debug

# Import GGUF with fallback  
try:
    import gguf
    from gguf import GGMLQuantizationType
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor
    from ..optimization.gguf_ops import replace_linear_with_quantized
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    GGMLQuantizationType = None
    
from .infer import VideoDiffusionInfer
from .model_cache import get_global_cache
from ..common.config import load_config, create_object
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from ..optimization.compatibility import FP8CompatibleDiT
from ..optimization.blockswap import is_blockswap_enabled, apply_block_swap_to_dit, cleanup_blockswap
from ..optimization.memory_manager import cleanup_dit, cleanup_vae
from ..utils.constants import get_script_directory, find_model_file, suppress_tensor_warnings

# Get script directory for config paths
script_directory = get_script_directory()


def _configs_equal(config1: Optional[Dict[str, Any]], config2: Optional[Dict[str, Any]]) -> bool:
    """
    Compare two configuration dictionaries for equality.
    Handles None values properly.
    
    Args:
        config1: First configuration dict (can be None)
        config2: Second configuration dict (can be None)
        
    Returns:
        True if configs are equivalent, False otherwise
    """
    # Both None = equal
    if config1 is None and config2 is None:
        return True
    
    # One None, one not = different
    if (config1 is None) != (config2 is None):
        return False
    
    # Compare dictionary contents
    return config1 == config2


def _describe_blockswap_config(config: Optional[Dict[str, Any]]) -> str:
    """
    Generate human-readable description of BlockSwap configuration.
    
    Args:
        config: BlockSwap configuration dictionary
        
    Returns:
        Human-readable description string
    """
    if not is_blockswap_enabled(config):
        return "disabled"
    
    blocks_to_swap = config.get("blocks_to_swap", 0)
    swap_io_components = config.get("swap_io_components", False)
    
    block_text = "block" if blocks_to_swap <= 1 else "blocks"
    parts = [f"{blocks_to_swap} {block_text}"]
    if swap_io_components:
        parts.append("I/O offload")
    
    return f"enabled ({', '.join(parts)})"


def _describe_compile_config(config: Optional[Dict[str, Any]]) -> str:
    """
    Generate human-readable description of torch.compile configuration.
    
    Args:
        config: torch.compile configuration dictionary
        
    Returns:
        Human-readable description string
    """
    if config is None or not config:
        return "disabled"
    
    # Core parameters
    mode = config.get("mode", "default")
    backend = config.get("backend", "inductor")
    
    parts = [f"{mode} mode"]
    
    # Optional flags
    if backend != "inductor":
        parts.append(f"{backend} backend")
    if config.get("fullgraph", False):
        parts.append("fullgraph")
    if config.get("dynamic", False):
        parts.append("dynamic")
    
    # Dynamo tuning parameters (show if non-default)
    cache_limit = config.get("dynamo_cache_size_limit", 64)
    recompile_limit = config.get("dynamo_recompile_limit", 128)
    
    if cache_limit != 64:
        parts.append(f"cache_limit={cache_limit}")
    if recompile_limit != 128:
        parts.append(f"recompile_limit={recompile_limit}")
    
    return f"enabled ({', '.join(parts)})"


def _describe_tiling_config(encode_tiled: bool, encode_tile_size: Optional[Tuple[int, int]], 
                           encode_tile_overlap: Optional[Tuple[int, int]],
                           decode_tiled: bool, decode_tile_size: Optional[Tuple[int, int]], 
                           decode_tile_overlap: Optional[Tuple[int, int]]) -> str:
    """
    Generate human-readable description of VAE tiling configuration.
    
    Args:
        encode_tiled: Whether encode tiling is enabled
        encode_tile_size: Tile size for encoding
        encode_tile_overlap: Tile overlap for encoding
        decode_tiled: Whether decode tiling is enabled
        decode_tile_size: Tile size for decoding
        decode_tile_overlap: Tile overlap for decoding
        
    Returns:
        Human-readable description string
    """
    if not encode_tiled and not decode_tiled:
        return "disabled"
    
    parts = []
    if encode_tiled:
        parts.append(f"encode Tile: {encode_tile_size}, Overlap: {encode_tile_overlap}")
    if decode_tiled:
        parts.append(f"decode Tile: {decode_tile_size}, Overlap: {decode_tile_overlap}")
    
    return "; ".join(parts)

    
def _update_model_config(
    runner: 'VideoDiffusionInfer',
    model_attr: str,
    model_type: str,
    new_configs: Dict[str, Any],
    cached_config_attrs: Dict[str, str],
    config_describers: Dict[str, Callable],
    special_handlers: Optional[Dict[str, Callable]] = None,
    debug: Debug = None
) -> bool:
    """
    Update model configuration uniformly for DiT or VAE.
    
    This generic function handles config comparison, logging, and attribute updates
    for both DiT and VAE models, reducing code duplication.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_attr: Attribute name for the model ('dit' or 'vae')
        model_type: Model type string for logging ('DiT' or 'VAE')
        new_configs: Dict mapping config names to new values
                    e.g., {'torch_compile': ..., 'block_swap': ...}
        cached_config_attrs: Dict mapping config names to runner attribute names
                           e.g., {'torch_compile': '_dit_compile_args', ...}
        config_describers: Dict mapping config names to description functions
                          e.g., {'torch_compile': _describe_compile_config, ...}
        special_handlers: Optional dict of config-specific handlers
                         e.g., {'block_swap': handle_blockswap_change}
        debug: Debug instance
        
    Returns:
        True if successful
    """
    model = getattr(runner, model_attr)
    
    # Check if model is on meta device (not materialized yet)
    try:
        param_device = next(model.parameters()).device
        if param_device.type == 'meta':
            # Model not yet materialized - just update config attributes
            for config_name, attr_name in cached_config_attrs.items():
                setattr(runner, attr_name, new_configs.get(config_name))
            # Store on model so config travels with the model when cached
            for config_name, attr_name in cached_config_attrs.items():
                model_attr_name = f'_config_{config_name.split("_")[-1]}'
                setattr(model, model_attr_name, new_configs.get(config_name))
            return True
    except StopIteration:
        pass
    
    # Get cached configurations and check what changed
    config_changes = []
    changes_detected = {}
    
    for config_name, attr_name in cached_config_attrs.items():
        cached_value = getattr(runner, attr_name, None)
        new_value = new_configs.get(config_name)
        changed = not _configs_equal(cached_value, new_value)
        changes_detected[config_name] = changed
        
        if changed:
            # Get description function for this config type
            desc_func = config_describers.get(config_name)
            if desc_func:
                if config_name == 'tiling':
                    # Special case for tiling which needs multiple params
                    old_cfg = cached_value or {}
                    new_cfg = new_value or {}
                    old_desc = desc_func(
                        old_cfg.get('encode_tiled', False),
                        old_cfg.get('encode_tile_size'),
                        old_cfg.get('encode_tile_overlap'),
                        old_cfg.get('decode_tiled', False),
                        old_cfg.get('decode_tile_size'),
                        old_cfg.get('decode_tile_overlap')
                    )
                    new_desc = desc_func(
                        new_cfg.get('encode_tiled', False),
                        new_cfg.get('encode_tile_size'),
                        new_cfg.get('encode_tile_overlap'),
                        new_cfg.get('decode_tiled', False),
                        new_cfg.get('decode_tile_size'),
                        new_cfg.get('decode_tile_overlap')
                    )
                else:
                    old_desc = desc_func(cached_value)
                    new_desc = desc_func(new_value)
                
                # Format config name for display
                display_name = config_name.replace('_', ' ').title()
                if config_name == 'torch_compile':
                    display_name = 'torch.compile'
                elif config_name == 'block_swap':
                    display_name = 'BlockSwap'
                    
                config_changes.append(f"{display_name}: {old_desc} → {new_desc}")
    
    # If nothing changed, reuse model as-is
    if not any(changes_detected.values()):
        debug.log(f"{model_type} configuration unchanged, reusing cached model", category=model_type.lower())
        # Still update attributes to ensure consistency
        for config_name, attr_name in cached_config_attrs.items():
            setattr(runner, attr_name, new_configs.get(config_name))
        return True
    
    # Log configuration changes
    debug.log(f"{model_type} configuration changed:", category=model_type.lower(), force=True)
    for change in config_changes:
        debug.log(f"  {change}", category=model_type.lower(), force=True)
    
    # Handle torch.compile unwrapping if needed
    if changes_detected.get('torch_compile', False):
        if model_attr == 'dit' and hasattr(model, '_orig_mod'):
            debug.log(f"Removing torch.compile from {model_type}", category="setup")
            model = model._orig_mod
            setattr(runner, model_attr, model)
        elif model_attr == 'vae':
            # Unwrap compiled VAE submodules if present
            if hasattr(model, 'encoder') and hasattr(model.encoder, '_orig_mod'):
                debug.log(f"Removing torch.compile from {model_type} encoder", category="setup")
                model.encoder = model.encoder._orig_mod
            if hasattr(model, 'decoder') and hasattr(model.decoder, '_orig_mod'):
                debug.log(f"Removing torch.compile from {model_type} decoder", category="setup")
                model.decoder = model.decoder._orig_mod
    
    # Execute special handlers for config-specific logic
    if special_handlers:
        for config_name, handler in special_handlers.items():
            if changes_detected.get(config_name, False):
                handler(runner, cached_config_attrs[config_name], 
                       new_configs[config_name], debug)
    
    # Update config attributes
    setattr(runner, model_attr, model)
    for config_name, attr_name in cached_config_attrs.items():
        setattr(runner, attr_name, new_configs.get(config_name))
    
    # Store on model so config travels with the model when cached
    for config_name, attr_name in cached_config_attrs.items():
        model_attr_name = f'_config_{config_name.split("_")[-1]}'
        setattr(model, model_attr_name, new_configs.get(config_name))
    
    # Mark that configs need to be applied
    config_needs_app_attr = f'_{model_attr}_config_needs_application'
    setattr(runner, config_needs_app_attr, True)
    
    return True


def _handle_blockswap_change(
    runner: 'VideoDiffusionInfer',
    attr_name: str,
    new_config: Optional[Dict[str, Any]],
    debug: Debug
) -> None:
    """
    Handle BlockSwap-specific configuration changes with proper cleanup.
    
    Called by _update_model_config when BlockSwap configuration changes are detected.
    Manages transition between BlockSwap states (enabled/disabled/reconfigured) with
    proper cleanup to avoid memory leaks and state corruption.
    
    Args:
        runner: VideoDiffusionInfer instance with DiT model
        attr_name: Runner attribute name storing cached config (e.g., '_dit_block_swap_config')
        new_config: New BlockSwap configuration dict or None to disable
        debug: Debug instance for logging
    """
    cached_config = getattr(runner, attr_name, None)
    
    # Determine BlockSwap status from configs
    had_blockswap = is_blockswap_enabled(cached_config)
    has_blockswap = is_blockswap_enabled(new_config)
    
    # If old config had BlockSwap features, clean them up first
    if had_blockswap and not has_blockswap:
        # Disabling BlockSwap completely
        debug.log("Disabling BlockSwap completely", category="blockswap")
        cleanup_blockswap(runner=runner, keep_state_for_cache=False)
    
    # Mark as inactive so the new config can be applied
    runner._blockswap_active = False


def _update_dit_config(
    runner: 'VideoDiffusionInfer',
    block_swap_config: Optional[Dict[str, Any]],
    torch_compile_args: Optional[Dict[str, Any]],
    debug: Debug
) -> bool:
    """
    Update DiT model configuration when reusing cached model.
    
    Compares new configuration settings against cached config to detect changes.
    Handles BlockSwap and torch.compile configuration updates with proper cleanup
    and reapplication when settings change.
    
    Args:
        runner: VideoDiffusionInfer instance with cached DiT model
        block_swap_config: New BlockSwap configuration dict with keys:
            - blocks_to_swap: int - Number of transformer blocks to offload
            - swap_io_components: bool - Whether to offload I/O components
        torch_compile_args: New torch.compile configuration dict with keys:
            - backend: str - Compiler backend (default: "inductor")
            - mode: str - Compilation mode (default: "default")
            - fullgraph: bool - Require single graph compilation
            - dynamic: bool - Enable dynamic shapes
            - dynamo_cache_size_limit: int - Cache size limit
            - dynamo_recompile_limit: int - Recompilation limit
        debug: Debug instance for logging
        
    Returns:
        bool: True if configuration was successfully updated
    """
    return _update_model_config(
        runner=runner,
        model_attr='dit',
        model_type='DiT',
        new_configs={
            'torch_compile': torch_compile_args,
            'block_swap': block_swap_config
        },
        cached_config_attrs={
            'torch_compile': '_dit_compile_args',
            'block_swap': '_dit_block_swap_config'
        },
        config_describers={
            'torch_compile': _describe_compile_config,
            'block_swap': _describe_blockswap_config
        },
        special_handlers={
            'block_swap': _handle_blockswap_change
        },
        debug=debug
    )


def _update_vae_config(
    runner: 'VideoDiffusionInfer',
    torch_compile_args: Optional[Dict[str, Any]],
    debug: Debug
) -> bool:
    """
    Update VAE model configuration when reusing cached model.
    
    Compares new configuration settings against cached config to detect changes.
    Handles torch.compile and tiling configuration updates with proper cleanup
    and reapplication when settings change.
    
    Args:
        runner: VideoDiffusionInfer instance with cached VAE model
        torch_compile_args: New torch.compile configuration dict with keys:
            - backend: str - Compiler backend (default: "inductor")
            - mode: str - Compilation mode (default: "default")
            - fullgraph: bool - Require single graph compilation
            - dynamic: bool - Enable dynamic shapes
            - dynamo_cache_size_limit: int - Cache size limit
            - dynamo_recompile_limit: int - Recompilation limit
        debug: Debug instance for logging
        
    Returns:
        bool: True if configuration was successfully updated
    """
    new_tiling_config = getattr(runner, '_new_vae_tiling_config', None)
    
    return _update_model_config(
        runner=runner,
        model_attr='vae',
        model_type='VAE',
        new_configs={
            'torch_compile': torch_compile_args,
            'tiling': new_tiling_config
        },
        cached_config_attrs={
            'torch_compile': '_vae_compile_args',
            'tiling': '_vae_tiling_config'
        },
        config_describers={
            'torch_compile': _describe_compile_config,
            'tiling': _describe_tiling_config
        },
        special_handlers=None,
        debug=debug
    )


def configure_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Optional[Debug] = None,
    dit_cache: bool = False,
    vae_cache: bool = False,
    dit_id: Optional[int] = None,
    vae_id: Optional[int] = None,
    block_swap_config: Optional[Dict[str, Any]] = None,
    encode_tiled: bool = False,
    encode_tile_size: Optional[Tuple[int, int]] = None,
    encode_tile_overlap: Optional[Tuple[int, int]] = None,
    decode_tiled: bool = False,
    decode_tile_size: Optional[Tuple[int, int]] = None,
    decode_tile_overlap: Optional[Tuple[int, int]] = None,
    torch_compile_args_dit: Optional[Dict[str, Any]] = None,
    torch_compile_args_vae: Optional[Dict[str, Any]] = None
) -> Tuple[VideoDiffusionInfer, Dict[str, Any]]:
    """
    Configure VideoDiffusionInfer runner with model loading and settings.
    
    Handles model changes and caching logic with independent DiT/VAE caching support.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging (required)
        dit_cache: Whether to cache DiT model between runs
        vae_cache: Whether to cache VAE model between runs
        dit_id: Node instance ID for DiT model caching (required if dit_cache=True)
        vae_id: Node instance ID for VAE model caching (required if vae_cache=True)
        block_swap_config: Optional BlockSwap configuration for DiT memory optimization
        encode_tiled: Enable tiled encoding to reduce VRAM during VAE encoding
        encode_tile_size: Tile size for encoding (height, width)
        encode_tile_overlap: Tile overlap for encoding (height, width)
        decode_tiled: Enable tiled decoding to reduce VRAM during VAE decoding
        decode_tile_size: Tile size for decoding (height, width)
        decode_tile_overlap: Tile overlap for decoding (height, width)
        torch_compile_args_dit: Optional torch.compile configuration for DiT model
        torch_compile_args_vae: Optional torch.compile configuration for VAE model
        
    Returns:
        Tuple[VideoDiffusionInfer, Dict[str, Any]]: (configured runner, cache context dict)
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic model reloading when models change
        - Optional torch.compile optimization for inference speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and BlockSwap integration
        
    Raises:
        ValueError: If debug instance is not provided
    """
    
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Phase 1: Initialize cache and get cached models
    cache_context = _initialize_cache_context(
        dit_cache, vae_cache, dit_id, vae_id, 
        dit_model, vae_model, debug
    )
    
    # Phase 2: Get or create runner
    runner = _acquire_runner(
        cache_context, dit_model, vae_model, 
        base_cache_dir, debug
    )
    
    # Phase 3: Configure runner settings
    _configure_runner_settings(
        runner, 
        encode_tiled, encode_tile_size, encode_tile_overlap,
        decode_tiled, decode_tile_size, decode_tile_overlap,
        torch_compile_args_dit, torch_compile_args_vae,
        block_swap_config, debug
    )
    
    # Phase 4: Setup models (load from cache or create new)
    _setup_models(
        runner, cache_context, dit_model, vae_model, 
        base_cache_dir, block_swap_config, debug
    )
    
    return runner, cache_context


def _initialize_cache_context(
    dit_cache: bool,
    vae_cache: bool,
    dit_id: Optional[int],
    vae_id: Optional[int],
    dit_model: str,
    vae_model: str,
    debug: Debug
) -> Dict[str, Any]:
    """
    Initialize cache context with global cache lookups and model name validation.
    
    Checks the global cache for existing DiT/VAE models and validates that cached
    models match the requested model names. Removes stale cache entries when model
    names don't match.
    
    Args:
        dit_cache: Whether DiT caching is enabled
        vae_cache: Whether VAE caching is enabled
        dit_id: Node ID for DiT model lookup
        vae_id: Node ID for VAE model lookup
        dit_model: Requested DiT model filename for validation
        vae_model: Requested VAE model filename for validation
        debug: Debug instance for logging
        
    Returns:
        Dict[str, Any]: Cache context dictionary containing:
            - global_cache: GlobalModelCache instance
            - dit_cache: DiT caching enabled flag
            - vae_cache: VAE caching enabled flag
            - dit_id: DiT node ID
            - vae_id: VAE node ID
            - dit_model: DiT model name
            - vae_model: VAE model name
            - cached_dit: Cached DiT model instance (if found and valid)
            - cached_vae: Cached VAE model instance (if found and valid)
            - dit_newly_cached: Flag indicating if DiT was just cached
            - vae_newly_cached: Flag indicating if VAE was just cached
            - reusing_runner: Flag indicating if runner template is reused
            
    Note:
        Automatically removes stale cache entries when model names don't match
        the cached model's stored name (_model_name attribute).
    """
    global_cache = get_global_cache()
    context = {
        'global_cache': global_cache,
        'dit_cache': dit_cache,
        'vae_cache': vae_cache,
        'dit_id': dit_id,
        'vae_id': vae_id,
        'dit_model': dit_model,
        'vae_model': vae_model,
        'cached_dit': None,
        'cached_vae': None,
        'dit_newly_cached': False,
        'vae_newly_cached': False,
        'reusing_runner': False
    }
    
    # Check for cached DiT model with model name validation
    # Model name validation prevents stale cache when user switches models in UI
    if dit_cache and dit_model and dit_id is not None:
        cached_model = global_cache.get_dit({'node_id': dit_id, 'cache_model': True}, debug)
        if cached_model:
            # Verify cached model matches requested model by checking _model_name attribute
            cached_model_name = getattr(cached_model, '_model_name', None)
            if cached_model_name == dit_model:
                # Cache hit with valid model - reuse it
                context['cached_dit'] = cached_model
            else:
                # Model changed - remove stale cache and log the change
                if cached_model_name:
                    debug.log(f"DiT model changed in cache ({cached_model_name} → {dit_model}), "
                             f"removing stale cached model", category="cache", force=True)
                global_cache.remove_dit({'node_id': dit_id}, debug)
    else:
        # Caching disabled or no ID - clean up any existing cache for this node
        if dit_id is not None:
            global_cache.remove_dit({'node_id': dit_id}, debug)

    # Check for cached VAE model with model name validation
    if vae_cache and vae_model and vae_id is not None:
        cached_model = global_cache.get_vae({'node_id': vae_id, 'cache_model': True}, debug)
        if cached_model:
            # Verify cached model matches requested model by checking _model_name attribute
            cached_model_name = getattr(cached_model, '_model_name', None)
            if cached_model_name == vae_model:
                context['cached_vae'] = cached_model
            else:
                # Model changed - remove stale cache and log the change
                if cached_model_name:
                    debug.log(f"VAE model changed in cache ({cached_model_name} → {vae_model}), "
                             f"removing stale cached model", category="cache", force=True)
                global_cache.remove_vae({'node_id': vae_id}, debug)
    else:
        if vae_id is not None:
            global_cache.remove_vae({'node_id': vae_id}, debug)
    
    return context


def _acquire_runner(
    cache_context: Dict[str, Any],
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Debug
) -> VideoDiffusionInfer:
    """
    Get or create VideoDiffusionInfer runner instance using unified caching approach.
    
    Checks global cache for existing runner template matching the DiT/VAE node ID pair.
    If found and model names match, reuses the template. Otherwise creates new runner.
    
    Args:
        cache_context: Cache context dict from _initialize_cache_context containing:
            - global_cache: GlobalModelCache instance
            - dit_id: DiT node ID for cache lookup
            - vae_id: VAE node ID for cache lookup
            - reusing_runner: Flag to be updated if template is reused
        dit_model: DiT model filename for validation (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename for validation (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory for model files
        debug: Debug instance for logging
        
    Returns:
        VideoDiffusionInfer: Runner instance (cached template or newly created)
    """
    # Try to get runner template from global cache
    template = cache_context['global_cache'].get_runner(
        cache_context['dit_id'], cache_context['vae_id'], debug
    )
    
    if template:
        # We have a template - check if we can use it
        current_dit = getattr(template, '_dit_model_name', None)
        current_vae = getattr(template, '_vae_model_name', None)
        models_match = (current_dit == dit_model and current_vae == vae_model)
        
        if models_match:
            # Perfect match - reuse template directly
            runner_key = f"{cache_context['dit_id']}+{cache_context['vae_id']}"
            debug.log(f"Reusing cached runner template: nodes {runner_key}", category="reuse", force=True)
            cache_context['reusing_runner'] = True
            return template
        else:
            # Template exists but models changed and no cached models - create new
            return _create_new_runner(dit_model, vae_model, base_cache_dir, debug)
    else:
        # No template - create new runner
        return _create_new_runner(dit_model, vae_model, base_cache_dir, debug)


def _create_new_runner(
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    debug: Debug
) -> VideoDiffusionInfer:
    """
    Create a new VideoDiffusionInfer runner instance from scratch.
    
    Loads appropriate configuration file based on model size (3B or 7B), creates
    runner instance, and initializes with default settings. Called when no cached
    runner template is available or when model selection changes.
    
    Args:
        dit_model: DiT model filename (determines config selection)
                  - Contains "7b" → loads configs_7b/main.yaml
                  - Otherwise → loads configs_3b/main.yaml
        vae_model: VAE model filename (stored for reference, not used in config selection)
        base_cache_dir: Base directory for model files (not used directly but passed for context)
        debug: Debug instance for logging and timing
        
    Returns:
        VideoDiffusionInfer: Newly created runner with:
            - Loaded OmegaConf configuration
            - Initialized diffusion sampler and schedule
            - Config set to mutable (readonly=False)
            - No models loaded (structure only)
    """
    debug.log(f"Creating new runner: DiT={dit_model}, VAE={vae_model}", 
             category="runner", force=True)
    
    debug.start_timer("config_load")
    config_path = os.path.join(script_directory, 
                              './configs_7b' if "7b" in dit_model else './configs_3b', 
                              'main.yaml')
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")
    
    debug.start_timer("runner_video_infer")
    runner = VideoDiffusionInfer(config, debug)
    OmegaConf.set_readonly(runner.config, False)
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    return runner


def _configure_runner_settings(
    runner: VideoDiffusionInfer,
    encode_tiled: bool,
    encode_tile_size: Optional[Tuple[int, int]],
    encode_tile_overlap: Optional[Tuple[int, int]],
    decode_tiled: bool,
    decode_tile_size: Optional[Tuple[int, int]],
    decode_tile_overlap: Optional[Tuple[int, int]],
    torch_compile_args_dit: Optional[Dict[str, Any]],
    torch_compile_args_vae: Optional[Dict[str, Any]],
    block_swap_config: Optional[Dict[str, Any]],
    debug: Debug
) -> None:
    """
    Configure runner settings for VAE tiling, torch.compile, and BlockSwap.
    
    Stores configuration settings on runner for later comparison and application.
    Settings are stored in temporary "_new_*" attributes and later validated/applied
    in _setup_models phase. This separation allows configuration change detection
    when reusing cached models.
    
    Args:
        runner: VideoDiffusionInfer instance to configure
        encode_tiled: Enable tiled VAE encoding to reduce VRAM during encoding
        encode_tile_size: Tile dimensions (height, width) for encoding in pixels
        encode_tile_overlap: Overlap dimensions (height, width) between encoding tiles
        decode_tiled: Enable tiled VAE decoding to reduce VRAM during decoding
        decode_tile_size: Tile dimensions (height, width) for decoding in pixels
        decode_tile_overlap: Overlap dimensions (height, width) between decoding tiles
        torch_compile_args_dit: torch.compile configuration for DiT model or None
        torch_compile_args_vae: torch.compile configuration for VAE model or None
        block_swap_config: BlockSwap configuration for DiT model or None
        debug: Debug instance (stored on runner for model access)
    """
    # VAE tiling settings
    runner.encode_tiled = encode_tiled
    runner.encode_tile_size = encode_tile_size
    runner.encode_tile_overlap = encode_tile_overlap
    runner.decode_tiled = decode_tiled
    runner.decode_tile_size = decode_tile_size
    runner.decode_tile_overlap = decode_tile_overlap
    
    # Store the new configs temporarily for later comparison
    # Don't set them as attributes yet - let the update functions handle that
    runner._new_dit_compile_args = torch_compile_args_dit
    runner._new_vae_compile_args = torch_compile_args_vae
    runner._new_dit_block_swap_config = block_swap_config
    runner._new_vae_tiling_config = {
        'encode_tiled': encode_tiled,
        'encode_tile_size': encode_tile_size,
        'encode_tile_overlap': encode_tile_overlap,
        'decode_tiled': decode_tiled,
        'decode_tile_size': decode_tile_size,
        'decode_tile_overlap': decode_tile_overlap
    }
    
    runner.debug = debug


def _setup_models(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    dit_model: str,
    vae_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[Dict[str, Any]],
    debug: Debug
) -> None:
    """
    Setup DiT and VAE models from cache or create new structures.
    
    Central orchestration function that:
    1. Sets up DiT model (cached or new structure)
    2. Sets up VAE model (cached or new structure)
    3. Validates and updates configurations for cached models
    4. Stores initial configurations for new models
    5. Cleans up temporary configuration attributes
    
    This function coordinates the complex interaction between model caching,
    configuration management, and the meta device initialization strategy.
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context from _initialize_cache_context with keys:
            - cached_dit, cached_vae: Cached model instances (or None)
            - dit_id, vae_id: Node IDs for caching
            - Other cache state flags
        dit_model: DiT model filename for loading/validation
        vae_model: VAE model filename for loading/validation
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration (passed to DiT setup)
        debug: Debug instance for logging and timing
    """
    debug.start_timer("model_structures")
    
    # Setup DiT
    dit_created = _setup_dit_model(runner, cache_context, dit_model, base_cache_dir, 
                                   block_swap_config, debug)
    
    # Only update DiT config if model was cached/reused (not newly created)
    if not dit_created and hasattr(runner, 'dit') and runner.dit is not None:
        _update_dit_config(runner, runner._new_dit_block_swap_config, 
                         runner._new_dit_compile_args, debug)
    elif dit_created:
        # For newly created models, just set initial config attributes (no comparison needed)
        runner._dit_compile_args = runner._new_dit_compile_args
        runner._dit_block_swap_config = runner._new_dit_block_swap_config
        # Also store on model so config travels with the model when cached
        if hasattr(runner, 'dit') and runner.dit:
            runner.dit._config_compile = runner._new_dit_compile_args
            runner.dit._config_swap = runner._new_dit_block_swap_config
    
    # Setup VAE
    vae_created = _setup_vae_model(runner, cache_context, vae_model, base_cache_dir, debug)
    
    # Only update VAE config if model was cached/reused (not newly created)
    if not vae_created and hasattr(runner, 'vae') and runner.vae is not None:
        _update_vae_config(runner, runner._new_vae_compile_args, debug)
    elif vae_created:
        # For newly created models, just set initial config attributes (no comparison needed)
        runner._vae_compile_args = runner._new_vae_compile_args
        runner._vae_tiling_config = runner._new_vae_tiling_config
        # Also store on model so config travels with the model when cached
        if hasattr(runner, 'vae') and runner.vae:
            runner.vae._config_compile = runner._new_vae_compile_args
            runner.vae._config_tiling = runner._new_vae_tiling_config
    
    # Clean up temporary attributes
    for attr in ['_new_dit_compile_args', '_new_vae_compile_args', '_new_dit_block_swap_config', '_new_vae_tiling_config']:
        if hasattr(runner, attr):
            delattr(runner, attr)
    
    debug.end_timer("model_structures", "Model structures prepared")


def _setup_dit_model(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    dit_model: str,
    base_cache_dir: str,
    block_swap_config: Optional[Dict[str, Any]],
    debug: Debug
) -> bool:
    """
    Setup DiT model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context dict with keys:
            - cached_dit: Cached DiT model instance (or None)
            - dit_id: Node ID for cache operations
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        block_swap_config: BlockSwap configuration to store (not applied here)
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_dit_name = getattr(runner, '_dit_model_name', None)
    if current_dit_name and current_dit_name != dit_model:
        if hasattr(runner, 'dit') and runner.dit is not None:
            debug.log(f"DiT model changed ({current_dit_name} → {dit_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_dit(runner=runner, debug=debug, cache_model=False)
    
    if cache_context['cached_dit'] is not None:
        # Reuse cached DiT model
        debug.log(f"Reusing cached DiT ({cache_context['dit_id']}): {dit_model}", 
                category="reuse", force=True)
        runner.dit = cache_context['cached_dit']
        runner._dit_checkpoint = find_model_file(dit_model, base_cache_dir)
        runner._dit_model_name = dit_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._dit_compile_args = getattr(runner.dit, '_config_compile', None)
        runner._dit_block_swap_config = getattr(runner.dit, '_config_swap', None)
        
        # blockswap_active will be set by apply_block_swap_to_dit
        # when the model is materialized to the inference device
        runner._blockswap_active = False
        
        return False
    elif not hasattr(runner, 'dit') or runner.dit is None:
        # Create new DiT model
        dit_checkpoint_path = find_model_file(dit_model, base_cache_dir)
        runner = prepare_model_structure(runner, "dit", dit_checkpoint_path, 
                                        runner.config, block_swap_config, debug)
        runner._dit_model_name = dit_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._dit_model_name = dit_model
        return False


def _setup_vae_model(
    runner: VideoDiffusionInfer,
    cache_context: Dict[str, Any],
    vae_model: str,
    base_cache_dir: str,
    debug: Debug
) -> bool:
    """
    Setup VAE model from cache or create new meta device structure.
    
    Handles three scenarios:
    1. Model changed: Cleanup old model, configure and create new structure
    2. Cached model available: Reuse cached model, restore config
    3. No model exists: Configure VAE settings, create new meta device structure
    
    Args:
        runner: VideoDiffusionInfer instance to setup
        cache_context: Cache context dict with keys:
            - cached_vae: Cached VAE model instance (or None)
            - vae_id: Node ID for cache operations
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        debug: Debug instance for logging
        
    Returns:
        bool: True if new model structure was created, False if cached model reused
    """
    
    # Check if model changed - clean up old model if different
    current_vae_name = getattr(runner, '_vae_model_name', None)
    if current_vae_name and current_vae_name != vae_model:
        if hasattr(runner, 'vae') and runner.vae is not None:
            debug.log(f"VAE model changed ({current_vae_name} → {vae_model}), cleaning old model", 
                     category="cache", force=True)
            cleanup_vae(runner=runner, debug=debug, cache_model=False)
    
    if cache_context['cached_vae'] is not None:
        # Reuse cached VAE model
        debug.log(f"Reusing cached VAE ({cache_context['vae_id']}): {vae_model}", 
                category="reuse", force=True)
        runner.vae = cache_context['cached_vae']
        runner._vae_checkpoint = find_model_file(vae_model, base_cache_dir)
        runner._vae_model_name = vae_model
        
        # Restore config attributes from model to runner (config travels with model)
        runner._vae_compile_args = getattr(runner.vae, '_config_compile', None)
        runner._vae_tiling_config = getattr(runner.vae, '_config_tiling', None)
        
        return False
    elif not hasattr(runner, 'vae') or runner.vae is None:
        # Create new VAE model
        # Configure VAE
        vae_config_path = os.path.join(script_directory, 
                                      'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
        vae_config = load_config(vae_config_path)
        
        spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
        temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
        vae_config.spatial_downsample_factor = spatial_downsample_factor
        vae_config.temporal_downsample_factor = temporal_downsample_factor

        runner.config.vae.model = OmegaConf.merge(runner.config.vae.model, vae_config)
        
        if torch.mps.is_available():
            original_vae_dtype = runner.config.vae.dtype
            runner.config.vae.dtype = "bfloat16"
            debug.log(f"MPS detected: Setting VAE dtype from {original_vae_dtype} to {runner.config.vae.dtype} for compatibility", 
                    category="precision", force=True)
        
        vae_checkpoint_path = find_model_file(vae_model, base_cache_dir)
        runner = prepare_model_structure(runner, "vae", vae_checkpoint_path, 
                                        runner.config, None, debug)
        
        debug.log(
            f"VAE downsample factors configured "
            f"(spatial: {spatial_downsample_factor}x, "
            f"temporal: {temporal_downsample_factor}x)",
            category="vae"
        )

        runner._vae_model_name = vae_model
        return True
    else:
        # Model already exists from previous run with same runner
        runner._vae_model_name = vae_model
        return False


def load_quantized_state_dict(checkpoint_path: str, device: torch.device = torch.device("cpu"),
                              debug: Optional[Debug] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint with support for multiple formats.
    
    Handles .safetensors, .gguf, and .pth files. GGUF models support quantization
    for memory-efficient loading. Validates required libraries are installed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device for tensor placement (torch.device object, defaults to CPU)
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    device_str = str(device)
    
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        state = load_safetensors_file(checkpoint_path, device=device_str)
    elif checkpoint_path.endswith('.gguf'):
        if not GGUF_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"GGUF library is required but not installed.\n"
                f"Please install it with: pip install gguf"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of quantized GGUF models", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device_str, mmap=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: torch.device, debug: Optional[Debug],
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    
    Args:
        checkpoint_path: Path to GGUF file
        device: Target device (torch.device object)
        debug: Debug instance
        handle_prefix: Prefix to strip from tensor names
        
    Returns:
        State dictionary with loaded tensors
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    device_str = str(device)
    debug.log(f"Loading {total_tensors} tensors to {str(device_str)}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"  Loaded {i+1}/{total_tensors} tensors...", category="dit")

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device_str}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"  Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True)
            self.debug.log(f"  Shape: {self.shape}", level="WARNING", category="dit", force=True)
            self.debug.log(f"  Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"  Function: {func}", level="WARNING", category="dit", force=True)
                        debug.log(f"  Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)


def prepare_model_structure(runner: VideoDiffusionInfer, model_type: str, 
                           checkpoint_path: str, config: OmegaConf, 
                           block_swap_config: Optional[Dict[str, Any]] = None,
                           debug: Optional[Debug] = None) -> VideoDiffusionInfer:
    """
    Prepare model structure on meta device without loading weights.
    This uses zero memory as meta device doesn't allocate real memory.
    
    Args:
        runner: VideoDiffusionInfer instance
        model_type: "dit" or "vae"
        checkpoint_path: Path to checkpoint (stored for later loading)
        config: Model configuration
        block_swap_config: BlockSwap config (stored for DiT)
        debug: Debug instance
        
    Returns:
        runner: Updated runner with model structure on meta device
    """
    if debug is None:
        raise ValueError(f"Debug instance required for prepare_model_structure")
    
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    model_config = config.dit.model if is_dit else config.vae.model
    
    # Always create on meta device for zero memory usage
    debug.log(f"Creating {model_type_upper} model structure on meta device", 
             category=model_type, force=True)
    debug.start_timer(f"{model_type}_structure")
    
    with torch.device("meta"):
        model = create_object(model_config)
    
    debug.end_timer(f"{model_type}_structure", f"{model_type_upper} structure created")
    
    # Store model and config for later materialization
    if is_dit:
        runner.dit = model
        runner._dit_checkpoint = checkpoint_path
        runner._dit_block_swap_config = block_swap_config
    else:
        runner.vae = model  
        runner._vae_checkpoint = checkpoint_path
        # Store VAE dtype override if needed
        runner._vae_dtype_override = getattr(torch, config.vae.dtype) if torch.mps.is_available() else None
    
    return runner


def materialize_model(runner: VideoDiffusionInfer, model_type: str, device: torch.device, 
                     config: OmegaConf, debug: Optional[Debug] = None) -> None:
    """
    Materialize model weights from checkpoint to memory.
    Call this right before the model is needed.
    
    Args:
        runner: Runner with model structure on meta device
        model_type: "dit" or "vae"
        device: Target device for inference (torch.device object)
        config: Full configuration
        debug: Debug instance
    """
    if debug is None:
        raise ValueError(f"Debug instance required for materialize_model")
        
    is_dit = (model_type == "dit")
    model_type_upper = "DiT" if is_dit else "VAE"
    
    # Get model and checkpoint path
    if is_dit:
        model = runner.dit
        checkpoint_path = runner._dit_checkpoint
        block_swap_config = runner._dit_block_swap_config
        override_dtype = None
    else:
        model = runner.vae
        checkpoint_path = runner._vae_checkpoint
        block_swap_config = None
        override_dtype = runner._vae_dtype_override
    
    # Check if already materialized
    if model is None:
        debug.log(f"No {model_type_upper} model structure found", level="WARNING", category=model_type, force=True)
        return
    param_device = next(model.parameters()).device
    if param_device.type != 'meta':
        debug.log(f"{model_type_upper} already materialized on {model.device}", category=model_type)
        return
    
    # Determine target device for materialization
    # For DiT with BlockSwap, materialize to offload device (blocks distributed later)
    # For VAE or DiT without BlockSwap, materialize to inference device
    offload_reason = ""
    if is_dit:
        blockswap_active = hasattr(runner, '_blockswap_active') and runner._blockswap_active
        if blockswap_active:
            # BlockSwap: materialize to offload device, blocks will be distributed
            target_device = runner._block_swap_config.get("offload_device")
            if target_device is None:
                target_device = torch.device("cpu")
            offload_reason = f" (BlockSwap offload device)"
        else:
            # No BlockSwap: materialize directly to inference device
            target_device = device
    else:
        # VAE: always materialize to inference device
        target_device = device
    
    # Start materialization
    debug.start_timer(f"{model_type}_materialize")
    
    # Load weights (this materializes from meta to target device)
    model = _load_model_weights(model, checkpoint_path, target_device, True,
                               model_type_upper, offload_reason, debug, override_dtype) 
   
    # Apply model-specific configurations (includes BlockSwap and torch.compile)
    model = apply_model_specific_config(model, runner, config, is_dit, debug)
    
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")
    
    # Clean up checkpoint paths (no longer needed after weights are loaded)
    # Note: Config attributes (_dit_block_swap_config, _dit_compile_args) are preserved
    # for configuration change detection on subsequent runs
    if is_dit:
        runner._dit_checkpoint = None
    else:
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: torch.device, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Debug, override_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights (torch.device object)
        used_meta: Whether model was created on meta device
        model_type: Model type string for logging
        cpu_reason: Reason string if using CPU
        debug: Debug instance
        override_dtype: Optional dtype override for weights
        
    Returns:
        Model with loaded weights
    """
    model_type_lower = model_type.lower()
    
    # Log loading action
    action = "Materializing" if used_meta else "Loading"
    target_device_str = str(target_device).upper()
    debug.log(f"{action} {model_type} weights to {target_device_str}{cpu_reason}: {checkpoint_path}", 
             category=model_type_lower, force=True)
    
    # Load state dict from file
    debug.start_timer(f"{model_type_lower}_weights_load")
    state = load_quantized_state_dict(checkpoint_path, target_device, debug)
    debug.end_timer(f"{model_type_lower}_weights_load", f"{model_type} weights loaded from file")
    
    # Apply dtype conversion if requested
    if override_dtype is not None:
        state = _convert_state_dtype(state, override_dtype, model_type, debug)
    
    # Log weight statistics
    _log_weight_stats(state, used_meta, model_type, debug)
    
    # Handle GGUF or standard loading
    if checkpoint_path.endswith('.gguf'):
        model = _load_gguf_weights(model, state, used_meta, model_type_lower, debug)
    else:
        model = _load_standard_weights(model, state, used_meta, model_type, model_type_lower, debug)
    
    # Clean up state dict
    del state
    
    # Initialize meta buffers if needed
    if used_meta:
        _initialize_meta_buffers_wrapped(model, target_device, debug)
    
    return model


def _convert_state_dtype(state: Dict[str, torch.Tensor], target_dtype: torch.dtype, 
                        model_type: str, debug: Debug) -> Dict[str, torch.Tensor]:
    """Convert floating point tensors in state dict to target dtype."""
    debug.log(f"Converting {model_type} weights to {target_dtype} during loading", category="precision")
    debug.start_timer(f"{model_type.lower()}_dtype_convert")
    
    for key in state:
        if torch.is_tensor(state[key]) and state[key].is_floating_point():
            state[key] = state[key].to(target_dtype)
    
    debug.end_timer(f"{model_type.lower()}_dtype_convert", f"{model_type} weights converted to {target_dtype}")
    return state


def _log_weight_stats(state: Dict[str, torch.Tensor], used_meta: bool, model_type: str, debug: Debug) -> None:
    """Log statistics about loaded weights."""
    num_params = len(state)
    total_size_mb = sum(p.nelement() * p.element_size() for p in state.values()) / (1024 * 1024)
    action = "Materializing" if used_meta else "Applying"
    debug.log(f"{action} {model_type}: {num_params} parameters, {total_size_mb:.2f}MB total", 
             category=model_type.lower())


def _load_standard_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                          used_meta: bool, model_type: str, model_type_lower: str,
                          debug: Debug) -> torch.nn.Module:
    """Load standard (non-GGUF) weights into model."""
    debug.start_timer(f"{model_type_lower}_state_apply")
    model.load_state_dict(state, strict=False, assign=True)
    
    action = "materialized" if used_meta else "applied"
    debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights {action}")
    
    if used_meta:
        debug.log(f"{model_type} materialized directly from meta with loaded weights", category=model_type_lower)
    else:
        debug.log(f"{model_type} weights applied", category=model_type_lower)
    
    return model


def _load_gguf_weights(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                      used_meta: bool, model_type_lower: str, debug: Debug) -> torch.nn.Module:
    """
    Load GGUF quantized weights into model with architecture validation.
    
    Args:
        model: Target model
        state: GGUF state dict with quantized tensors
        used_meta: Whether model was initialized on meta device
        model_type_lower: Lowercase model type for logging
        debug: Debug instance
        
    Returns:
        Model with GGUF weights loaded
    """
    debug.log("Loading GGUF weights", category="dit")
    
    # Get model state dict for validation
    model_state = model.state_dict()
    
    # Validate architecture compatibility
    _validate_gguf_architecture(state, model_state, debug)
    
    # Load GGUF parameters
    stats = _apply_gguf_parameters(model, state, model_state, debug)
    
    # Log results
    debug.log(f"GGUF loading complete: {stats['loaded']} parameters loaded", category="success")
    debug.log(f"Quantized parameters: {stats['quantized']}", category="info")
    
    # Report any mismatches
    _report_parameter_mismatches(state, model_state, stats['loaded_names'], debug)
    
    # Replace Linear/Conv2d layers with quantized versions for optimal precision handling
    if stats['quantized'] > 0:
        debug.log("Replacing layers with GGUF-optimized versions for precision handling", category="dit")
        
        replacements, quant_types = replace_linear_with_quantized(model, debug=debug)
        
        if replacements > 0:
            debug.log(f"Replaced {replacements} layers with GGUF-optimized versions", category="success")
            
            # Show actual quantization types found and precision strategy
            if quant_types:
                qtypes_str = ', '.join([f"{qtype}:{count}" for qtype, count in quant_types.items()])
                debug.log(
                    f"GGUF precision path: {qtypes_str} → FP16 (preserve) → BF16/FP32 (compute)", 
                    category="precision"
                )
            else:
                debug.log(
                    "GGUF precision: Dequantizing to FP16 first, then converting to compute dtype", 
                    category="precision"
                )
        else:
            debug.log("Warning: No layers were replaced despite having quantized parameters", 
                     level="WARNING", category="dit", force=True)
    
    return model


def _validate_gguf_architecture(state: Dict[str, torch.Tensor], 
                                model_state: Dict[str, torch.Tensor], debug: Debug) -> None:
    """
    Validate GGUF model architecture matches target model.
    
    Raises:
        ValueError: If architecture mismatch is detected
    """
    key_params = [
        "blocks.0.attn.proj_qkv.vid.weight",
        "blocks.0.attn.proj_qkv.txt.weight", 
        "blocks.0.mlp.vid.proj_in.weight"
    ]
    
    for key in key_params:
        if key in state and key in model_state:
            model_shape = model_state[key].shape
            gguf_shape = _get_tensor_shape(state[key])
            
            if model_shape != gguf_shape:
                # Check if it's just a quantization difference
                if hasattr(state[key], 'tensor_shape') and state[key].tensor_shape == model_shape:
                    continue
                    
                raise ValueError(
                    f"GGUF model architecture mismatch: This GGUF model is incompatible with the current architecture.\n\n"
                    f"Detected mismatch:\n"
                    f"  Parameter: {key}\n"
                    f"  Expected shape: {model_shape}\n"
                    f"  GGUF shape: {gguf_shape}\n\n"
                    f"Possible solutions:\n"
                    f"1. Use a GGUF model that matches the current architecture\n"
                    f"2. Try using a regular FP16 model instead\n"
                    f"3. Verify you're using the correct model variant (3B vs 7B)"
                )
    
    debug.log(f"Architecture check complete, no shape mismatch", category="success")


def _apply_gguf_parameters(model: torch.nn.Module, state: Dict[str, torch.Tensor], 
                           model_state: Dict[str, torch.Tensor], debug: Debug) -> Dict[str, Any]:
    """
    Apply GGUF parameters to model, handling both meta and materialized models.
    
    Returns:
        Statistics dictionary with loaded count, quantized count, and parameter names
    """
    loaded_names = set()
    quantized_count = 0
    
    for name, param in state.items():
        if name not in model_state:
            continue
            
        model_param = model_state[name]
        param_shape = _get_tensor_shape(param)
        
        if param_shape != model_param.shape:
            debug.log(f"Unexpected shape mismatch for {name}: {param_shape} vs {model_param.shape}", 
                     level="ERROR", category="dit", force=True)
            raise ValueError(f"Shape mismatch for parameter {name}")
        
        # Apply parameter based on device type
        with torch.no_grad():
            if model_param.device.type == 'meta':
                _set_parameter_on_meta_model(model, name, param, debug)
            else:
                _set_parameter_on_materialized_model(model, name, param, debug)
        
        loaded_names.add(name)
        if _is_quantized_tensor(param):
            quantized_count += 1
    
    return {
        'loaded': len(loaded_names), 
        'quantized': quantized_count, 
        'loaded_names': loaded_names
    }


def _set_parameter_on_meta_model(model: torch.nn.Module, param_name: str, 
                                 param_value: torch.Tensor, debug: Debug) -> None:
    """Set parameter on meta device model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    new_param = _create_gguf_parameter(param_value, debug)
    setattr(module, attr_name, new_param)


def _set_parameter_on_materialized_model(model: torch.nn.Module, param_name: str, 
                                         param_value: torch.Tensor, debug: Debug) -> None:
    """Set parameter on already materialized model."""
    module, attr_name = _navigate_to_parameter(model, param_name)
    
    if _is_quantized_tensor(param_value):
        # For quantized tensors, replace with wrapped parameter
        new_param = _create_gguf_parameter(param_value, debug)
        setattr(module, attr_name, new_param)
    else:
        # For regular tensors, just copy
        existing_param = getattr(module, attr_name)
        existing_param.copy_(param_value)


def _navigate_to_parameter(model: torch.nn.Module, param_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Navigate to the module containing a parameter.
    
    Args:
        model: Root model
        param_path: Dot-separated path to parameter
        
    Returns:
        Tuple of (parent module, parameter name)
    """
    path_parts = param_path.split('.')
    module = model
    
    # Navigate to parent module
    for part in path_parts[:-1]:
        module = getattr(module, part)
    
    return module, path_parts[-1]


def _create_gguf_parameter(tensor: torch.Tensor, debug: Optional[Debug]) -> torch.nn.Parameter:
    """
    Create a parameter from a GGUF tensor, preserving quantization info.
    
    Args:
        tensor: GGUF tensor (may be quantized)
        debug: Debug instance for logging
        
    Returns:
        Parameter with GGUF attributes and dequantize method if quantized
    """
    param = torch.nn.Parameter(tensor, requires_grad=False)
    
    # Preserve GGUF attributes if present
    if hasattr(tensor, 'tensor_type'):
        param.tensor_type = tensor.tensor_type
        param.tensor_shape = tensor.tensor_shape
        
        # Add dequantize method for runtime dequantization
        param.gguf_dequantize = _create_dequantize_method(tensor, debug)
    
    return param


def _create_dequantize_method(tensor: torch.Tensor, debug: Optional[Debug]) -> callable:
    """
    Create a dequantization method for a GGUF tensor.
    
    Args:
        tensor: GGUF quantized tensor with tensor_type and tensor_shape attributes
        debug: Debug instance
        
    Returns:
        Callable dequantization method
    """
    def dequantize(device: Optional[torch.device] = None, 
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize GGUF tensor on demand."""
        if hasattr(tensor, 'dequantize'):
            return tensor.dequantize(device, dtype)
        
        try:
            # Fallback to manual dequantization using gguf library
            numpy_data = tensor.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, tensor.tensor_type)
            result = torch.from_numpy(dequantized).to(device or tensor.device, dtype)
            result.requires_grad_(False)
            return result.reshape(tensor.tensor_shape)
        except Exception as e:
            if debug:
                debug.log(f"Warning: Could not dequantize tensor: {e}", level="WARNING", category="dit", force=True)
            return tensor.to(device or tensor.device, dtype)
    
    return dequantize


def _get_tensor_shape(tensor: torch.Tensor) -> torch.Size:
    """Get the logical shape of a tensor (handling GGUF quantized tensors)."""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    return tensor.shape


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is GGUF quantized."""
    return hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape')


def _report_parameter_mismatches(state: Dict[str, torch.Tensor], 
                                 model_state: Dict[str, torch.Tensor], 
                                 loaded_names: set, debug: Debug) -> None:
    """Report any parameter mismatches between GGUF and model."""
    # Check for unmatched GGUF parameters
    unmatched = [name for name in state if name not in model_state]
    if unmatched:
        debug.log(f"Warning: {len(unmatched)} parameters from GGUF not found in model", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"  First few unmatched: {unmatched[:5]}", level="WARNING", category="dit", force=True)
    
    # Check for missing model parameters  
    missing = [name for name in model_state if name not in loaded_names]
    if missing:
        debug.log(f"Warning: {len(missing)} model parameters not loaded from GGUF", 
                 level="WARNING", category="dit", force=True)
        debug.log(f"  First few missing: {missing[:5]}", level="WARNING", category="dit", force=True)


def _initialize_meta_buffers_wrapped(model: torch.nn.Module, target_device: torch.device, debug: Debug) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = _initialize_meta_buffers(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def _initialize_meta_buffers(model: torch.nn.Module, target_device: torch.device, debug: Optional[Debug]) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization (torch.device object)
        debug: Debug instance for logging
        
    Returns:
        Number of buffers initialized
    """
    initialized_count = 0
    
    # Simply initialize all meta device buffers to zeros on target device
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model
            
            # Create a zero tensor of the same shape on target device
            # This is safe for all non-persistent buffers (caches, dummy tensors, etc.)
            initialized_buffer = torch.zeros_like(buffer, device=target_device)
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
            initialized_count += 1
    
    return initialized_count


def apply_model_specific_config(model: torch.nn.Module, runner: VideoDiffusionInfer, 
                                config: OmegaConf, is_dit: bool, 
                                debug: Optional[Debug]) -> torch.nn.Module:
    """
    Apply model-specific configurations (FP8, BlockSwap, torch.compile).
    
    This function is idempotent and can be safely called on both newly materialized
    and already-configured models. It checks state flags to determine what needs
    to be applied.
    
    Critical: For DiT, BlockSwap must be applied BEFORE torch.compile.
    torch.compile captures the computational graph, so any wrapping done
    after compilation (like BlockSwap's forward wrapping) won't work.
    
    Args:
        model: Loaded model instance
        runner: Runner to attach model to
        config: Full configuration object
        is_dit: Whether this is a DiT model (vs VAE)
        debug: Debug instance
        
    Returns:
        Configured model with BlockSwap and torch.compile applied if configured
    """
    if is_dit:
        # DiT-specific
        # Apply FP8 compatibility wrapper 
        if not isinstance(model, FP8CompatibleDiT):
            debug.log("Applying FP8/RoPE compatibility wrapper to DiT model", category="setup")
            debug.start_timer("FP8CompatibleDiT")
            model = FP8CompatibleDiT(model, skip_conversion=False, debug=debug)
            debug.end_timer("FP8CompatibleDiT", "FP8/RoPE compatibility wrapper application")
        else:
            debug.log("Reusing existing FP8/RoPE compatibility wrapper", category="reuse")

        # Apply BlockSwap before torch.compile (only if not already active)
        # BlockSwap wraps forward methods, and torch.compile needs to capture the wrapped version
        if hasattr(runner, '_dit_block_swap_config') and runner._dit_block_swap_config:
            # Check if BlockSwap needs to be applied
            needs_blockswap = not (hasattr(runner, '_blockswap_active') and runner._blockswap_active)
            if needs_blockswap:
                runner.dit = model
                apply_block_swap_to_dit(runner, runner._dit_block_swap_config, debug)
                # Mark as active after successful application
                runner._blockswap_active = True
        
        # Apply torch.compile after BlockSwap (only if not already compiled)
        # Check: model has _orig_mod attribute means it's already torch.compiled
        if hasattr(runner, '_dit_compile_args') and runner._dit_compile_args:
            if not hasattr(model, '_orig_mod'):
                model = _apply_torch_compile(model, runner._dit_compile_args, "DiT", debug)
        
        runner.dit = model

        # Clear the config application flag after successful application
        if hasattr(runner, '_dit_config_needs_application'):
            runner._dit_config_needs_application = False
        
    else:
        # VAE-specific configurations
        # Set to eval mode (no gradients needed for inference)
        if model.training:
            debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
            debug.start_timer("model_requires_grad")
            model.requires_grad_(False).eval()
            debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available - always apply as it's lightweight
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Set memory limits if available - always apply to ensure limits are set
        if hasattr(model, "set_memory_limit") and hasattr(config.vae, "memory_limit"):
            debug.log("Configuring VAE memory limits for causal convolutions", category="vae")
            debug.start_timer("vae_set_memory_limit")
            model.set_memory_limit(**config.vae.memory_limit)
            debug.end_timer("vae_set_memory_limit", "VAE memory limits configured")

        # Apply torch.compile if configured (only if not already compiled)
        if hasattr(runner, '_vae_compile_args') and runner._vae_compile_args:
            # Check if encoder/decoder already compiled (have _orig_mod)
            encoder_compiled = hasattr(model, 'encoder') and hasattr(model.encoder, '_orig_mod')
            decoder_compiled = hasattr(model, 'decoder') and hasattr(model.decoder, '_orig_mod')
            
            if not (encoder_compiled and decoder_compiled):
                model = _apply_vae_submodule_compile(model, runner._vae_compile_args, debug)
            else:
                debug.log("Reusing existing torch.compile for VAE submodules", category="reuse")
        
        # Propagate debug instance to submodules
        model.debug = debug
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
        
        # Clear the config application flag after successful application
        if hasattr(runner, '_vae_config_needs_application'):
            runner._vae_config_needs_application = False
    
    return model


def _configure_torch_compile(compile_args: Dict[str, Any], model_type: str, 
                            debug: Optional[Debug]) -> Tuple[Dict[str, Any], bool]:
    """
    Extract and configure torch.compile settings.
    
    Centralizes common configuration logic for both full model and submodule compilation.
    
    Args:
        compile_args: Compilation configuration dictionary
        model_type: Model type string for logging (e.g., "DiT", "VAE")
        debug: Debug instance for logging
        
    Returns:
        Tuple of (compile_settings dict, success boolean)
        compile_settings contains: backend, mode, fullgraph, dynamic
    """
    # Extract settings with defaults
    settings = {
        'backend': compile_args.get('backend', 'inductor'),
        'mode': compile_args.get('mode', 'default'),
        'fullgraph': compile_args.get('fullgraph', False),
        'dynamic': compile_args.get('dynamic', False)
    }
    
    dynamo_cache_size_limit = compile_args.get('dynamo_cache_size_limit', 64)
    dynamo_recompile_limit = compile_args.get('dynamo_recompile_limit', 128)
    
    # Log compilation configuration
    debug.log(f"Configuring torch.compile for {model_type}...", category="setup", force=True)
    debug.log(f"  Backend: {settings['backend']} | Mode: {settings['mode']} | "
             f"Fullgraph: {settings['fullgraph']} | Dynamic: {settings['dynamic']}", 
             category="setup")
    debug.log(f"  Dynamo cache_size_limit: {dynamo_cache_size_limit} | "
             f"recompile_limit: {dynamo_recompile_limit}", category="setup")
    
    # Configure torch._dynamo settings
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        torch._dynamo.config.recompile_limit = dynamo_recompile_limit
        debug.log(f"  torch._dynamo configured successfully", category="success")
        return settings, True
    except Exception as e:
        debug.log(f"  Could not configure torch._dynamo settings: {e}", 
                 level="WARNING", category="setup", force=True)
        return settings, False


def _apply_torch_compile(model: torch.nn.Module, compile_args: Dict[str, Any], 
                        model_type: str, debug: Optional[Debug]) -> torch.nn.Module:
    """
    Apply torch.compile to entire model with configured settings.
    
    Args:
        model: Model to compile
        compile_args: Compilation configuration
        model_type: "DiT" or "VAE" for logging
        debug: Debug instance
        
    Returns:
        Compiled model, or original model if compilation fails
    """
    try:
        # Configure compilation settings
        settings, _ = _configure_torch_compile(compile_args, model_type, debug)
        
        # Compile entire model
        debug.start_timer(f"{model_type.lower()}_compile")
        compiled_model = torch.compile(model, **settings)
        debug.end_timer(f"{model_type.lower()}_compile", 
                       f"{model_type} model wrapped for compilation", force=True)
        debug.log(f"  Actual compilation will happen on first batch (expect initial delay, then speedup)", category="info")
        
        return compiled_model
        
    except Exception as e:
        debug.log(f"torch.compile failed for {model_type}: {e}", 
                 level="WARNING", category="setup", force=True)
        debug.log(f"  Falling back to uncompiled model", 
                 level="WARNING", category="setup", force=True)
        return model


def _disable_compile_for_dynamic_modules(module: torch.nn.Module) -> None:
    """
    Mark modules with dynamic shapes to be excluded from torch.compile.
    This prevents recompilation issues with variable tensor sizes.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, InflatedCausalConv3d):
            # Mark module to skip compilation
            submodule._dynamo_disable = True


def _apply_vae_submodule_compile(model: torch.nn.Module, compile_args: Dict[str, Any], 
                                 debug: Optional[Debug]) -> torch.nn.Module:
    """
    Apply torch.compile to VAE core submodules instead of entire model.
    
    The VAE's high-level encode/decode methods contain complex control flow
    (temporal slicing, tiling, stateful memory management) that prevents 
    torch.compile from optimizing effectively. Instead, we compile only the 
    core neural networks (encoder, decoder) which have straightforward forward 
    passes suitable for compilation.
    
    Note: quant_conv and post_quant_conv are disabled (None) in the current VAE 
    architecture/yaml, so they are not compiled.
    
    Args:
        model: VAE model instance
        compile_args: Compilation configuration
        debug: Debug instance
        
    Returns:
        VAE model with compiled submodules
    """
    try:
        # Configure compilation settings
        settings, _ = _configure_torch_compile(compile_args, "VAE submodules", debug)
        
        # Compile submodules
        compiled_modules = []
        debug.start_timer("vae_submodule_compile")
        
        if hasattr(model, 'encoder') and model.encoder is not None:
            # Disable compilation for InflatedCausalConv3d modules due to dynamic shapes
            _disable_compile_for_dynamic_modules(model.encoder)
            model.encoder = torch.compile(model.encoder, **settings)
            compiled_modules.append('encoder')
            debug.log(f"  VAE encoder found and added to compilation queue", category="success")

        if hasattr(model, 'decoder') and model.decoder is not None:
            # Disable compilation for InflatedCausalConv3d modules due to dynamic shapes  
            _disable_compile_for_dynamic_modules(model.decoder)
            model.decoder = torch.compile(model.decoder, **settings)
            compiled_modules.append('decoder')
            debug.log(f"  VAE decoder found and added to compilation queue", category="success")
        
        debug.end_timer("vae_submodule_compile", 
                       f"VAE submodules compiled: {', '.join(compiled_modules)}", force=True)
        debug.log(f"  Actual compilation will happen on first batch (expect initial delay, then speedup)", category="info")
        
        return model
        
    except Exception as e:
        debug.log(f"torch.compile failed for VAE submodules: {e}", 
                 level="WARNING", category="setup", force=True)
        debug.log(f"  Falling back to uncompiled VAE", 
                 level="WARNING", category="setup", force=True)
        return model


def _propagate_debug_to_modules(module: torch.nn.Module, debug: Optional[Debug]) -> None:
    """
    Propagate debug instance to specific submodules that need it.
    Only targets modules that actually use debug to avoid unnecessary memory overhead.
    
    Args:
        module: Parent module to propagate through
        debug: Debug instance to attach
    """
    if debug is None:
        return  # Early exit if no debug instance
        
    target_modules = {'ResnetBlock3D', 'Upsample3D', 'InflatedCausalConv3d', 'GroupNorm'}
    
    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in target_modules:
            if not hasattr(submodule, 'debug'):  # Only set if not already present
                submodule.debug = debug