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
from typing import Dict, Optional, Tuple, Any

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

from ..utils.constants import get_script_directory, find_model_file, suppress_tensor_warnings
from ..optimization.memory_manager import clear_memory, cleanup_dit, cleanup_vae
from ..optimization.compatibility import FP8CompatibleDiT
from ..common.config import load_config, create_object
from .infer import VideoDiffusionInfer
from ..optimization.blockswap import apply_block_swap_to_dit, cleanup_blockswap
from ..common.distributed import get_device
from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

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
    if config is None:
        return "disabled"
    
    blocks_to_swap = config.get("blocks_to_swap", 0)
    swap_io_components = config.get("swap_io_components", False)
    
    # Early return only if both block swap and I/O swap are disabled
    if blocks_to_swap <= 0 and not swap_io_components:
        return "disabled"
    
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


def _check_model_reload_status(cached_runner: VideoDiffusionInfer, 
                               cache_dit: bool, cache_vae: bool,
                               dit_model: str, vae_model: str,
                               debug: Debug) -> Tuple[bool, bool]:
    """
    Determine which models need reloading.
    
    Returns:
        Tuple[bool, bool]: (need_reload_dit, need_reload_vae)
    """
    need_reload_dit = not cache_dit or not hasattr(cached_runner, 'dit') or cached_runner.dit is None
    need_reload_vae = not cache_vae or not hasattr(cached_runner, 'vae') or cached_runner.vae is None
    
    if need_reload_dit:
        reason = "cache disabled" if not cache_dit else "not in memory"
        debug.log(f"DiT will be reloaded ({reason}): {dit_model}", category="cache", force=True)
    else:
        cached_dit_name = getattr(cached_runner, '_dit_model_name', dit_model)
        debug.log(f"DiT cached in RAM - reusing: {cached_dit_name}", category="reuse", force=True)
        
    if need_reload_vae:
        reason = "cache disabled" if not cache_vae else "not in memory"
        debug.log(f"VAE will be reloaded ({reason}): {vae_model}", category="cache", force=True)
    else:
        cached_vae_name = getattr(cached_runner, '_vae_model_name', vae_model)
        debug.log(f"VAE cached in RAM - reusing: {cached_vae_name}", category="reuse", force=True)
    
    return need_reload_dit, need_reload_vae


def _reload_missing_models(cached_runner: VideoDiffusionInfer,
                          need_reload_dit: bool, need_reload_vae: bool,
                          dit_model: str, vae_model: str, base_cache_dir: str,
                          block_swap_config: Optional[Dict[str, Any]],
                          debug: Debug) -> Optional[VideoDiffusionInfer]:
    """
    Reload models that are missing from cache.
    Cleanup old materialized models before creating new meta structures.
    
    Returns:
        cached_runner if successful, None if config is missing and full recreation needed
    """
    if not (need_reload_dit or need_reload_vae):
        return cached_runner
    
    # Verify config exists
    if not hasattr(cached_runner, 'config') or cached_runner.config is None:
        debug.log("Config missing - full runner recreation required", category="cache", force=True)
        return None
    
    config = cached_runner.config
    
    # Cleanup old models BEFORE creating new ones
    # This releases old materialized models so new meta structures can be created cleanly
    if need_reload_dit and hasattr(cached_runner, 'dit') and cached_runner.dit is not None:
        debug.log("Cleaning old DiT model before reload", category="cache", force=True)
        cleanup_dit(runner=cached_runner, debug=debug, keep_model_in_ram=False)
    
    if need_reload_vae and hasattr(cached_runner, 'vae') and cached_runner.vae is not None:
        debug.log("Cleaning old VAE model before reload", category="cache", force=True)
        cleanup_vae(runner=cached_runner, debug=debug, keep_model_in_ram=False)
    
    # Now create new meta structures (old models are cleaned up)
    if need_reload_dit:
        dit_checkpoint_path = find_model_file(dit_model, base_cache_dir)
        debug.log(f"Preparing DiT structure: {dit_model}", category="dit")
        cached_runner = prepare_model_structure(cached_runner, "dit", dit_checkpoint_path, 
                                               config, block_swap_config, debug)
    
    if need_reload_vae:
        vae_checkpoint_path = find_model_file(vae_model, base_cache_dir)
        debug.log(f"Preparing VAE structure: {vae_model}", category="vae")
        cached_runner = prepare_model_structure(cached_runner, "vae", vae_checkpoint_path, 
                                               config, None, debug)
    
    return cached_runner


def _update_model_configs_inplace(
    cached_runner: 'VideoDiffusionInfer',
    dit_model: str,
    vae_model: str,
    block_swap_config: Optional[Dict[str, Any]],
    torch_compile_args_dit: Optional[Dict[str, Any]],
    torch_compile_args_vae: Optional[Dict[str, Any]],
    base_cache_dir: str,
    debug: Debug
) -> Optional['VideoDiffusionInfer']:
    """
    Update model configurations in-place without recreating models when possible.
    Only returns None if the actual model files changed (requiring full recreation).
    
    Args:
        cached_runner: Existing runner with cached models
        dit_model: DiT model filename
        vae_model: VAE model filename  
        block_swap_config: BlockSwap configuration
        torch_compile_args_dit: DiT torch.compile args
        torch_compile_args_vae: VAE torch.compile args
        base_cache_dir: Base directory for models
        debug: Debug instance
        
    Returns:
        cached_runner if configs updated successfully, None if models need recreation
    """
    if not cached_runner or not hasattr(cached_runner, 'config'):
        return None
    
    # Check if model FILES changed (this requires full recreation)
    current_dit = getattr(cached_runner, '_dit_model_name', None)
    current_vae = getattr(cached_runner, '_vae_model_name', None)
    
    if current_dit != dit_model or current_vae != vae_model:
        return None
    
    # Handle DiT configuration updates
    if hasattr(cached_runner, 'dit') and cached_runner.dit is not None:
        dit_updated = _update_dit_config_inplace(
            cached_runner, block_swap_config, torch_compile_args_dit, debug
        )
        if not dit_updated:
            return None
    
    # Handle VAE configuration updates  
    if hasattr(cached_runner, 'vae') and cached_runner.vae is not None:
        vae_updated = _update_vae_config_inplace(
            cached_runner, torch_compile_args_vae, debug
        )
        if not vae_updated:
            return None
    
    return cached_runner


def _update_dit_config_inplace(
    cached_runner: 'VideoDiffusionInfer',
    block_swap_config: Optional[Dict[str, Any]],
    torch_compile_args: Optional[Dict[str, Any]],
    debug: Debug
) -> bool:
    """Update DiT configuration in-place. Returns True if successful."""
    model = cached_runner.dit
    
    # Check if model is on meta device (not materialized yet)
    try:
        param_device = next(model.parameters()).device
        if param_device.type == 'meta':
            debug.log("DiT on meta device, materializing before config update", category="dit")
            materialize_model(cached_runner, "dit", "cpu", cached_runner.config, 
                            preserve_vram=True, debug=debug)
            model = cached_runner.dit
    except StopIteration:
        pass
    
    # Get cached configurations
    cached_compile_args = getattr(cached_runner, '_dit_compile_args', None)
    cached_blockswap_config = getattr(cached_runner, '_dit_block_swap_config', None)
    
    # Check if configurations changed
    compile_changed = not _configs_equal(cached_compile_args, torch_compile_args)
    blockswap_changed = not _configs_equal(cached_blockswap_config, block_swap_config)

    # If nothing changed, reuse model as-is
    if not compile_changed and not blockswap_changed:
        debug.log("DiT configuration unchanged, reusing cached model", category="dit")
        return True
    
    # Build detailed configuration change description
    config_changes = []
    if compile_changed:
        old_desc = _describe_compile_config(cached_compile_args)
        new_desc = _describe_compile_config(torch_compile_args)
        config_changes.append(f"torch.compile: {old_desc} → {new_desc}")
    if blockswap_changed:
        old_desc = _describe_blockswap_config(cached_blockswap_config)
        new_desc = _describe_blockswap_config(block_swap_config)
        config_changes.append(f"BlockSwap: {old_desc} → {new_desc}")
    
    debug.log(f"DiT configuration changed, updating in-place:", category="dit", force=True)
    for change in config_changes:
        debug.log(f"  {change}", category="dit", force=True)
    
    # Remove existing torch.compile wrapper if present and compile config changed
    if compile_changed and hasattr(model, '_orig_mod'):
        debug.log("Removing existing torch.compile from DiT", category="setup")
        model = model._orig_mod
    
    # Get base model through FP8 wrapper if present
    base_model = model.dit_model if hasattr(model, 'dit_model') else model
    
    # Handle BlockSwap configuration changes
    if blockswap_changed:
        # Determine change type from config comparison
        old_blocks = cached_blockswap_config.get("blocks_to_swap", 0) if cached_blockswap_config else 0
        old_swap_io = cached_blockswap_config.get("swap_io_components", False) if cached_blockswap_config else False
        new_blocks = block_swap_config.get("blocks_to_swap", 0) if block_swap_config else 0
        new_swap_io = block_swap_config.get("swap_io_components", False) if block_swap_config else False
        
        # Check if old config had ANY BlockSwap features (blocks or I/O components)
        had_blockswap = old_blocks > 0 or old_swap_io
        has_blockswap = new_blocks > 0 or new_swap_io
        
        # If old config had BlockSwap features, clean them up first
        if had_blockswap:
            if not has_blockswap:
                # Disabling BlockSwap completely
                debug.log("Disabling BlockSwap completely", category="blockswap")
                cleanup_blockswap(runner=cached_runner, keep_state_for_cache=False)
            else:
                # Reconfiguring BlockSwap (will reapply below)
                debug.log("Reconfiguring BlockSwap", category="blockswap")
                cleanup_blockswap(runner=cached_runner, keep_state_for_cache=True)
        
        cached_runner._blockswap_active = False
    
    # Apply new BlockSwap if configured (before torch.compile)
    if block_swap_config and (block_swap_config.get("blocks_to_swap", 0) > 0 or 
                              block_swap_config.get("swap_io_components", False)):
        cached_runner.dit = model  # Temporarily set for BlockSwap
        apply_block_swap_to_dit(cached_runner, block_swap_config, debug)
        model = cached_runner.dit  # Get potentially wrapped model
    
    # Apply torch.compile if configured (after BlockSwap)
    if torch_compile_args:
        model = _apply_torch_compile(model, torch_compile_args, "DiT", debug)
    
    cached_runner.dit = model
    cached_runner._dit_compile_args = torch_compile_args
    cached_runner._dit_block_swap_config = block_swap_config
    return True


def _update_vae_config_inplace(
    cached_runner: 'VideoDiffusionInfer',
    torch_compile_args: Optional[Dict[str, Any]],
    debug: Debug
) -> bool:
    """Update VAE configuration in-place. Returns True if successful."""
    model = cached_runner.vae
    
    # Check if model is on meta device
    try:
        param_device = next(model.parameters()).device
        if param_device.type == 'meta':
            debug.log("VAE on meta device, materializing before config update", category="vae")
            materialize_model(cached_runner, "vae", "cpu", cached_runner.config, 
                            preserve_vram=True, debug=debug)
            model = cached_runner.vae
    except StopIteration:
        pass
    
    # Get cached configuration
    cached_compile_args = getattr(cached_runner, '_vae_compile_args', None)
    
    # Check if configuration changed
    compile_changed = not _configs_equal(cached_compile_args, torch_compile_args)
    
    # If nothing changed, reuse model as-is
    if not compile_changed:
        debug.log("VAE configuration unchanged, reusing cached model", category="vae")
        cached_runner._vae_compile_args = torch_compile_args
        return True
    
    # Build detailed configuration change description
    old_desc = _describe_compile_config(cached_compile_args)
    new_desc = _describe_compile_config(torch_compile_args)
    
    debug.log(f"VAE configuration changed, updating in-place:", category="vae", force=True)
    debug.log(f"  torch.compile: {old_desc} → {new_desc}", category="vae", force=True)
    
    # Unwrap compiled VAE submodules if present
    if hasattr(model, 'encoder') and hasattr(model.encoder, '_orig_mod'):
        debug.log("Unwrapping compiled VAE encoder", category="setup")
        model.encoder = model.encoder._orig_mod
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, '_orig_mod'):
        debug.log("Unwrapping compiled VAE decoder", category="setup")
        model.decoder = model.decoder._orig_mod
    
    # Reapply torch.compile with new configuration if configured
    if torch_compile_args:
        model = _apply_vae_submodule_compile(model, torch_compile_args, debug)
    
    cached_runner.vae = model
    cached_runner._vae_compile_args = torch_compile_args
    return True


def configure_runner(dit_model: str, vae_model: str, base_cache_dir: str, preserve_vram: bool, debug: Debug,
                    cache_model_dit: bool = False, cache_model_vae: bool = False,
                    block_swap_config: Optional[Dict[str, Any]] = None,
                    encode_tiled: bool = False, encode_tile_size: Optional[Tuple[int, int]] = None,
                    encode_tile_overlap: Optional[Tuple[int, int]] = None,
                    decode_tiled: bool = False, decode_tile_size: Optional[Tuple[int, int]] = None,
                    decode_tile_overlap: Optional[Tuple[int, int]] = None,
                    torch_compile_args_dit: Optional[Dict[str, Any]] = None,
                    torch_compile_args_vae: Optional[Dict[str, Any]] = None,
                    cached_runner: Optional['VideoDiffusionInfer'] = None) -> 'VideoDiffusionInfer':
    """
    Configure SeedVR2 inference runner with model loading and optimization.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        base_cache_dir: Base directory containing model files
        preserve_vram: Whether to preserve VRAM by offloading models between pipeline steps
        debug: Debug instance for logging
        cache_model_dit: Whether to cache DiT model in RAM between runs
        cache_model_vae: Whether to cache VAE model in RAM between runs
        block_swap_config: Optional BlockSwap configuration for DiT memory optimization
        encode_tiled: Enable tiled encoding to reduce VRAM during VAE encoding
        encode_tile_size: Tile size for encoding (height, width)
        encode_tile_overlap: Tile overlap for encoding (height, width)
        decode_tiled: Enable tiled decoding to reduce VRAM during VAE decoding
        decode_tile_size: Tile size for decoding (height, width)
        decode_tile_overlap: Tile overlap for decoding (height, width)
        torch_compile_args_dit: Optional torch.compile configuration for DiT model
        torch_compile_args_vae: Optional torch.compile configuration for VAE model
        cached_runner: Optional cached runner to reuse (passed if either model is cached)
        
    Returns:
        VideoDiffusionInfer: Configured runner instance ready for inference
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic config loading based on model type (3B vs 7B)
        - Automatic import path resolution for different environments
        - Optional torch.compile optimization for 20-40% speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and RoPE cache pre-initialization
        - BlockSwap integration for large model support on limited VRAM
    """
    # Check if debug instance is available
    if debug is None:
        raise ValueError("Debug instance must be provided to configure_runner")
    
    # Check if we can reuse the cached runner (if either model is cached)
    if cached_runner and (cache_model_dit or cache_model_vae):
        debug.start_timer("cache_reuse")
        
        # 1. Update VAE tiling parameters (runtime, no reload needed)
        for key, value in {
            'encode_tiled': encode_tiled,
            'encode_tile_size': encode_tile_size,
            'encode_tile_overlap': encode_tile_overlap,
            'decode_tiled': decode_tiled,
            'decode_tile_size': decode_tile_size,
            'decode_tile_overlap': decode_tile_overlap
        }.items():
            setattr(cached_runner, key, value)
        
        # 2. Check model status first (for logging purposes)
        need_reload_dit, need_reload_vae = _check_model_reload_status(
            cached_runner, cache_model_dit, cache_model_vae, 
            dit_model, vae_model, debug
        )

        # 3. Try to update configurations in-place if models are the same files
        if not need_reload_dit and not need_reload_vae:
            updated_runner = _update_model_configs_inplace(
                cached_runner, dit_model, vae_model, block_swap_config,
                torch_compile_args_dit, torch_compile_args_vae, base_cache_dir, debug
            )
            
            if updated_runner is not None:
                # Configs updated successfully without reloading
                updated_runner.debug = debug
                debug.end_timer("cache_reuse", "Cache reuse with in-place config updates")
                return updated_runner
        
        # 4. Reload models that are missing or changed
        if need_reload_dit or need_reload_vae:
            cached_runner = _reload_missing_models(
                cached_runner, need_reload_dit, need_reload_vae,
                dit_model, vae_model, base_cache_dir, block_swap_config, debug
            )
            
            if cached_runner is not None:
                # Update configurations for models that DIDN'T reload
                # This ensures consistent logging for unchanged models
                if not need_reload_dit and hasattr(cached_runner, 'dit') and cached_runner.dit is not None:
                    _update_dit_config_inplace(
                        cached_runner, block_swap_config, torch_compile_args_dit, debug
                    )
                else:
                    # Model was reloaded, just set config attributes
                    cached_runner._dit_compile_args = torch_compile_args_dit
                    cached_runner._dit_block_swap_config = block_swap_config
                
                if not need_reload_vae and hasattr(cached_runner, 'vae') and cached_runner.vae is not None:
                    _update_vae_config_inplace(
                        cached_runner, torch_compile_args_vae, debug
                    )
                else:
                    # Model was reloaded, just set config attributes
                    cached_runner._vae_compile_args = torch_compile_args_vae
                
                cached_runner.debug = debug
                debug.end_timer("cache_reuse", "Cache reuse complete")
                return cached_runner
    
    # Full runner creation
    debug.log(f"Creating new runner: DiT={dit_model}, VAE={vae_model}", 
             category="runner", force=True)
    
    # If we reach here, create a new runner
    debug.start_timer("config_load")

    # Select config based on model type   
    if "7b" in dit_model:
        config_path = os.path.join(script_directory, './configs_7b', 'main.yaml')
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
    
    config = load_config(config_path)
    debug.end_timer("config_load", "Config loading")

    # Load and configure VAE with additional parameters
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    debug.start_timer("vae_config_load")
    vae_config = OmegaConf.load(vae_config_path)
    debug.end_timer("vae_config_load", "VAE configuration YAML parsed from disk")
    
    debug.start_timer("vae_config_set")
    # Configure VAE parameters
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    debug.end_timer("vae_config_set", f"VAE downsample factors configuration")
    
    # Merge additional VAE config with main config (preserving __object__ from main config)
    debug.start_timer("vae_config_merge")
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    debug.end_timer("vae_config_merge", "VAE config merged with main pipeline config")
    
    debug.start_timer("runner_video_infer")
    # Create runner
    runner = VideoDiffusionInfer(config, debug, 
                                encode_tiled=encode_tiled, 
                                encode_tile_size=encode_tile_size, 
                                encode_tile_overlap=encode_tile_overlap,
                                decode_tiled=decode_tiled,
                                decode_tile_size=decode_tile_size,
                                decode_tile_overlap=decode_tile_overlap)
    OmegaConf.set_readonly(runner.config, False)
    runner._dit_compile_args = torch_compile_args_dit
    runner._vae_compile_args = torch_compile_args_vae
    debug.end_timer("runner_video_infer", "Video diffusion inference runner initialization")
    
    # Set device
    device = str(get_device())
    #if torch.mps.is_available():
    #    device = "mps"
    
    debug.start_timer("model_structures")
    # Prepare DiT structure
    dit_checkpoint_path = find_model_file(dit_model, base_cache_dir)
    runner = prepare_model_structure(runner, "dit", dit_checkpoint_path, config, 
                                    block_swap_config, debug)
    
    # Set VAE dtype for MPS compatibility
    if torch.mps.is_available():
        original_vae_dtype = config.vae.dtype
        config.vae.dtype = "bfloat16"
        debug.log(f"MPS detected: Setting VAE dtype from {original_vae_dtype} to {config.vae.dtype} for compatibility", 
                    category="precision", force=True)
    
    # Prepare VAE structure
    vae_checkpoint_path = find_model_file(vae_model, base_cache_dir)
    runner = prepare_model_structure(runner, "vae", vae_checkpoint_path, config, 
                                    None, debug)
    
    debug.log(f"VAE downsample factors configured (spatial: {spatial_downsample_factor}x, temporal: {temporal_downsample_factor}x)", category="vae")
    
    debug.end_timer("model_structures", "Model structures prepared")
    
    debug.start_timer("vae_memory_limit")
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    debug.end_timer("vae_memory_limit", "VAE memory limit set")
    
    # Store debug instance on runner for consistent access
    runner.debug = debug
    
    return runner


def load_quantized_state_dict(checkpoint_path: str, device: str = "cpu", debug: Optional[Debug] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dictionary with optimal memory management
    
    Args:
        checkpoint_path (str): Path to checkpoint file (.safetensors or .pth)
        device (str/torch.device): Target device for tensor placement
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
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
        state = load_safetensors_file(checkpoint_path, device=device)
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
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: str, debug: Optional[Debug],
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
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
    
    debug.log(f"Loading {total_tensors} tensors to {device}...", category="dit")
    
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

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device}", category="success")

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


def materialize_model(runner: VideoDiffusionInfer, model_type: str, device: str, 
                     config: OmegaConf, preserve_vram: bool = False,
                     debug: Optional[Debug] = None) -> None:
    """
    Materialize model weights from checkpoint to memory.
    Call this right before the model is needed.
    
    Args:
        runner: Runner with model structure on meta device
        model_type: "dit" or "vae"
        device: Target device for inference
        config: Full configuration
        preserve_vram: Whether to keep on CPU
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
    
    # Determine target device
    blockswap_active = is_dit and block_swap_config and block_swap_config.get("blocks_to_swap", 0) > 0
    use_cpu = preserve_vram or blockswap_active
    target_device = "cpu" if use_cpu else device
    
    # Build reason string for logging
    cpu_reason = ""
    if target_device == "cpu":
        reasons = []
        if blockswap_active:
            reasons.append("BlockSwap")
        if preserve_vram:
            reasons.append("preserve_vram")
        cpu_reason = f" ({', '.join(reasons)})" if reasons else ""
    
    # Start materialization
    debug.start_timer(f"{model_type}_materialize")
    
    # Load weights (this materializes from meta to target device)
    model = _load_model_weights(model, checkpoint_path, target_device, True,
                               model_type_upper, cpu_reason, debug, override_dtype) 
   
    # Apply model-specific configurations (includes BlockSwap and torch.compile)
    model = _apply_model_specific_config(model, runner, config, is_dit, debug)
    
    debug.end_timer(f"{model_type}_materialize", f"{model_type_upper} materialized")
    
    # Clean up checkpoint paths (no longer needed after weights are loaded)
    # Note: Config attributes (_dit_block_swap_config, _dit_compile_args) are preserved
    # for configuration change detection on subsequent runs
    if is_dit:
        runner._dit_checkpoint = None
    else:
        runner._vae_checkpoint = None
        runner._vae_dtype_override = None


def _load_model_weights(model: torch.nn.Module, checkpoint_path: str, target_device: str, 
                        used_meta: bool, model_type: str, cpu_reason: str, 
                        debug: Debug, override_dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Load model weights from checkpoint file with optimized GGUF support.
    
    For meta-initialized models, materializes to target device.
    For standard models, loads weights and applies state dict.
    
    Args:
        model: Model instance (may be on meta device)
        checkpoint_path: Path to checkpoint file
        target_device: Target device for weights
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
    debug.log(f"{action} {model_type} weights to {target_device.upper()}{cpu_reason}: {checkpoint_path}", 
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


def _initialize_meta_buffers_wrapped(model: torch.nn.Module, target_device: str, debug: Debug) -> None:
    """Initialize meta buffers with timing."""
    debug.start_timer("buffer_init")
    initialized = _initialize_meta_buffers(model, target_device, debug)
    if initialized > 0:
        debug.log(f"Initialized {initialized} non-persistent buffers", category="success")
    debug.end_timer("buffer_init", "Buffer initialization")


def _initialize_meta_buffers(model: torch.nn.Module, target_device: str, debug: Optional[Debug]) -> int:
    """
    Initialize any buffers still on meta device after materialization.
    
    Non-persistent buffers aren't included in state_dict and remain on meta
    device after load_state_dict. This function moves them to the target device.
    
    Args:
        model: Model potentially containing meta device buffers
        target_device: Target device for initialization
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


def _apply_model_specific_config(model: torch.nn.Module, runner: VideoDiffusionInfer, 
                                config: OmegaConf, is_dit: bool, 
                                debug: Optional[Debug]) -> torch.nn.Module:
    """
    Apply model-specific configurations and attach to runner.
    
    For DiT, BlockSwap must be applied BEFORE torch.compile.
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
        # DiT-specific: Apply FP8 compatibility wrapper
        if not isinstance(model, FP8CompatibleDiT):
            debug.log("Applying FP8/RoPE compatibility wrapper to DiT model", category="setup")
            debug.start_timer("FP8CompatibleDiT")
            model = FP8CompatibleDiT(model, skip_conversion=False, debug=debug)
            debug.end_timer("FP8CompatibleDiT", "FP8/RoPE compatibility wrapper application")
        
        # Apply BlockSwap BEFORE torch.compile
        # BlockSwap wraps forward methods, and torch.compile needs to capture the wrapped version
        if hasattr(runner, '_dit_block_swap_config') and runner._dit_block_swap_config:
            runner.dit = model
            apply_block_swap_to_dit(runner, runner._dit_block_swap_config, debug)
        
        # Apply torch.compile AFTER BlockSwap so it compiles the wrapped forwards
        if hasattr(runner, '_dit_compile_args') and runner._dit_compile_args:
            model = _apply_torch_compile(model, runner._dit_compile_args, "DiT", debug)
        
        runner.dit = model
        
    else:
        # VAE-specific configurations
        
        # Set to eval mode (no gradients needed for inference)
        debug.log("VAE model set to eval mode (gradients disabled)", category="vae")
        debug.start_timer("model_requires_grad")
        model.requires_grad_(False).eval()
        debug.end_timer("model_requires_grad", "VAE model set to eval mode")
        
        # Configure causal slicing if available
        if hasattr(model, "set_causal_slicing") and hasattr(config.vae, "slicing"):
            debug.log("Configuring VAE causal slicing for temporal processing", category="vae")
            debug.start_timer("vae_set_causal_slicing")
            model.set_causal_slicing(**config.vae.slicing)
            debug.end_timer("vae_set_causal_slicing", "VAE causal slicing configuration")
        
        # Apply torch.compile if configured
        if hasattr(runner, '_vae_compile_args') and runner._vae_compile_args:
            model = _apply_vae_submodule_compile(model, runner._vae_compile_args, debug)
        
        # Propagate debug instance to submodules
        model.debug = debug
        _propagate_debug_to_modules(model, debug)
        runner.vae = model
    
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