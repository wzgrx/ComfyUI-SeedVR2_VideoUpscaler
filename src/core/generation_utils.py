"""
Generation Utilities for SeedVR2

This module provides setup, configuration, and utility functions for the generation pipeline.
These are helper functions used to prepare, configure, and support the video upscaling process,
but do not contain the core 4-phase orchestration logic itself.

Setup Functions:
- prepare_video_transforms: Create video transformation pipeline (resize, pad, normalize)
- setup_video_transform: Initialize transforms and compute target dimensions
- setup_generation_context: Initialize context with device configuration
- prepare_runner: Configure VideoDiffusionInfer with all settings

Video Processing Utilities:
- prepend_video_frames: Add frames at start to reduce initial artifacts
- blend_overlapping_frames: Smooth blending for temporal overlap between batches
- cut_videos: Format videos to meet 4n+1 frame constraint

Configuration Helpers:
- load_text_embeddings: Load positive/negative text embeddings for DiT
- calculate_optimal_batch_params: Compute batch parameters and padding waste
- check_interrupt: Check for user interruption

Debugging Utilities:
- _draw_tile_boundaries: Draw tile boundaries for debugging VAE tiling
- ensure_precision_initialized: Log model dtype information

These utilities support the 4-phase pipeline implemented in generation_phases.py.
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from torchvision.transforms import Compose, Lambda, Normalize

from .model_configuration import configure_runner
from .infer import VideoDiffusionInfer
from ..data.image.transforms.divisible_crop import DivisiblePad
from ..data.image.transforms.na_resize import NaResize
from ..optimization.memory_manager import manage_tensor
from ..utils.constants import get_script_directory

# Get script directory for embeddings
script_directory = get_script_directory()


def prepare_video_transforms(resolution: int, max_resolution: int = 0, debug: Optional['Debug'] = None) -> Compose:
    """
    Prepare optimized video transformation pipeline
    
    Args:
        resolution (int): Target resolution for shortest edge
        max_resolution (int): Maximum resolution for any edge (0 = no limit)
        debug (Debug, optional): Debug instance for logging
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Optional max resolution constraint on longest edge
        - Padding to divisible by 16 (no data loss)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    if debug:
        msg = f"Initializing video transformation pipeline for {resolution}px (shortest edge)"
        if max_resolution > 0:
            msg += f", max {max_resolution}px (any edge)"
        debug.log(msg, category="setup", indent_level=1)
    
    return Compose([
        NaResize(
            resolution=resolution,
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
            max_resolution=max_resolution,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisiblePad((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])


def setup_video_transform(ctx: Dict[str, Any], resolution: int, max_resolution: int = 0, 
                         debug: Optional['Debug'] = None, 
                         sample_frame: Optional[torch.Tensor] = None) -> Tuple[int, int, int, int]:
    """
    Setup video transformation pipeline and compute target dimensions.
    
    Args:
        ctx: Generation context dictionary
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        debug: Debug instance for logging
        sample_frame: Optional sample frame tensor (C, H, W) to compute dimensions
        
    Returns:
        (true_height, true_width, padded_height, padded_width) if dimensions computed,
        (0, 0, 0, 0) otherwise
    """
    # Check if transform exists AND is not None
    existing_transform = ctx.get('video_transform')
    
    if existing_transform is not None:
        # Transform exists - check if we need to compute dimensions
        if 'true_target_dims' in ctx and sample_frame is not None:
            # Return cached dimensions + recompute padded from sample
            true_h, true_w = ctx['true_target_dims']
            transformed = existing_transform(sample_frame)
            padded_h, padded_w = transformed.shape[-2:]
            if debug:
                debug.log("Reusing pre-initialized video transformation pipeline", category="reuse")
            return true_h, true_w, padded_h, padded_w
        elif debug:
            debug.log("Reusing pre-initialized video transformation pipeline", category="reuse")
        return 0, 0, 0, 0
    
    # Create transformation pipeline (first time or after cleanup)
    ctx['video_transform'] = prepare_video_transforms(resolution, max_resolution, debug)
    
    # Compute dimensions if sample frame provided
    if sample_frame is not None:
        # Get true target size (after resize, before padding)
        temp_transform = Compose([
            NaResize(resolution=resolution, mode="side", downsample_only=False, max_resolution=max_resolution),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
        resized_sample = temp_transform(sample_frame)
        true_h, true_w = resized_sample.shape[-2:]
        
        # Round to even numbers for video codec compatibility (libx264 requirement)
        true_h = (true_h // 2) * 2
        true_w = (true_w // 2) * 2
        
        # Cache for later use in trimming
        ctx['true_target_dims'] = (true_h, true_w)
        
        # Get padded dimensions
        transformed_sample = ctx['video_transform'](sample_frame)
        padded_h, padded_w = transformed_sample.shape[-2:]
        
        if debug:
            if true_h == padded_h and true_w == padded_w:
                debug.log(f"Target dimensions: {true_w}x{true_h} (no padding needed)", 
                         category="setup", indent_level=1)
            else:
                debug.log(f"Target dimensions: {true_w}x{true_h} (padded to {padded_w}x{padded_h} for processing)", 
                         category="setup", indent_level=1)
        
        del temp_transform, resized_sample, transformed_sample
        return true_h, true_w, padded_h, padded_w
    
    return 0, 0, 0, 0


def compute_generation_info(
    ctx: Dict[str, Any],
    images: torch.Tensor,
    resolution: int,
    max_resolution: int,
    batch_size: int,
    seed: int,
    prepend_frames: int,
    temporal_overlap: int,
    debug: Optional['Debug'] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute all generation parameters and dimensions for logging.
    
    Args:
        ctx: Generation context dictionary
        images: Input frames tensor [T, H, W, C]
        resolution: Target resolution for shortest edge
        max_resolution: Maximum resolution for any edge (0 = no limit)
        batch_size: Frames per batch
        seed: Random seed
        prepend_frames: Number of frames to prepend
        temporal_overlap: Overlapping frames between batches
        debug: Debug instance for logging
    
    Returns:
        Tuple of (processed_images, info_dict)
        - processed_images: Input images with prepending applied if needed
        - info_dict: Information dictionary for logging
    """
    # Track input frames before any modifications
    input_frames = len(images)
    input_h, input_w = images.shape[1], images.shape[2]
    channels_info = "RGBA" if images.shape[-1] == 4 else "RGB"
    
    # Apply prepending if requested
    if prepend_frames > 0:
        images = prepend_video_frames(images, prepend_frames, debug)
    
    # Track total frames after prepending
    total_frames = len(images)
    ctx['total_frames'] = total_frames
    
    # Setup transform and compute dimensions on final frame count
    sample_frame = images[0].permute(2, 0, 1).unsqueeze(0)
    true_h, true_w, padded_h, padded_w = setup_video_transform(
        ctx, resolution, max_resolution, debug, sample_frame
    )
    del sample_frame
    
    info = {
        'input_frames': input_frames,
        'input_h': input_h,
        'input_w': input_w,
        'total_frames': total_frames,
        'true_h': true_h,
        'true_w': true_w,
        'padded_h': padded_h,
        'padded_w': padded_w,
        'channels_info': channels_info,
        'batch_size': batch_size,
        'seed': seed,
        'prepend_frames': prepend_frames,
        'temporal_overlap': temporal_overlap,
        'resolution': resolution,
        'max_resolution': max_resolution
    }
    
    return images, info


def log_generation_start(info: Dict[str, Any], debug: Optional['Debug'] = None) -> None:
    """
    Log generation start information in a consistent format.
    
    Args:
        info: Information dictionary from compute_generation_info()
        debug: Debug instance for logging
    """
    if debug is None:
        return
    
    debug.log("", category="none", force=True)
    debug.log("Starting upscaling generation...", category="generation", force=True)
    
    # Build concise parameter info
    params_info = f"Batch size: {info['batch_size']}"
    if info['prepend_frames'] > 0:
        params_info += f", Prepend frames: {info['prepend_frames']}"
    if info['temporal_overlap'] > 0:
        params_info += f", Temporal overlap: {info['temporal_overlap']}"
    params_info += f", Seed: {info['seed']}, Channels: {info['channels_info']}"
    
    # Build resolution constraint info
    res_constraint = f"shortest edge: {info['resolution']}px"
    if info['max_resolution'] > 0:
        res_constraint += f", max edge: {info['max_resolution']}px"
    
    # Log dimension flow with full context
    if info['true_h'] > 0:
        frame_text = "frame" if info['input_frames'] <= 1 else "frames"
        if info['true_h'] == info['padded_h'] and info['true_w'] == info['padded_w']:
            debug.log(
                f"Input: {info['input_frames']} {frame_text}, "
                f"{info['input_w']}x{info['input_h']}px → Output: {info['true_w']}x{info['true_h']}px "
                f"({res_constraint})",
                category="generation", force=True, indent_level=1
            )
        else:
            debug.log(
                f"Input: {info['input_frames']} {frame_text}, "
                f"{info['input_w']}x{info['input_h']}px → Padded: {info['padded_w']}x{info['padded_h']}px → "
                f"Output: {info['true_w']}x{info['true_h']}px ({res_constraint})",
                category="generation", force=True, indent_level=1
            )
    
    debug.log(f"{params_info}", category="generation", force=True, indent_level=1)


def prepend_video_frames(frames: torch.Tensor, prepend_count: int, debug: Optional['Debug'] = None) -> torch.Tensor:
    """
    Prepend reversed frames to the video to help prevent artifacts at the start of the video.

    Args:
        frames: Input video tensor [T, H, W, C]
        prepend_count: Number of frames to prepend
        debug: Optional debug instance for logging
        
    Returns:
        torch.Tensor: Video with prepended frames [T+prepend_count, H, W, C]
    """
    if prepend_count <= 0:
        return frames
    
    if len(frames) == 0:
        raise ValueError("Cannot prepend frames to empty video tensor")
    
    if debug:
        debug.log(f"Prepending {prepend_count} reversed frames to help prevent artifacts at the start of the video", category="video", indent_level=1)
    
    # If prepend_count >= total frames, repeat last frame to fill gap
    if prepend_count >= len(frames):
        repeat_count = prepend_count - len(frames) + 1
        repeated_frames = frames[-1:].repeat(repeat_count, 1, 1, 1)
        # Reverse all frames except first (PyTorch-compatible)
        reversed_frames = frames[1:].flip(0)
        return torch.cat([repeated_frames, reversed_frames, frames], dim=0)
    
    # Normal case: reverse frames from index 1 to prepend_count (inclusive)
    # frames[1:prepend_count+1] selects the range, then .flip(0) reverses it
    reversed_frames = frames[1:prepend_count+1].flip(0)
    return torch.cat([reversed_frames, frames], dim=0)


def blend_overlapping_frames(prev_tail: torch.Tensor, cur_head: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Blend two overlapping frame sequences in-place.
    
    Args:
        prev_tail: Last `overlap` frames from previous batch [overlap, H, W, C]
        cur_head: First `overlap` frames from current batch [overlap, H, W, C]
        overlap: Number of overlapping frames
        
    Returns:
        torch.Tensor: Blended frames [overlap, H, W, C]
    """
    device = prev_tail.device
    dtype = prev_tail.dtype
    
    # Smooth crossfade with Hann window for overlap >= 3, linear for smaller overlaps
    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        w_prev_1d = 0.5 + 0.5 * torch.cos(torch.pi * u)  # Hann window
    else:
        w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)
    
    w_prev = w_prev_1d.view(overlap, 1, 1, 1)
    w_cur = 1.0 - w_prev
    
    return prev_tail * w_prev + cur_head * w_cur


def setup_generation_context(
    dit_device: Optional[Union[str, torch.device]] = None,
    vae_device: Optional[Union[str, torch.device]] = None,
    dit_offload_device: Optional[Union[str, torch.device]] = None,
    vae_offload_device: Optional[Union[str, torch.device]] = None,
    tensor_offload_device: Optional[Union[str, torch.device]] = None,
    debug: Optional['Debug'] = None
) -> Dict[str, Any]:
    """
    Initialize generation context with device configuration.
    
    Processes device objects, configures environment variables, and creates the
    generation context dictionary with all necessary state.
    
    Args:
        dit_device: Device for DiT model (str or torch.device, defaults to 'cpu')
        vae_device: Device for VAE model (str or torch.device, defaults to 'cpu')
        dit_offload_device: Device to offload DiT to when not in use (optional)
        vae_offload_device: Device to offload VAE to when not in use (optional)
        tensor_offload_device: Device to offload intermediate tensors to (optional)
        debug: Debug instance for logging
        
    Returns:
        Dict[str, Any]: Generation context dictionary with torch.device objects
    """
    # Normalize devices to torch.device objects (follows PyTorch convention)
    def _normalize_device(device_spec: Optional[Union[str, torch.device]]) -> torch.device:
        """Convert device specification to torch.device object."""
        if device_spec is None:
            return torch.device("cpu")
        if isinstance(device_spec, torch.device):
            return device_spec
        return torch.device(device_spec)
    
    dit_device = _normalize_device(dit_device)
    vae_device = _normalize_device(vae_device)
    dit_offload_device = _normalize_device(dit_offload_device) if dit_offload_device is not None else None
    vae_offload_device = _normalize_device(vae_offload_device) if vae_offload_device is not None else None
    tensor_offload_device = _normalize_device(tensor_offload_device) if tensor_offload_device is not None else None
    
    # Set LOCAL_RANK to 0 for single-GPU inference mode
    # CLI multi-GPU uses CUDA_VISIBLE_DEVICES to restrict visibility per worker
    os.environ.setdefault("LOCAL_RANK", "0")
    
    # Detect ComfyUI integration for interrupt support
    try:
        import comfy.model_management
        interrupt_fn = comfy.model_management.throw_exception_if_processing_interrupted
        comfyui_available = True
    except:
        interrupt_fn = None
        comfyui_available = False
    
    # Create generation context
    ctx = {
        'dit_device': dit_device,
        'vae_device': vae_device,
        'dit_offload_device': dit_offload_device,
        'vae_offload_device': vae_offload_device,
        'tensor_offload_device': tensor_offload_device,
        'compute_dtype': torch.bfloat16, # Hardcoded - gives the best compromise between memory & quality without artifacts
        'interrupt_fn': interrupt_fn,
        'video_transform': None,
        'text_embeds': None,
        'all_transformed_videos': [],
        'all_latents': [],
        'all_upscaled_latents': [],
        'batch_samples': [],
        'final_video': None,
        'comfyui_available': comfyui_available,
        'interrupt_fn': interrupt_fn,
    }
    
    if debug:
        # Build device configuration summary
        offload_info = []
        if dit_offload_device:
            offload_info.append(f"DiT offload={str(dit_offload_device)}")
        if vae_offload_device:
            offload_info.append(f"VAE offload={str(vae_offload_device)}")
        if tensor_offload_device:
            offload_info.append(f"Tensor offload={str(tensor_offload_device)}")
        
        offload_str = ", ".join(offload_info) if offload_info else "none"
        
        debug.log(
            f"Generation context initialized: "
            f"DiT={str(dit_device)}, VAE={str(vae_device)}, "
            f"Offload=[{offload_str}], "
            f"LOCAL_RANK={os.environ['LOCAL_RANK']}",
            category="setup"
        )
        reason = "quality" if ctx['compute_dtype'] == torch.float32 else "compatibility"
        debug.log(f"Unified compute dtype: {ctx['compute_dtype']} across entire pipeline for maximum {reason}", category="precision")
    
    return ctx


def prepare_runner(
    dit_model: str,
    vae_model: str,
    model_dir: str,
    debug: 'Debug',
    ctx: Dict[str, Any],
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
    tile_debug: str = "false",
    attention_mode: str = 'sdpa',
    torch_compile_args_dit: Optional[Dict[str, Any]] = None,
    torch_compile_args_vae: Optional[Dict[str, Any]] = None
) -> Tuple['VideoDiffusionInfer', Dict[str, Any]]:
    """
    Prepare runner with model state management and global cache integration.
    Handles model changes and caching logic with independent DiT/VAE caching support.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        model_dir: Base directory containing model files
        debug: Debug instance for logging (required)
        ctx: Generation context from setup_generation_context
        dit_cache: Whether to cache DiT model between runs
        vae_cache: Whether to cache VAE model between runs
        dit_id: Node instance ID for DiT model caching
        vae_id: Node instance ID for VAE model caching
        block_swap_config: Optional BlockSwap configuration for DiT memory optimization
        encode_tiled: Enable tiled encoding to reduce VRAM during VAE encoding
        encode_tile_size: Tile size for encoding (height, width)
        encode_tile_overlap: Tile overlap for encoding (height, width)
        decode_tiled: Enable tiled decoding to reduce VRAM during VAE decoding
        decode_tile_size: Tile size for decoding (height, width)
        decode_tile_overlap: Tile overlap for decoding (height, width)
        tile_debug: Tile visualization mode (false/encode/decode)
        attention_mode: Attention computation backend ('sdpa' or 'flash_attn')
        torch_compile_args_dit: Optional torch.compile configuration for DiT model
        torch_compile_args_vae: Optional torch.compile configuration for VAE model
        
    Returns:
        Tuple['VideoDiffusionInfer', Dict[str, Any]]: Tuple containing:
            - VideoDiffusionInfer: Configured runner instance with models loaded and settings applied
            - Dict[str, Any]: Cache context dictionary containing cache state and metadata with keys:
                - 'global_cache': GlobalModelCache instance
                - 'dit_cache', 'vae_cache': Caching enabled flags
                - 'dit_id', 'vae_id': Node IDs for cache lookup
                - 'cached_dit', 'cached_vae': Cached model instances (if found)
                - 'reusing_runner': Flag indicating if runner template was reused
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic model reloading when models change
        - Optional torch.compile optimization for inference speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and BlockSwap integration
    """
    dit_changed = False
    vae_changed = False
    
    # Configure runner
    debug.log("Configuring inference runner...", category="runner")
    runner, cache_context = configure_runner(
        dit_model=dit_model,
        vae_model=vae_model,
        base_cache_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=dit_cache,
        vae_cache=vae_cache,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config=block_swap_config,
        encode_tiled=encode_tiled,
        encode_tile_size=encode_tile_size,
        encode_tile_overlap=encode_tile_overlap,
        decode_tiled=decode_tiled,
        decode_tile_size=decode_tile_size,
        decode_tile_overlap=decode_tile_overlap,
        tile_debug=tile_debug,
        attention_mode=attention_mode,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae
    )

    return runner, cache_context


def load_text_embeddings(script_directory: str, device: torch.device, 
                        dtype: torch.dtype, debug: Optional['Debug'] = None) -> Dict[str, List[torch.Tensor]]:
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (torch.device): Target device
        dtype (torch.dtype): Target dtype
        debug: Optional debug instance for logging
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
        - Consistent movement logging
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt'))
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt'))
    
    text_pos_embeds = manage_tensor(
        tensor=text_pos_embeds,
        target_device=device,
        tensor_name="text_pos_embeds",
        dtype=dtype,
        debug=debug,
        reason="DiT inference"
    )
    text_neg_embeds = manage_tensor(
        tensor=text_neg_embeds,
        target_device=device,
        tensor_name="text_neg_embeds",
        dtype=dtype,
        debug=debug,
        reason="DiT inference"
    )
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def calculate_optimal_batch_params(total_frames: int, batch_size: int, 
                                  temporal_overlap: int) -> Dict[str, Any]:
    """
    Calculate optimal batch processing parameters for 4n+1 constraint.
    
    Args:
        total_frames (int): Total number of frames to process
        batch_size (int): Desired batch size
        temporal_overlap (int): Number of overlapping frames between batches
        
    Returns:
        dict: {
            'step': Effective step size between batches,
            'temporal_overlap': Adjusted temporal overlap,
            'best_batch': Optimal batch size for temporal stability,
            'padding_waste': Total frames that will be padded,
            'is_optimal': Whether current batch_size causes no padding
        }
        
    The 4n+1 constraint (1, 5, 9, 13, 17, 21...) is required by the model.
    Best batch prioritizes temporal stability (larger batches) over padding waste.
    """
    # Calculate step size
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Find all valid 4n+1 batch sizes up to total_frames
    valid_sizes = [i for i in range(1, total_frames + 1) if i % 4 == 1]
    
    # Best batch: largest valid size ≤ total_frames (maximizes temporal stability)
    best_batch = max(valid_sizes) if valid_sizes else 1
    
    # Calculate padding waste for current batch_size
    padding_waste = 0
    current_frame = 0
    
    while current_frame < total_frames:
        frames_in_batch = min(batch_size, total_frames - current_frame)
        
        # Find next 4n+1 target
        if frames_in_batch % 4 == 1:
            target = frames_in_batch
        else:
            target = ((frames_in_batch - 1) // 4 + 1) * 4 + 1
        
        padding_waste += target - frames_in_batch
        current_frame += step if step > 0 else batch_size
    
    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': padding_waste == 0
    }


def cut_videos(videos: torch.Tensor) -> torch.Tensor:
    """
    Correct video cutting respecting the constraint: frames % 4 == 1
    
    Args:
        videos (torch.Tensor): Video tensor to format
        
    Returns:
        torch.Tensor: Properly formatted video tensor
        
    Features:
        - Ensures frames % 4 == 1 constraint for model compatibility
        - Intelligent padding with last frame repetition
        - Memory-efficient tensor operations
    """
    t = videos.size(1)
    
    if t % 4 == 1:
        return videos
    
    # Calculate next valid number (4n + 1)
    padding_needed = (4 - (t % 4)) % 4 + 1
    
    # Apply padding to reach 4n+1 format
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    
    return result


def check_interrupt(ctx: Dict[str, Any]) -> None:
    """Single interrupt check to avoid redundant imports"""
    if ctx.get('interrupt_fn') is not None:
        ctx['interrupt_fn']()


def _draw_tile_boundaries(image: torch.Tensor, debug: 'Debug', tile_boundaries: list, phase: str) -> torch.Tensor:
    """
    Draw tile boundary overlays on all frames for debugging (non-destructive).
    
    Args:
        image: Image tensor [T, H, W, C] or [H, W, C] in range [0, 1]
        debug: Debug instance for logging
        tile_boundaries: List of tile boundary info dictionaries
        phase: Phase name ('encode' or 'decode') for logging
        
    Returns:
        Image with boundary overlays drawn inside tiles on all frames
    """
    if not tile_boundaries:
        return image

    # Try to import required libraries
    try:
        import cv2
        import numpy as np
        import random
        import colorsys
    except ImportError as e:
        debug.log(f"Tile debug ignored: missing imports ({e})", level="WARNING", category="video")
        return image

    # Handle both [T, H, W, C] and [H, W, C]
    squeeze_t = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_t = True
    
    T, H, W, C = image.shape
    original_dtype = image.dtype
    
    log_frames = f"all {T} frames" if T > 1 else "1 frame"
    debug.log(f"Drawing {phase} tile boundaries ({len(tile_boundaries)} tiles) on {log_frames}", category="video", indent_level=1, force=True)

    # Scale line thickness and font size based on video width
    # Reference points: 512px (min) to 1920px (max)
    min_width, max_width = 512, 1920
    min_line_thickness, max_line_thickness = 2, 6
    min_font_scale, max_font_scale = 0.8, 2.5
    min_text_thickness, max_text_thickness = 2, 4
    
    # Calculate scale factor (clamped between 0 and 1)
    scale_factor = max(0.0, min(1.0, (W - min_width) / (max_width - min_width)))
    
    # Apply scaling
    line_thickness = int(min_line_thickness + scale_factor * (max_line_thickness - min_line_thickness))
    font_scale = min_font_scale + scale_factor * (max_font_scale - min_font_scale)
    text_thickness = int(min_text_thickness + scale_factor * (max_text_thickness - min_text_thickness))
    
    # Generate high-contrast colors using HSV color space
    num_tiles = len(tile_boundaries)
    colors = []
    for i in range(num_tiles):
        hue = (i * 360 / num_tiles) % 360
        saturation = 0.9 + (i % 2) * 0.1
        brightness = 0.8 + ((i // 2) % 2) * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, brightness)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for OpenCV
    
    random.seed(42)
    random.shuffle(colors)
    
    # Process all frames
    annotated_frames = []
    for frame_idx in range(T):
        # Convert frame to numpy (handle RGB and RGBA)
        img = (image[frame_idx].float().cpu().numpy() * 255).astype(np.uint8)  # [H, W, C]
        
        # Draw boundary lines inside each tile
        for idx, tile_info in enumerate(tile_boundaries):
            tile_id = tile_info['id']
            x, y = tile_info['x'], tile_info['y']
            w, h = tile_info['w'], tile_info['h']
            color = colors[idx]
            
            inset = line_thickness // 2
            
            # Draw four edges
            cv2.line(img, (x, y + inset), (x + w, y + inset), color, line_thickness)
            cv2.line(img, (x, y + h - inset), (x + w, y + h - inset), color, line_thickness)
            cv2.line(img, (x + inset, y), (x + inset, y + h), color, line_thickness)
            cv2.line(img, (x + w - inset, y), (x + w - inset, y + h), color, line_thickness)
            
            # Draw tile number
            text = str(tile_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            margin = int(15 * scale_factor) if scale_factor > 0.5 else 8  # Scale margin too
            text_x = x + margin
            text_y = y + text_h + margin
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, text_thickness, cv2.LINE_AA)
        
        # Convert back to tensor
        frame_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).to(device=image.device, dtype=original_dtype)
        annotated_frames.append(frame_tensor)
    
    # Stack all frames
    image = torch.stack(annotated_frames, dim=0)
    
    if squeeze_t:
        image = image.squeeze(0)
    
    return image


def ensure_precision_initialized(
    ctx: Dict[str, Any],
    runner: 'VideoDiffusionInfer',
    debug: Optional['Debug'] = None
) -> None:
    """
    Log model dtypes for debugging. Compute dtype is hardcoded in context.
    
    Since compute_dtype is hardcoded to bfloat16 in setup_generation_context(),
    this function only logs model dtypes for informational purposes.
    
    Args:
        ctx: Generation context dictionary (compute_dtype already set)
        runner: VideoDiffusionInfer instance with loaded models
        debug: Optional Debug instance for logging
    """
    if not debug:
        return
    
    try:
        # Get model dtypes for informational logging
        dit_dtype = None
        vae_dtype = None
        
        if runner.dit is not None:
            try:
                param_device = next(runner.dit.parameters()).device
                if param_device.type != 'meta':
                    dit_dtype = next(runner.dit.parameters()).dtype
            except StopIteration:
                pass
        
        if runner.vae is not None:
            try:
                param_device = next(runner.vae.parameters()).device
                if param_device.type != 'meta':
                    vae_dtype = next(runner.vae.parameters()).dtype
            except StopIteration:
                pass
        
        # Build precision info string
        parts = []
        if dit_dtype is not None:
            parts.append(f"DiT={dit_dtype}")
        if vae_dtype is not None:
            parts.append(f"VAE={vae_dtype}")
        parts.append(f"compute={ctx['compute_dtype']}")
        
        if parts:
            debug.log(f"Model precision: {', '.join(parts)}", category="precision")
            
    except Exception as e:
        debug.log(f"Could not log model dtypes: {e}", level="WARNING", category="precision", force=True)
