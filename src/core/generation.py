"""
Generation Logic Module for SeedVR2

This module implements a four-phase batch processing pipeline for video upscaling:
- Phase 1: Batch VAE encoding of all input frames
- Phase 2: Batch DiT upscaling of all encoded latents
- Phase 3: Batch VAE decoding of all upscaled latents
- Phase 4: Post-processing and final video assembly

This architecture minimizes model swapping overhead by completing each phase
for all batches before moving to the next phase, significantly improving
performance especially when using model offloading.

Key Features:
- Four-phase pipeline (encode-all → upscale-all → decode-all → postprocess-all) for efficiency
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Temporal overlap support for smooth transitions between batches
- Adaptive dtype detection and configuration
- Memory-efficient pre-allocated batch processing
- Stream-based assembly eliminates memory spikes for long videos
- Advanced video format handling (4n+1 constraint)
- Clean separation of concerns with phase-specific resource management
- Each phase handles its own cleanup in finally blocks
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from torchvision.transforms import Compose, Lambda, Normalize

from .alpha_upscaling import process_alpha_for_batch
from .infer import VideoDiffusionInfer
from .model_manager import configure_runner, materialize_model, apply_model_specific_config
from ..common.seed import set_seed
from ..common.half_precision_fixes import ensure_float32_precision
from ..data.image.transforms.divisible_crop import DivisibleCrop
from ..data.image.transforms.na_resize import NaResize
from ..optimization.memory_manager import (
    cleanup_dit,
    cleanup_vae,
    cleanup_text_embeddings,
    clear_memory,
    manage_tensor,
    manage_model_device,
    release_tensor_memory,
    release_tensor_collection
)
from ..optimization.performance import (
    optimized_video_rearrange, 
    optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from ..utils.color_fix import (
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    hsv_saturation_histogram_match, 
    wavelet_reconstruction,
    adaptive_instance_normalization
)
from ..utils.constants import get_script_directory

# Get script directory for embeddings
script_directory = get_script_directory()


def prepare_video_transforms(res_w: int, debug: Optional['Debug'] = None) -> Compose:
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        debug (Debug, optional): Debug instance for logging
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    if debug:
        debug.log(f"Initializing video transformation pipeline for {res_w}px", category="setup")
    
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])


def setup_video_transform(ctx: Dict[str, Any], res_w: int, debug: Optional['Debug'] = None) -> None:
    """
    Setup video transformation pipeline in context with consistent timing and logging.
    
    Args:
        ctx: Generation context dictionary
        res_w: Target resolution width
        debug: Debug instance
    """
    if ctx.get('video_transform') is None:
        debug.start_timer("video_transform")
        ctx['video_transform'] = prepare_video_transforms(res_w, debug)
        debug.end_timer("video_transform", "Video transform pipeline initialization")
    else:
        debug.log("Reusing pre-initialized video transformation pipeline", category="reuse")


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


def check_interrupt(ctx: Dict[str, Any]) -> None:
    """Single interrupt check to avoid redundant imports"""
    if ctx.get('interrupt_fn') is not None:
        ctx['interrupt_fn']()


def _ensure_precision_initialized(
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


def setup_generation_context(
    dit_device: Optional[torch.device] = None,
    vae_device: Optional[torch.device] = None,
    dit_offload_device: Optional[torch.device] = None,
    vae_offload_device: Optional[torch.device] = None,
    tensor_offload_device: Optional[torch.device] = None,
    debug: Optional['Debug'] = None
) -> Dict[str, Any]:
    """
    Initialize generation context with device configuration.
    
    Processes device objects, configures environment variables, and creates the
    generation context dictionary with all necessary state.
    
    Args:
        dit_device: Device for DiT model (torch.device object, defaults to cpu)
        vae_device: Device for VAE model (torch.device object, defaults to cpu)
        dit_offload_device: Device to offload DiT to when not in use (optional torch.device)
        vae_offload_device: Device to offload VAE to when not in use (optional torch.device)
        tensor_offload_device: Device to offload intermediate tensors to (optional torch.device)
        debug: Debug instance for logging
        
    Returns:
        Dict[str, Any]: Generation context dictionary containing:
            - dit_device: torch.device for DiT inference
            - vae_device: torch.device for VAE inference
            - offload device configurations
            - state containers (latents, samples, etc.)
            - ComfyUI integration hooks
    """
    # Apply default devices if not specified
    default_device = "cpu"
    if dit_device is None:
        dit_device = default_device
    if vae_device is None:
        vae_device = default_device
    
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
        reason = "quality" if ctx["compute_dtype"] == torch.float32 else "compatibility"
        debug.log(f"Unified compute dtype: {ctx["compute_dtype"]} across entire pipeline for maximum {reason}", category="precision")
    
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
) -> Tuple['VideoDiffusionInfer', bool, Dict[str, Any]]:
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
        Tuple['VideoDiffusionInfer', bool, Dict[str, Any]]: Tuple containing:
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


def encode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    images: torch.Tensor,
    debug: 'Debug',
    batch_size: int = 5,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    temporal_overlap: int = 0,
    res_w: int = 1072,
    input_noise_scale: float = 0.0,
    color_correction: str = "wavelet"
) -> Dict[str, Any]:
    """
    Phase 1: VAE Encoding for all batches
    
    Encodes video frames to latents in batches, handling temporal overlap and 
    memory optimization. Creates context automatically if not provided.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Generation context from setup_generation_context (required)
        images: Input frames tensor [T, H, W, C] range [0,1] (required) 
        debug: Debug instance for logging (required)
        batch_size: Frames per batch (4n+1 format: 1, 5, 9, 13...)
        seed: Random seed for deterministic VAE sampling (default: 42)
        progress_callback: Optional callback(current, total, frames, phase_name)
        temporal_overlap: Overlapping frames between batches for continuity
        res_w: Target resolution for shortest edge
        input_noise_scale: Scale for input noise (0.0-1.0). Adds noise to input images
                          before VAE encoding to reduce artifacts at high resolutions.
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
                         Determines if transformed videos need to be stored for later use.
        
    Returns:
        dict: Context containing:
            - all_transformed_videos: List of (video, original_length) tuples
            - all_latents: List of encoded latents ready for upscaling
            - Other state for subsequent phases
            
    Raises:
        ValueError: If required inputs are missing or invalid
        RuntimeError: If encoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to encode_all_batches")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 1: VAE encoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase1_encoding")

    # Context must be provided
    if ctx is None:
        raise ValueError("Generation context must be provided to encode_all_batches")
    
    # Validate and store inputs
    if images is None:
        raise ValueError("Images to encode must be provided")
    else:
        ctx['input_images'] = images
    
    # Get total frame count from context (set in video_upscaler before encoding)
    total_frames = ctx.get('total_frames', len(images))
    
    # Set it if not already set (for standalone/CLI usage)
    if 'total_frames' not in ctx:
        ctx['total_frames'] = total_frames
    
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # Setup video transformation pipeline if not already done
    setup_video_transform(ctx, res_w, debug)
    
    # Detect if input is RGBA (4 channels)
    ctx['is_rgba'] = images[0].shape[-1] == 4
    
    # Display batch optimization tips
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['padding_waste'] > 0:
            debug.log("", category="none", force=True)
            debug.log(f"Padding waste: {batch_params['padding_waste']}", category="info", force=True)
            debug.log(f"Why padding? Each batch must be 4n+1 frames (1, 5, 9, 13, 17, 21, ...)", category="info", force=True, indent_level=1)
            debug.log(f"Current batch_size creates partial batches that need padding to meet this constraint", category="info", force=True, indent_level=1)
            debug.log(f"This increases memory usage and processing time unnecessarily", category="info", force=True, indent_level=1)
        
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"For {total_frames} frames, batch_size={batch_params['best_batch']} avoids padding waste", category="tip", force=True)
            debug.log(f"Match batch_size to shot length for better temporal coherence", category="tip", force=True, indent_level=1)

        
        if batch_params['padding_waste'] > 0 or (batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames):
            debug.log("", category="none", force=True)
    
    # Calculate batching parameters
    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Calculate number of batches
    num_encode_batches = 0
    for idx in range(0, total_frames, step):
        end_idx = min(idx + batch_size, total_frames)
        if idx > 0 and end_idx - idx <= temporal_overlap:
            break
        num_encode_batches += 1
    
    # Pre-allocate lists for memory efficiency
    ctx['all_latents'] = [None] * num_encode_batches
    ctx['all_ori_lengths'] = [None] * num_encode_batches
    if color_correction != "none":
        ctx['all_transformed_videos'] = [None] * num_encode_batches
    else:
        ctx['all_transformed_videos'] = None
    
    encode_idx = 0
    
    try:
        # Materialize VAE if still on meta device
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_vae_config_needs_application', False):
                debug.log("Applying updated VAE configuration", category="vae", force=True)
                apply_model_specific_config(runner.vae, runner, runner.config, False, debug)
        
        # Initialize precision after VAE is materialized with actual weights
        _ensure_precision_initialized(ctx, runner, debug)

        # Cache VAE now that it's fully configured and ready for inference
        if ctx['cache_context']['vae_cache'] and not ctx['cache_context']['cached_vae']:
            runner.vae._model_name = ctx['cache_context']['vae_model']
            ctx['cache_context']['global_cache'].set_vae(
                {'node_id': ctx['cache_context']['vae_id'], 'cache_model': True}, 
                runner.vae, ctx['cache_context']['vae_model'], debug
            )
            ctx['cache_context']['vae_newly_cached'] = True
        
        # Set deterministic seed for VAE encoding (separate from diffusion noise)
        # Uses seed + 1,000,000 to avoid collision with upscaling batch seeds
        # This ensures VAE sampling is deterministic while maintaining quality
        seed_vae = seed + 1000000
        set_seed(seed_vae)
        debug.log(f"Using seed: {seed_vae} (VAE uses seed+1000000 for deterministic sampling)", category="vae")
        
        # Move VAE to GPU for encoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for encoding", detailed_tensors=False)

        # Initialize tile_boundaries for encoding debug
        if runner.tile_debug == "encode" and runner.encode_tiled:
            debug.encode_tile_boundaries = []
            debug.log("Tile debug enabled: collecting encode tile boundaries for visualization", category="vae")
        
        # Process encoding
        for batch_idx in range(0, total_frames, step):
            check_interrupt(ctx)
            
            # Calculate indices with temporal overlap
            if batch_idx == 0:
                start_idx = 0
                end_idx = min(batch_size, total_frames)
            else:
                start_idx = batch_idx
                end_idx = min(start_idx + batch_size, total_frames)
                if end_idx - start_idx <= temporal_overlap:
                    break
            
            current_frames = end_idx - start_idx
            
            debug.log(f"Encoding batch {encode_idx+1}/{num_encode_batches}", category="vae", force=True)
            debug.start_timer(f"encode_batch_{encode_idx+1}")
            
            # Process current batch
            video = images[start_idx:end_idx]
            video = video.permute(0, 3, 1, 2)
            video = manage_tensor(
                tensor=video,
                target_device=ctx['vae_device'],
                tensor_name=f"video_batch_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )

            # Check temporal dimension and pad ONCE if needed (format: T, C, H, W)
            t = video.size(0)
            debug.log(f"Sequence of {t} frames", category="video", force=True, indent_level=1)

            ori_length = t

            if t % 4 != 1:
                target = ((t-1)//4+1)*4+1
                padding_frames = target - t
                debug.log(f"Applying padding: {padding_frames} frame{'s' if padding_frames != 1 else ''} added ({t} -> {target})", category="video", force=True, indent_level=1)
                
                # Pad original video once (TCHW format, need to convert to CTHW)
                video = optimized_single_video_rearrange(video)  # TCHW -> CTHW
                video = cut_videos(video)
                video = optimized_single_video_rearrange(video)  # CTHW -> TCHW

            # Extract RGB for transforms (view, not copy)
            if ctx.get('is_rgba', False):
                rgb_for_transform = video[:, :3, :, :]
                debug.log(f"Extracted Alpha channel for edge-guided upscaling", category="alpha", indent_level=1)
            else:
                rgb_for_transform = video

            # Apply transformations (to RGB from already-padded video)
            transformed_video = ctx['video_transform'](rgb_for_transform)

            del rgb_for_transform

            # Apply input noise if requested (to reduce artifacts at high resolutions)
            if input_noise_scale > 0:
                debug.log(f"Applying input noise (scale: {input_noise_scale:.2f})", category="video", indent_level=1)
                
                # Generate noise matching the video shape
                noise = torch.randn_like(transformed_video)
                
                # Subtle noise amplitude
                noise = noise * 0.05
                
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                
                # Apply blend
                transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                
                del noise

            # Store original length for proper trimming later
            ctx['all_ori_lengths'][encode_idx] = ori_length

            # Extract and store Alpha and RGB from padded original video (before encoding)
            if ctx.get('is_rgba', False):
                if 'all_alpha_channels' not in ctx:
                    ctx['all_alpha_channels'] = [None] * num_encode_batches
                if 'all_input_rgb' not in ctx:
                    ctx['all_input_rgb'] = [None] * num_encode_batches
                
                # Extract from padded RGBA video (format: T, 4, H, W)
                alpha_channel = video[:, 3:4, :, :]
                rgb_video_original = video[:, :3, :, :]
                
                # Store on tensor_offload_device to save VRAM (or keep on device if none)
                if ctx['tensor_offload_device'] is not None:
                    ctx['all_alpha_channels'][encode_idx] = manage_tensor(
                        tensor=alpha_channel,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"alpha_channel_{encode_idx+1}",
                        debug=debug,
                        reason="storing Alpha channel for upscaling",
                        indent_level=1
                    )
                    ctx['all_input_rgb'][encode_idx] = manage_tensor(
                        tensor=rgb_video_original,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"rgb_original_{encode_idx+1}",
                        debug=debug,
                        reason="storing RGB edge guidance for Alpha upscaling",
                        indent_level=1
                    )
                else:
                    ctx['all_alpha_channels'][encode_idx] = alpha_channel
                    ctx['all_input_rgb'][encode_idx] = rgb_video_original
                
                del alpha_channel, rgb_video_original

            del video

            # Move to VAE device with correct dtype for encoding
            transformed_video = manage_tensor(
                tensor=transformed_video,
                target_device=ctx['vae_device'],
                tensor_name=f"transformed_video_{encode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE encoding",
                indent_level=1
            )
            
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])
            
            # Store transformed video for color correction after encoding
            if color_correction != "none":
                if ctx['tensor_offload_device'] is not None:
                    # Move to offload device to free VRAM
                    ctx['all_transformed_videos'][encode_idx] = manage_tensor(
                        tensor=transformed_video,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"transformed_video_{encode_idx+1}",
                        debug=debug,
                        reason="storing input reference for color correction",
                        indent_level=1
                    )
                else:
                    # No offload device - keep reference on VAE device
                    ctx['all_transformed_videos'][encode_idx] = transformed_video
            
            # Clean up transformed_video reference if not needed or already offloaded
            if color_correction == "none" or ctx['tensor_offload_device'] is not None:
                del transformed_video
            
            # Convert from VAE dtype to compute dtype and offload to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (cond_latents[0].is_cuda or cond_latents[0].is_mps):
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="storing encoded latents for upscaling",
                    indent_level=1
                )
            else:
                # Stay on current device but convert to compute dtype
                ctx['all_latents'][encode_idx] = manage_tensor(
                    tensor=cond_latents[0],
                    target_device=cond_latents[0].device,
                    tensor_name=f"latent_{encode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="VAE dtype → compute dtype",
                    indent_level=1
                )
            
            del cond_latents
            
            debug.end_timer(f"encode_batch_{encode_idx+1}", f"Encoded batch {encode_idx+1}")
            
            if progress_callback:
                progress_callback(encode_idx+1, num_encode_batches, 
                                current_frames, "Phase 1: Encoding")
            
            encode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 1 (Encoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Offload VAE to configured offload device if specified
        if ctx['vae_offload_device'] is not None:
            manage_model_device(model=runner.vae, target_device=ctx['vae_offload_device'], 
                                model_name="VAE", debug=debug, reason="VAE offload", runner=runner)
    
    debug.end_timer("phase1_encoding", "Phase 1: VAE encoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 1 (VAE encoding)", show_tensors=False)
    
    return ctx


def upscale_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cfg_scale: float = 1,
    seed: int = 42,
    latent_noise_scale: float = 0.0,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 2: DiT Upscaling for all encoded batches.
    
    Processes all encoded latents through the diffusion model for upscaling.
    Requires context from encode_all_batches with encoded latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from encode_all_batches containing latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cfg_scale: Classifier-free guidance scale (default: 1.0)
        seed: Random seed for reproducible generation
        latent_noise_scale: Noise scale for latent space augmentation (0.0-1.0).
                           Adds noise during diffusion conditioning. Can soften details
                           but may help with certain artifacts. 0.0 = no noise (crisp),
                           1.0 = maximum noise (softer)
        cache_model: If True, keep DiT model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - all_upscaled_latents: List of upscaled latents ready for decoding
            - Preserved state from encoding phase
            
    Raises:
        ValueError: If context is missing or has no encoded latents
        RuntimeError: If upscaling fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to upscale_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for upscale_all_batches. Run encode_all_batches first.")
        
    # Validate we have encoded latents
    if 'all_latents' not in ctx or not ctx['all_latents']:
        raise ValueError("No encoded latents found. Run encode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscaling")
    
    # Load text embeddings if not already loaded
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)
        debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Configure diffusion parameters
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx['dit_device'], dtype=ctx['compute_dtype'])

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_latents'] if l is not None])

    # Safety check for empty latents
    if num_valid_latents == 0:
        debug.log("No valid latents to upscale", level="WARNING", category="dit", force=True)
        ctx['all_upscaled_latents'] = []
        return ctx
    
    # Pre-allocate list for upscaled latents
    ctx['all_upscaled_latents'] = [None] * num_valid_latents
    
    upscale_idx = 0
    
    try:
        # Materialize DiT if still on meta device
        if runner.dit and next(runner.dit.parameters()).device.type == 'meta':
            materialize_model(runner, "dit", ctx['dit_device'], runner.config, debug)
        else:
            # Model already materialized (cached) - apply any pending configs if needed
            if getattr(runner, '_dit_config_needs_application', False):
                debug.log("Applying updated DiT configuration", category="dit", force=True)
                apply_model_specific_config(runner.dit, runner, runner.config, True, debug)
    
        # Initialize precision after DiT is materialized with actual weights
        _ensure_precision_initialized(ctx, runner, debug)

        # Cache DiT now that it's fully configured and ready for inference
        if ctx['cache_context']['dit_cache'] and not ctx['cache_context']['cached_dit']:
            runner.dit._model_name = ctx['cache_context']['dit_model']
            ctx['cache_context']['global_cache'].set_dit(
                {'node_id': ctx['cache_context']['dit_id'], 'cache_model': True}, 
                runner.dit, ctx['cache_context']['dit_model'], debug
            )
            ctx['cache_context']['dit_newly_cached'] = True
            # If both models now cached, cache runner template
            vae_is_cached = ctx['cache_context']['cached_vae'] or ctx['cache_context']['vae_newly_cached']
            if vae_is_cached:
                ctx['cache_context']['global_cache'].set_runner(
                    ctx['cache_context']['dit_id'], ctx['cache_context']['vae_id'], 
                    runner, debug
                )
        
        # Set base seed for DiT noise generation
        # Ensures deterministic noise across all batches in this upscaling phase
        set_seed(seed)
        debug.log(f"Using seed: {seed}, CFG scale: {cfg_scale}", category="dit")
        
        # Move DiT to GPU for upscaling (no-op if already there)
        manage_model_device(model=runner.dit, target_device=ctx['dit_device'], 
                            model_name="DiT", debug=debug, runner=runner)

        debug.log_memory_state("After DiT loading for upscaling", detailed_tensors=False)

        for batch_idx, latent in enumerate(ctx['all_latents']):
            if latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Upscaling batch {upscale_idx+1}/{num_valid_latents}", category="generation", force=True)
            debug.start_timer(f"upscale_batch_{upscale_idx+1}")
            
            # Move to DiT device with correct dtype for upscaling (no-op if already there)
            latent = manage_tensor(
                tensor=latent,
                target_device=ctx['dit_device'],
                tensor_name=f"latent_{upscale_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="DiT upscaling",
                indent_level=1
            )

            # Generate noise (randn_like automatically uses latent's device)
            base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            
            noises = [base_noise]
            aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
            
            # Log latent noise application if enabled
            if latent_noise_scale > 0:
                debug.log(f"Applying latent noise (scale: {latent_noise_scale:.3f})", category="generation")
            
            def _add_noise(x, aug_noise):
                if latent_noise_scale == 0.0:
                    return x
                t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
                shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
                t = runner.timestep_transform(t, shape)
                x = runner.schedule.forward(x, aug_noise, t)
                del t, shape
                return x
            
            # Generate condition
            condition = runner.get_condition(
                noises[0],
                task="sr",
                latent_blur=_add_noise(latent, aug_noises[0]),
            )
            conditions = [condition]
            
            # Detect DiT model dtype (handle FP8CompatibleDiT wrapper)
            dit_model = runner.dit.dit_model if hasattr(runner.dit, 'dit_model') else runner.dit
            try:
                dit_dtype = next(dit_model.parameters()).dtype
            except StopIteration:
                dit_dtype = ctx['compute_dtype']  # Fallback for meta device or empty model
            
            # Use autocast if DiT dtype differs from compute dtype
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                if dit_dtype != ctx['compute_dtype']:
                    with torch.autocast(str(ctx['dit_device']), ctx['compute_dtype'], enabled=True):
                        upscaled_latents = runner.inference(
                            noises=noises,
                            conditions=conditions,
                            **ctx['text_embeds'],
                        )
                else:
                    upscaled_latents = runner.inference(
                        noises=noises,
                        conditions=conditions,
                        **ctx['text_embeds'],
                    )
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")
            
            # Offload upscaled latents to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None and (upscaled_latents[0].is_cuda or upscaled_latents[0].is_mps):
                ctx['all_upscaled_latents'][upscale_idx] = manage_tensor(
                    tensor=upscaled_latents[0],
                    target_device=ctx['tensor_offload_device'],
                    tensor_name=f"upscaled_latent_{upscale_idx+1}",
                    debug=debug,
                    reason="storing upscaled latents for decoding",
                    indent_level=1
                )
            else:
                ctx['all_upscaled_latents'][upscale_idx] = upscaled_latents[0]
            
            # Free original latent - release tensor memory first
            release_tensor_memory(ctx['all_latents'][batch_idx])
            ctx['all_latents'][batch_idx] = None
            
            del noises, aug_noises, latent, conditions, condition, base_noise, upscaled_latents
            
            debug.end_timer(f"upscale_batch_{upscale_idx+1}", f"Upscaled batch {upscale_idx+1}")
            
            if progress_callback:
                progress_callback(upscale_idx+1, num_valid_latents,
                                1, "Phase 2: Upscaling")
            
            upscale_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 2 (Upscaling): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Log BlockSwap summary if it was used
        if hasattr(runner, '_blockswap_active') and runner._blockswap_active:
            swap_summary = debug.get_swap_summary()
            if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                debug.log("BlockSwap Summary", category="blockswap")
                debug.log(f"BlockSwap overhead: {total_time:.2f}ms", category="blockswap", indent_level=1)
                debug.log(f"Total swaps: {swap_summary['total_swaps']}", category="blockswap", indent_level=1)
                
                # Show block swap details
                if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                    avg_ms = swap_summary.get('block_avg_ms', 0)
                    total_ms = swap_summary.get('block_total_ms', 0)
                    min_ms = swap_summary.get('block_min_ms', 0)
                    max_ms = swap_summary.get('block_max_ms', 0)
                    
                    debug.log(f"Block swaps: {swap_summary['block_swaps']} "
                            f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                            category="blockswap", indent_level=1)
                    
                    # Show most frequently swapped block
                    if 'most_swapped_block' in swap_summary:
                        debug.log(f"Most swapped: Block {swap_summary['most_swapped_block']} "
                                f"({swap_summary['most_swapped_count']} times)", category="blockswap", indent_level=1)
                
                # Show I/O swap details if present
                if 'io_swaps' in swap_summary and swap_summary['io_swaps'] > 0:
                    debug.log(f"I/O swaps: {swap_summary['io_swaps']} "
                            f"(avg: {swap_summary.get('io_avg_ms', 0):.2f}ms, total: {swap_summary.get('io_total_ms', 0):.2f}ms)", 
                            category="blockswap", indent_level=1)

        # Cleanup DiT as it's no longer needed after upscaling
        cleanup_dit(runner=runner, debug=debug, cache_model=cache_model)
        
        # Cleanup text embeddings as they're no longer needed after upscaling
        cleanup_text_embeddings(ctx, debug)
        
        clear_memory(debug=debug, deep=True, force=True, timer_name="upscale_all_batches_finally")
    
    debug.end_timer("phase2_upscaling", "Phase 2: DiT upscaling complete", show_breakdown=True)
    debug.log_memory_state("After phase 2 (DiT upscaling)", show_tensors=False)
    
    return ctx


def decode_all_batches(
    runner: 'VideoDiffusionInfer',
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    cache_model: bool = False
) -> Dict[str, Any]:
    """
    Phase 3: VAE Decoding.
    
    Decodes all upscaled latents back to pixel space.
    Requires context from upscale_all_batches with upscaled latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from upscale_all_batches containing upscaled latents (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cache_model: If True, keep VAE model for reuse instead of deleting it
        
    Returns:
        dict: Updated context containing:
            - batch_samples: List of decoded samples ready for post-processing
            - VAE cleanup completed
            
    Raises:
        ValueError: If context is missing or has no upscaled latents
        RuntimeError: If decoding fails
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to decode_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for decode_all_batches. Run upscale_all_batches first.")
    
    # Validate we have upscaled latents
    if 'all_upscaled_latents' not in ctx or not ctx['all_upscaled_latents']:
        raise ValueError("No upscaled latents found. Run upscale_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 3: VAE decoding ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase3_decoding")

    # Count valid latents
    num_valid_latents = len([l for l in ctx['all_upscaled_latents'] if l is not None])
    
    # Pre-allocate to match original batches (use ori_lengths which is always available)
    num_batches = len([l for l in ctx['all_ori_lengths'] if l is not None])
    ctx['batch_samples'] = [None] * num_batches
    
    decode_idx = 0
    
    try:
        # VAE should already be materialized from encoding phase
        if runner.vae and next(runner.vae.parameters()).device.type == 'meta':
            materialize_model(runner, "vae", ctx['vae_device'], runner.config, debug)

        # Precision should already be initialized from encoding phase
        _ensure_precision_initialized(ctx, runner, debug)

        # Move VAE to GPU for decoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=ctx['vae_device'], 
                          model_name="VAE", debug=debug, runner=runner)
        
        debug.log_memory_state("After VAE loading for decoding", detailed_tensors=False)

        # Initialize tile_boundaries for decoding debug
        if runner.tile_debug == "decode" and runner.decode_tiled:
            debug.decode_tile_boundaries = []
            debug.log("Tile debug enabled: collecting decode tile boundaries for visualization", category="vae")
        
        # Process decoding
        for batch_idx, upscaled_latent in enumerate(ctx['all_upscaled_latents']):
            if upscaled_latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Decoding batch {decode_idx+1}/{num_valid_latents}", category="vae", force=True)
            debug.start_timer(f"decode_batch_{decode_idx+1}")
            
            # Move to VAE device with correct dtype for decoding (no-op if already there)
            upscaled_latent = manage_tensor(
                tensor=upscaled_latent,
                target_device=ctx['vae_device'],
                tensor_name=f"upscaled_latent_{decode_idx+1}",
                dtype=ctx['compute_dtype'],
                debug=debug,
                reason="VAE decoding",
                indent_level=1
            )
            
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent])
            debug.end_timer("vae_decode", "VAE decode")
            
            # Process samples
            debug.start_timer("optimized_video_rearrange")
            samples = optimized_video_rearrange(samples)
            debug.end_timer("optimized_video_rearrange", "Video rearrange")
            
            # Convert from VAE dtype to compute dtype and offload to avoid VRAM accumulation
            if ctx['tensor_offload_device'] is not None:
                # samples is always a single-element list from vae_decode([upscaled_latent])]
                if samples[0].is_cuda or samples[0].is_mps:
                    samples[0] = manage_tensor(
                        tensor=samples[0],
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"sample_{decode_idx+1}",
                        dtype=ctx['compute_dtype'],
                        debug=debug,
                        reason="storing decoded samples for post-processing",
                        indent_level=1
                    )
            else:
                # No offload device, but still convert to compute dtype
                samples[0] = manage_tensor(
                    tensor=samples[0],
                    target_device=samples[0].device,
                    tensor_name=f"sample_{decode_idx+1}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="VAE dtype → compute dtype",
                    indent_level=1
                )
            ctx['batch_samples'][decode_idx] = samples
            
            # Free the upscaled latent and GPU samples
            release_tensor_memory(ctx['all_upscaled_latents'][batch_idx])
            ctx['all_upscaled_latents'][batch_idx] = None
            del upscaled_latent, samples
            
            debug.end_timer(f"decode_batch_{decode_idx+1}", f"Decoded batch {decode_idx+1}")
            
            if progress_callback:
                progress_callback(decode_idx+1, num_valid_latents,
                                1, "Phase 3: Decoding")
            
            decode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 3 (Decoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Cleanup VAE as it's no longer needed
        cleanup_vae(runner=runner, debug=debug, cache_model=cache_model)
        
        # Clean up upscaled latents storage
        if 'all_upscaled_latents' in ctx:
            release_tensor_collection(ctx['all_upscaled_latents'])
            del ctx['all_upscaled_latents']
        
    debug.end_timer("phase3_decoding", "Phase 3: VAE decoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 3 (VAE decoding)", show_tensors=False)
    
    return ctx


def postprocess_all_batches(
    ctx: Dict[str, Any],
    debug: 'Debug',
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
    color_correction: str = "wavelet"
) -> Dict[str, Any]:
    """
    Phase 4: Post-processing and Final Video Assembly.
    
    Applies color correction and assembles the final video from decoded batches.
    Uses stream-based direct writing to minimize memory usage - processes each
    batch and writes directly to pre-allocated output tensor without accumulating
    intermediate results.
    
    Args:
        ctx: Context from decode_all_batches containing batch_samples (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
        
    Returns:
        dict: Updated context containing:
            - final_video: Assembled video tensor [T, H, W, C] range [0,1]
            - All intermediate storage cleared for memory efficiency
            
    Raises:
        ValueError: If context is missing or has no batch samples
    """
    if debug is None:
        raise ValueError("Debug instance must be provided to postprocess_all_batches")
    
    if ctx is None:
        raise ValueError("Context is required for postprocess_all_batches. Run decode_all_batches first.")
    
    # Validate we have batch samples
    if 'batch_samples' not in ctx or not ctx['batch_samples']:
        raise ValueError("No batch samples found. Run decode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 4: Post-processing ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase4_postprocess")
    
    # Total_frames represents the original input frame count (set in Phase 1)
    total_frames = ctx.get('total_frames', 0)
    
    # Early exit if no frames to process
    if total_frames == 0:
        ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
        debug.log("No frames to process", level="WARNING", category="generation", force=True)
        return ctx
    
    # Count valid samples for progress reporting
    num_valid_samples = len([s for s in ctx['batch_samples'] if s is not None])
    
    # Calculate total post-processing work units
    # For RGBA: each batch needs 2 steps (alpha processing + color correction/assembly)
    # For RGB: each batch needs 1 step (color correction/assembly only)
    has_alpha_processing = (ctx.get('is_rgba', False) and 
                           'all_alpha_channels' in ctx and 
                           'all_input_rgb' in ctx and
                           isinstance(ctx.get('all_alpha_channels'), list))
    
    if has_alpha_processing:
        total_postprocess_steps = num_valid_samples * 2  # Alpha + main processing
    else:
        total_postprocess_steps = num_valid_samples  # Main processing only
    
    current_postprocess_step = 0
    
    # Pre-allocation will happen after processing first sample to get exact dimensions
    ctx['final_video'] = None
    current_frame_idx = 0
    
    # Alpha processing - handle RGBA inputs with edge-guided upscaling
    if ctx.get('is_rgba', False) and 'all_alpha_channels' in ctx and 'all_input_rgb' in ctx:
        debug.log("Processing Alpha channel with edge-guided upscaling...", category="alpha")
        
        # Validate alpha channel data exists
        if not isinstance(ctx.get('all_alpha_channels'), list) or not isinstance(ctx.get('all_input_rgb'), list):
            debug.log("WARNING: Alpha channel data malformed, skipping alpha processing", 
                     level="WARNING", category="alpha", force=True)
        else:
            for batch_idx in range(len(ctx['batch_samples'])):
                if ctx['batch_samples'][batch_idx] is None:
                    continue
                
                # Bounds checking for alpha channel lists
                if batch_idx >= len(ctx['all_alpha_channels']) or ctx['all_alpha_channels'][batch_idx] is None:
                    continue
                    
                # Validate alpha channel tensor integrity
                if not isinstance(ctx['all_alpha_channels'][batch_idx], torch.Tensor):
                    debug.log(f"WARNING: Alpha channel {batch_idx} is not a tensor, skipping", 
                             level="WARNING", category="alpha", force=True)
                    continue
                
                debug.log(f"Processing Alpha batch {batch_idx+1}/{num_valid_samples}", category="alpha", force=True)
                debug.start_timer(f"alpha_batch_{batch_idx+1}")

                # Process Alpha and merge with RGB
                ctx['batch_samples'][batch_idx] = process_alpha_for_batch(
                    rgb_samples=ctx['batch_samples'][batch_idx],
                    alpha_original=ctx['all_alpha_channels'][batch_idx],
                    rgb_original=ctx['all_input_rgb'][batch_idx],
                    device=ctx['vae_device'],
                    compute_dtype=ctx['compute_dtype'],
                    debug=debug
                )
                
                # Free memory immediately
                release_tensor_memory(ctx['all_alpha_channels'][batch_idx])
                ctx['all_alpha_channels'][batch_idx] = None

                release_tensor_memory(ctx['all_input_rgb'][batch_idx])
                ctx['all_input_rgb'][batch_idx] = None
            
                debug.end_timer(f"alpha_batch_{batch_idx+1}", f"Alpha batch {batch_idx+1}")
                
                # Update progress for alpha processing step
                current_postprocess_step += 1
                if progress_callback:
                    progress_callback(current_postprocess_step, total_postprocess_steps,
                                    1, "Phase 4: Post-processing")

        debug.log("Alpha processing complete for all batches", category="alpha")
    
    try:
        # Stream-based processing: write directly to final_video without accumulation
        for batch_idx, samples in enumerate(ctx['batch_samples']):
            if samples is None:
                continue
                
            check_interrupt(ctx)
            
            debug.log(f"Post-processing batch {batch_idx+1}/{num_valid_samples}", category="video", force=True)
            debug.start_timer(f"postprocess_batch_{batch_idx+1}")
            
            # Post-process each sample in the batch
            for i, sample in enumerate(samples):
                # Move to VAE device with correct dtype for processing (no-op if already there)
                sample = manage_tensor(
                    tensor=sample,
                    target_device=ctx['vae_device'],
                    tensor_name=f"sample_{batch_idx+1}_{i}",
                    dtype=ctx['compute_dtype'],
                    debug=debug,
                    reason="post-processing",
                    indent_level=1
                )
                
                # Get original length for trimming (always available)
                video_idx = min(batch_idx, len(ctx['all_ori_lengths']) - 1)
                ori_length = ctx['all_ori_lengths'][video_idx] if 'all_ori_lengths' in ctx else sample.shape[0]
                
                # Retrieve transformed video early for consistent trimming
                input_video = None
                if color_correction != "none" and ctx.get('all_transformed_videos') is not None:
                    if video_idx < len(ctx['all_transformed_videos']) and ctx['all_transformed_videos'][video_idx] is not None:
                        transformed_video = ctx['all_transformed_videos'][video_idx]
                        # Convert transformed video from C T H W to T C H W format
                        input_video = optimized_single_video_rearrange(transformed_video)
                
                # Trim both sample and input_video to original length if necessary (handles padding)
                if ori_length < sample.shape[0]:
                    sample = sample[:ori_length]
                    if input_video is not None:
                        input_video = input_video[:ori_length]
                
                # Apply color correction if enabled (RGB only)
                if color_correction != "none" and input_video is not None:
                    # Check if RGBA (samples are in T, C, H, W format at this point)
                    has_alpha = ctx.get('is_rgba', False)
                    alpha_channel = None
                    
                    if has_alpha:
                        # Check actual channel count
                        if sample.shape[1] == 4:
                            # Extract and temporarily store alpha for reattachment after color correction
                            alpha_channel = sample[:, 3:4, :, :]  # (T, 1, H, W)
                            sample = sample[:, :3, :, :]  # Keep only RGB (T, 3, H, W)
                    
                    # Ensure both tensors are on same device (GPU) for color correction
                    if input_video.device != sample.device:
                        input_video = manage_tensor(
                            tensor=input_video,
                            target_device=sample.device,
                            tensor_name=f"input_video_{batch_idx+1}",
                            debug=debug,
                            reason="color correction",
                            indent_level=1
                        )
                        
                    # Apply selected color correction method
                    debug.start_timer(f"color_correction_{color_correction}")
                    
                    if color_correction == "lab":
                        debug.log("Applying LAB perceptual color transfer", category="video", force=True, indent_level=1)
                        sample = lab_color_transfer(sample, input_video, debug, luminance_weight=0.8)
                    elif color_correction == "wavelet_adaptive":
                        debug.log("Applying wavelet with adaptive saturation correction", category="video", force=True, indent_level=1)
                        sample = wavelet_adaptive_color_correction(sample, input_video, debug)
                    elif color_correction == "wavelet":
                        debug.log("Applying wavelet color reconstruction", category="video", force=True, indent_level=1)
                        sample = wavelet_reconstruction(sample, input_video, debug)
                    elif color_correction == "hsv":
                        debug.log("Applying HSV hue-conditional saturation matching", category="video", force=True, indent_level=1)
                        sample = hsv_saturation_histogram_match(sample, input_video, debug)
                    elif color_correction == "adain":
                        debug.log("Applying AdaIN color correction", category="video", force=True, indent_level=1)
                        sample = adaptive_instance_normalization(sample, input_video)
                    else:
                        debug.log(f"Unknown color correction method: {color_correction}", level="WARNING", category="video", force=True, indent_level=1)
                    
                    debug.end_timer(f"color_correction_{color_correction}", f"Color correction ({color_correction})")
                    
                    # Free the transformed video
                    ctx['all_transformed_videos'][video_idx] = None
                    del input_video, transformed_video

                    # Recombine with Alpha if it was present in input
                    if has_alpha and alpha_channel is not None:
                        # Concatenate in channels-first: (T, 3, H, W) + (T, 1, H, W) -> (T, 4, H, W)
                        sample = torch.cat([sample, alpha_channel], dim=1)
                
                else:
                    debug.log("Color correction disabled (set to none)", category="video", indent_level=1)
                
                # Free the original length entry
                if 'all_ori_lengths' in ctx and video_idx < len(ctx['all_ori_lengths']):
                    ctx['all_ori_lengths'][video_idx] = None
                
                # Convert to final format (still on GPU at this point)
                sample = optimized_sample_to_image_format(sample)
                
                # Apply normalization only to RGB channels, preserve Alpha as-is
                if ctx.get('is_rgba', False) and sample.shape[-1] == 4:
                    # Split RGBA: sample is (T, H, W, C) format after optimized_sample_to_image_format
                    rgb_channels = sample[..., :3]  # (T, H, W, 3)
                    alpha_channel = sample[..., 3:4]  # (T, H, W, 1)
                    
                    # Normalize only RGB from [-1, 1] to [0, 1]
                    rgb_channels = rgb_channels.clip(-1, 1).mul_(0.5).add_(0.5)
                    
                    # Merge back with unchanged Alpha
                    sample = torch.cat([rgb_channels, alpha_channel], dim=-1)
                else:
                    # RGB only: apply normalization as usual
                    sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
                
                # Draw tile boundaries for debugging (if tile info available)
                for phase, attr in [('encode', 'encode_tile_boundaries'), ('decode', 'decode_tile_boundaries')]:
                    tiles = getattr(debug, attr, None)
                    if tiles:
                        sample = _draw_tile_boundaries(sample, debug, tiles, phase)
                        setattr(debug, attr, [])
                        break
                
                # Move to tensor_offload_device if specified
                if ctx['tensor_offload_device'] is not None:
                    sample = manage_tensor(
                        tensor=sample,
                        target_device=ctx['tensor_offload_device'],
                        tensor_name=f"sample_{batch_idx+1}_{i}_final",
                        debug=debug,
                        reason="storing final processed samples",
                        indent_level=1
                    )
                
                # Get batch dimensions
                batch_frames = sample.shape[0]
                
                # Pre-allocate output tensor on first write
                if ctx['final_video'] is None:
                    H, W = sample.shape[1], sample.shape[2]
                    # Use input format, not first sample's shape, for consistent allocation
                    C = 4 if ctx.get('is_rgba', False) else 3
                    channels_str = "RGBA" if C == 4 else "RGB"
                    
                    debug.log(f"Pre-allocating output tensor: {total_frames} frames, {W}x{H}px, {channels_str}", category="setup")
                    
                    target_device = ctx['tensor_offload_device'] if ctx['tensor_offload_device'] is not None else ctx['vae_device']
                    ctx['final_video'] = torch.empty((total_frames, H, W, C), dtype=ctx['compute_dtype'], device=target_device)

                # Direct write - if dimensions don't match, let it fail with clear error
                ctx['final_video'][current_frame_idx:current_frame_idx + batch_frames] = sample
                current_frame_idx += batch_frames
                
                # Immediately release sample memory
                del sample
                
            # Clear batch samples as we go to free memory progressively
            release_tensor_memory(ctx['batch_samples'][batch_idx])
            ctx['batch_samples'][batch_idx] = None
            
            debug.end_timer(f"postprocess_batch_{batch_idx+1}", f"Post-processed batch {batch_idx+1}")
            
            # Update progress for main processing step
            current_postprocess_step += 1
            if progress_callback:
                progress_callback(current_postprocess_step, total_postprocess_steps,
                                1, "Phase 4: Post-processing")

        # Verify final assembly
        if ctx['final_video'] is not None:
            final_shape = ctx['final_video'].shape
            Hf, Wf, Cf = final_shape[1], final_shape[2], final_shape[3]
            channels_str = "RGBA" if Cf == 4 else "RGB" if Cf == 3 else f"{Cf}-channel"
            
            debug.log(f"Final video assembled: {current_frame_idx} frames written, Resolution: {Wf}x{Hf}px, Channels: {channels_str}", 
                     category="generation", force=True)
            
            if current_frame_idx != total_frames:
                debug.log(f"WARNING: Frame count mismatch - expected {total_frames}, wrote {current_frame_idx}", 
                         level="WARNING", category="generation", force=True)
        else:
            ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=ctx['compute_dtype'])
            debug.log("No frames were processed", level="WARNING", category="generation", force=True)
            
    except Exception as e:
        debug.log(f"Error in Phase 4 (Post-processing): {e}", level="ERROR", category="generation", force=True)
        raise
    finally:
        # 1. Clean up batch_samples from context (already mostly freed during processing)
        if 'batch_samples' in ctx and ctx['batch_samples']:
            release_tensor_collection(ctx['batch_samples'])
            ctx['batch_samples'].clear()
            del ctx['batch_samples']
        
        # 2. Clean up video transform caches
        if ctx.get('video_transform'):
            if hasattr(ctx['video_transform'], 'transforms'):
                for transform in ctx['video_transform'].transforms:
                    # Clear cache attributes
                    for cache_attr in ['cache', '_cache']:
                        if hasattr(transform, cache_attr):
                            setattr(transform, cache_attr, None)
                    # Clear remaining attributes
                    if hasattr(transform, '__dict__'):
                        transform.__dict__.clear()
            del ctx['video_transform']
            ctx['video_transform'] = None
        
        # 3. Clean up storage lists (all_latents, all_alpha_channels, etc.)
        tensor_storage_keys = ['all_latents', 'all_transformed_videos', 
                            'all_alpha_channels', 'all_input_rgb']
        for key in tensor_storage_keys:
            if key in ctx and ctx[key]:
                release_tensor_collection(ctx[key])
                del ctx[key]
        
        # 4. Clean up non-tensor storage
        if 'all_ori_lengths' in ctx:
            del ctx['all_ori_lengths']
        
        # 5. Final deep memory clear
        clear_memory(debug=debug, deep=True, force=True, timer_name="final_memory_clear")
        
    debug.end_timer("phase4_postprocess", "Phase 4: Post-processing complete", show_breakdown=True)
    debug.log_memory_state("After phase 4 (Post-processing)", show_tensors=False)
    
    return ctx