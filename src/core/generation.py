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
- Adaptive dtype detection and optimal autocast configuration
- Memory-efficient pre-allocated batch processing
- Advanced video format handling (4n+1 constraint)
- Clean separation of concerns with phase-specific resource management
- Each phase handles its own cleanup in finally blocks
"""

import os
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.debug import Debug
    from .infer import VideoDiffusionInfer
from torchvision.transforms import Compose, Lambda, Normalize

from ..utils.constants import get_script_directory
from ..common.distributed import get_device
from .model_manager import configure_runner, materialize_model
from ..optimization.memory_manager import (
    cleanup_dit,
    cleanup_vae,
    cleanup_text_embeddings,
    complete_cleanup,
    clear_memory,
    manage_model_device
)
from ..optimization.performance import (
    optimized_video_rearrange, 
    optimized_single_video_rearrange, 
    optimized_sample_to_image_format
)
from ..optimization.blockswap import apply_block_swap_to_dit
from ..common.seed import set_seed
from ..data.image.transforms.divisible_crop import DivisibleCrop
from ..data.image.transforms.na_resize import NaResize
from ..utils.color_fix import wavelet_reconstruction, adaptive_instance_normalization
from ..utils.vae_rgba_adapter import adapt_vae_for_rgba, restore_vae_rgb_only

# Get script directory for embeddings
script_directory = get_script_directory()

def prepare_video_transforms(res_w: int) -> Compose:
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
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


def load_text_embeddings(script_directory: str, device: Union[str, torch.device], 
                        dtype: torch.dtype) -> Dict[str, List[torch.Tensor]]:
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (str): Target device
        dtype (torch.dtype): Target dtype
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=dtype)
    
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


def prepare_generation_context(dit_device: str, vae_device: str, debug: Optional['Debug'] = None) -> Dict[str, Any]:
    """
    Initialize generation context for the upscaling pipeline.
    
    Creates a context dictionary that holds all state and configuration needed
    throughout the generation process.
    Supports independent device placement for DiT and VAE models.
    
    Args:
        dit_device: Device for DiT model (required, e.g., "cuda:0", "cuda:1", "cpu")
        vae_device: Device for VAE model (required, e.g., "cuda:0", "cuda:1", "cpu")
        debug: Debug instance for logging
        
    Raises:
        ValueError: If dit_device or vae_device is None
        
    Note:
        Precision settings (compute_dtype, autocast_dtype) are lazily initialized
        when first needed via _ensure_precision_initialized() to detect actual
        model dtypes and configure optimal computation settings.
    """
    if dit_device is None or vae_device is None:
        raise ValueError("Device must be provided to prepare_generation_context")
    
    try:
        import comfy.model_management
        interrupt_fn = comfy.model_management.throw_exception_if_processing_interrupted
        comfyui_available = True
    except:
        interrupt_fn = None
        comfyui_available = False
    
    ctx = {
        'dit_device': dit_device,
        'vae_device': vae_device,
        'compute_dtype': None,
        'autocast_dtype': None,
        'video_transform': None,
        'text_embeds': None,
        'all_transformed_videos': [],
        'all_latents': [],
        'all_upscaled_latents': [],
        'batch_samples': [],
        'final_video': None,
        'comfyui_available': comfyui_available,
        'interrupt_fn': interrupt_fn,  # Store the function reference
    }
    
    if debug:
        debug.log("Initialized generation context", category="setup")
    
    return ctx


def _ensure_precision_initialized(ctx: Dict[str, Any], 
                                runner: 'VideoDiffusionInfer', 
                                debug: Optional['Debug'] = None) -> None:
    """Lazily initialize precision settings if not already done"""
    if ctx.get('compute_dtype') is not None:
        return  # Already initialized
    
    try:
        # Get real dtype of loaded models
        dit_dtype = next(runner.dit.parameters()).dtype
        vae_dtype = next(runner.vae.parameters()).dtype

        # Use BFloat16 for all models
        # - FP8 models: BFloat16 required for arithmetic operations
        # - FP16 models: BFloat16 provides better numerical stability and prevents black frames
        # - BFloat16 models: Already optimal
        if dit_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        elif dit_dtype == torch.float16:
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        else:
            ctx['compute_dtype'] = torch.bfloat16
            ctx['autocast_dtype'] = torch.bfloat16
        
        if debug:
            debug.log(f"Initialized precision: DiT={dit_dtype}, VAE={vae_dtype}, compute={ctx['compute_dtype']}, autocast={ctx['autocast_dtype']}", category="precision")
    except Exception as e:
        # Fallback to safe defaults
        ctx['compute_dtype'] = torch.bfloat16
        ctx['autocast_dtype'] = torch.bfloat16
        
        if debug:
            debug.log(f"Could not detect model dtypes: {e}, falling back to BFloat16", level="WARNING", category="model", force=True)


def setup_device_environment(dit_device: Optional[str] = None, 
                            vae_device: Optional[str] = None,
                            debug: Optional['Debug'] = None) -> Tuple[str, str]:
    """
    Setup device environment variables before model loading.
    This must be called before configure_runner.
    
    Handles independent device configuration for DiT and VAE models,
    allowing them to run on different GPUs for load balancing.
    
    Args:
        dit_device: Device string for DiT model (e.g., "cuda:0", "none")
        vae_device: Device string for VAE model (e.g., "cuda:1", "none")
        debug: Debug instance for logging
        
    Returns:
        Tuple[str, str]: (processed_dit_device, processed_vae_device)
    """
    # Get default device if not specified
    default_device = "cpu"
    if torch.cuda.is_available() or torch.mps.is_available():
        default_device = str(get_device())
    
    # Apply defaults
    if dit_device is None:
        dit_device = default_device
    if vae_device is None:
        vae_device = default_device
    
    # Set LOCAL_RANK based on primary device (DiT device)
    # This is for distributed compatibility with the main model
    if dit_device != "none" and ":" in dit_device:
        os.environ["LOCAL_RANK"] = dit_device.split(":")[1]
    else:
        os.environ["LOCAL_RANK"] = "0"
    
    if debug:
        debug.log(f"Device environment configured: DiT={dit_device}, VAE={vae_device}, LOCAL_RANK={os.environ['LOCAL_RANK']}", category="setup")
    
    return dit_device, vae_device


def prepare_runner(dit_model: str, vae_model: str, model_dir: str, preserve_vram: bool, debug: 'Debug',
                   cache_model_dit: bool = False, cache_model_vae: bool = False, block_swap_config: Optional[Dict[str, Any]] = None,
                   encode_tiled: bool = False, encode_tile_size: Optional[Tuple[int, int]] = None,
                   encode_tile_overlap: Optional[Tuple[int, int]] = None,
                   decode_tiled: bool = False, decode_tile_size: Optional[Tuple[int, int]] = None,
                   decode_tile_overlap: Optional[Tuple[int, int]] = None,
                   torch_compile_args_dit: Optional[Dict[str, Any]] = None,
                   torch_compile_args_vae: Optional[Dict[str, Any]] = None,
                   cached_runner: Optional['VideoDiffusionInfer'] = None) -> Tuple['VideoDiffusionInfer', bool]:
    """
    Prepare runner with model state management.
    Handles model changes and caching logic with independent DiT/VAE caching support.
    
    Args:
        dit_model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
        vae_model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
        model_dir: Base directory containing model files
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
        Tuple[VideoDiffusionInfer, bool]: (configured runner, whether models changed)
        
    Features:
        - Independent DiT and VAE caching for flexible memory management
        - Dynamic model reloading when models change
        - Optional torch.compile optimization for inference speedup
        - Separate encode/decode tiling configuration for optimal performance
        - Memory optimization and BlockSwap integration
    """
    dit_changed = False
    vae_changed = False
    
    # Check for model changes if we have a cached runner
    if cached_runner is not None:
        current_dit = getattr(cached_runner, '_dit_model_name', None)
        current_vae = getattr(cached_runner, '_vae_model_name', None)
        
        # Check DiT model change
        if current_dit != dit_model:
            dit_changed = True
            debug.log(f"DiT model changed from {current_dit} to {dit_model}, clearing DiT cache...", category="cache")
            cleanup_dit(runner=cached_runner, debug=debug, keep_model_in_ram=False)
        
        # Check VAE model change
        if current_vae != vae_model:
            vae_changed = True
            debug.log(f"VAE model changed from {current_vae} to {vae_model}, clearing VAE cache...", category="cache")
            cleanup_vae(runner=cached_runner, debug=debug, keep_model_in_ram=False)
        
        # If either model changed, invalidate cached_runner for that model
        # Keep runner alive only if at least one model is still valid and cached
        if dit_changed and vae_changed:
            # Both changed - full recreation needed
            cached_runner = None
        elif dit_changed and not cache_model_vae:
            # DiT changed and VAE not cached - full recreation
            cached_runner = None
        elif vae_changed and not cache_model_dit:
            # VAE changed and DiT not cached - full recreation
            cached_runner = None
        # Otherwise keep cached_runner for partial reuse
    
    # Configure runner
    debug.log("Configuring inference runner...", category="runner")
    runner = configure_runner(
        dit_model=dit_model,
        vae_model=vae_model,
        base_cache_dir=model_dir,
        preserve_vram=preserve_vram,
        debug=debug,
        cache_model_dit=cache_model_dit,
        cache_model_vae=cache_model_vae,
        block_swap_config=block_swap_config,
        encode_tiled=encode_tiled,
        encode_tile_size=encode_tile_size,
        encode_tile_overlap=encode_tile_overlap,
        decode_tiled=decode_tiled,
        decode_tile_size=decode_tile_size,
        decode_tile_overlap=decode_tile_overlap,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae,
        cached_runner=cached_runner if (cache_model_dit or cache_model_vae) else None
    )
    
    # Store both model names for future comparisons
    runner._dit_model_name = dit_model
    runner._vae_model_name = vae_model
    
    model_changed = dit_changed or vae_changed
    return runner, model_changed


def encode_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                       images: Optional[torch.Tensor] = None, batch_size: int = 1, 
                       preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                       progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
                       temporal_overlap: int = 0, res_w: int = 1072, 
                       input_noise_scale: float = 0.0,
                       color_correction: str = "wavelet") -> Dict[str, Any]:
    """
    Phase 1: VAE Encoding for all batches
    
    Encodes video frames to latents in batches, handling temporal overlap and 
    memory optimization. Creates context automatically if not provided.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Generation context from prepare_generation_context (required)
        images: Input frames tensor [T, H, W, C] in float16, range [0,1]. 
                Required if ctx doesn't contain 'input_images'
        batch_size: Frames per batch (4n+1 format: 1, 5, 9, 13...)
        preserve_vram: If True, offload VAE between operations
        debug: Debug instance for logging (required)
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
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
    # Validate and store inputs
    if images is None and 'input_images' not in ctx:
        raise ValueError("Either images must be provided or ctx must contain 'input_images'")
    
    if images is not None:
        ctx['input_images'] = images
    else:
        images = ctx['input_images']
    
    total_frames = len(images)
    if total_frames == 0:
        raise ValueError("No frames to process")
    
    # Setup video transformation pipeline if not already done
    if ctx.get('video_transform') is None:
        debug.start_timer("video_transform")
        ctx['video_transform'] = prepare_video_transforms(res_w)
        debug.log(f"Initialized video transformation pipeline for {res_w}px", category="setup")
        debug.end_timer("video_transform", "Video transform pipeline initialization")
    else:
        debug.log(f"Reusing pre-initialized video transformation pipeline", category="setup")
    
    # Detect if input is RGBA (4 channels) - we'll adapt AFTER loading weights
    if images[0].shape[-1] == 4:  # RGBA detected
        ctx['is_rgba'] = True
    else:
        ctx['is_rgba'] = False
    
    # Display batch optimization tips
    if total_frames > 0:
        batch_params = calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap)
        if batch_params['padding_waste'] > 0:
            debug.log("", category="none", force=True)
            debug.log(f"Padding waste: {batch_params['padding_waste']}", category="info", force=True)
            debug.log(f"  Why padding? Each batch must be 4n+1 frames (1, 5, 9, 13, 17, 21, ...)", category="info", force=True)
            debug.log(f"  Current batch_size creates partial batches that need padding to meet this constraint", category="info", force=True)
            debug.log(f"  This increases memory usage and processing time unnecessarily", category="info", force=True)
        
        if batch_params['best_batch'] != batch_size and batch_params['best_batch'] <= total_frames:
            debug.log("", category="none", force=True)
            debug.log(f"For {total_frames} frames, use batch_size={batch_params['best_batch']} for better efficiency", category="tip", force=True)
            debug.log(f"  Consider larger batches: better temporal coherence BUT uses more memory", category="tip", force=True)
        
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
            materialize_model(runner, "vae", str(ctx['vae_device']), runner.config, 
                            preserve_vram, debug)
        
        # Move VAE to GPU for encoding (no-op if already there)
        manage_model_device(model=runner.vae, target_device=str(ctx['vae_device']), 
                          model_name="VAE", preserve_vram=False, debug=debug,
                          runner=runner)
    
        # Adapt VAE for RGBA after weights are loaded
        if ctx.get('is_rgba', False):
            runner.vae = adapt_vae_for_rgba(runner.vae, debug)
        
        debug.log_memory_state("After VAE loading", detailed_tensors=False)
        
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
            video = video.permute(0, 3, 1, 2).to(ctx['vae_device'], dtype=ctx['compute_dtype'])
            
            # Apply transformations
            transformed_video = ctx['video_transform'](video)

            del video

            # Apply input noise if requested (to reduce artifacts at high resolutions)
            if input_noise_scale > 0:
                debug.log(f"  Applying input noise (scale: {input_noise_scale:.2f})", category="video")
                
                # Generate noise matching the video shape
                noise = torch.randn_like(transformed_video)
                
                # Subtle noise amplitude
                noise = noise * 0.05
                
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                
                # Apply blend
                transformed_video = transformed_video * (1 - blend_factor) + (transformed_video + noise) * blend_factor
                
                del noise

            ori_length = transformed_video.size(1)
            
            # Log sequence info
            t = transformed_video.size(1)
            debug.log(f"  Sequence of {t} frames", category="video", force=True)

            # Handle 4n+1 constraint
            if t % 4 != 1:
                target = ((t-1)//4+1)*4+1
                padding_frames = target - t
                debug.log(f"  Applying padding: {padding_frames} frame{'s' if padding_frames != 1 else ''} added ({t} -> {target})", category="video", force=True)
                transformed_video = cut_videos(transformed_video)
            
            # Store original length for proper trimming later
            ctx['all_ori_lengths'][encode_idx] = ori_length
            
            # Store transformed video on CPU if needed for color correction
            if color_correction != "none":
                # Move to CPU to free VRAM
                transformed_video_cpu = transformed_video.to('cpu', non_blocking=False)
                ctx['all_transformed_videos'][encode_idx] = transformed_video_cpu
            
            # Encode to latents
            cond_latents = runner.vae_encode([transformed_video])
            
            # Offload latents to CPU to save VRAM between phases
            ctx['all_latents'][encode_idx] = cond_latents[0].to('cpu', non_blocking=False)
            
            del cond_latents, transformed_video
            
            debug.end_timer(f"encode_batch_{encode_idx+1}", f"Encoded batch {encode_idx+1}")
            
            if progress_callback:
                progress_callback(encode_idx+1, num_encode_batches, 
                                current_frames, "Phase 1: Encoding")
            
            encode_idx += 1
            
    except Exception as e:
        debug.log(f"Error in Phase 1 (Encoding): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Always offload VAE if needed
        if preserve_vram:
            manage_model_device(model=runner.vae, target_device='cpu', 
                              model_name="VAE", preserve_vram=preserve_vram, debug=debug,
                              runner=runner)
    
    debug.end_timer("phase1_encoding", "Phase 1: VAE encoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 1 (VAE encoding)", show_tensors=False)
    
    return ctx


def upscale_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                        preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                        progress_callback: Optional[Callable[[int, int, int, str], None]] = None, 
                        cfg_scale: float = 1.0, seed: int = 100, 
                        latent_noise_scale: float = 0.0, cache_model: bool = False) -> Dict[str, Any]:
    """
    Phase 2: DiT Upscaling for all encoded batches.
    
    Processes all encoded latents through the diffusion model for upscaling.
    Requires context from encode_all_batches with encoded latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from encode_all_batches containing latents (required)
        preserve_vram: If True, offload DiT between operations
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cfg_scale: Classifier-free guidance scale (default: 1.0)
        seed: Random seed for noise generation
        latent_noise_scale: Noise scale for latent space augmentation (0.0-1.0).
                           Adds noise during diffusion conditioning. Can soften details
                           but may help with certain artifacts. 0.0 = no noise (crisp),
                           1.0 = maximum noise (softer)
        cache_model: If True, keep DiT model in RAM for reuse instead of deleting it
        
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
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
    # Validate we have encoded latents
    if 'all_latents' not in ctx or not ctx['all_latents']:
        raise ValueError("No encoded latents found. Run encode_all_batches first.")
    
    debug.log("", category="none", force=True)
    debug.log("━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━", category="none", force=True)
    debug.start_timer("phase2_upscaling")
    
    # Load text embeddings if not already loaded
    if ctx.get('text_embeds') is None:
        ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'])
        debug.log("Loaded text embeddings for DiT", category="dit")
    
    # Configure diffusion parameters
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(dtype=ctx['compute_dtype'])
    
    # Set seed for generation
    set_seed(seed)
    
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
            materialize_model(runner, "dit", str(ctx['dit_device']), runner.config, 
                            preserve_vram, debug)
        
        # Move DiT to GPU for upscaling (no-op if already there)
        manage_model_device(model=runner.dit, target_device=str(ctx['dit_device']), 
                            model_name="DiT", preserve_vram=False, debug=debug,
                            runner=runner)
        
        debug.log_memory_state("After DiT loading", detailed_tensors=False)

        for batch_idx, latent in enumerate(ctx['all_latents']):
            if latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Upscaling batch {upscale_idx+1}/{num_valid_latents}", category="generation", force=True)
            debug.start_timer(f"upscale_batch_{upscale_idx+1}")
            
            # Move latent from CPU to device with correct dtype
            latent = latent.to(ctx['dit_device'], dtype=ctx['compute_dtype'], non_blocking=False)
            
            # Generate noise
            if torch.mps.is_available():
                base_noise = torch.randn_like(latent, dtype=ctx['compute_dtype'])
            else:
                with torch.cuda.device(ctx['dit_device']):
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
            
            # Run inference
            debug.start_timer(f"dit_inference_{upscale_idx+1}")
            with torch.no_grad():
                with torch.autocast(str(ctx['dit_device']), ctx['autocast_dtype'], enabled=True):
                    upscaled = runner.inference(
                        noises=noises,
                        conditions=conditions,
                        **ctx['text_embeds'],
                    )
            debug.end_timer(f"dit_inference_{upscale_idx+1}", f"DiT inference {upscale_idx+1}")
            
            # Store upscaled result on CPU to save VRAM
            ctx['all_upscaled_latents'][upscale_idx] = upscaled[0].to('cpu', non_blocking=False)
            
            # Free original latent
            ctx['all_latents'][batch_idx] = None
            
            del noises, aug_noises, latent, conditions, condition, base_noise, upscaled
            
            if preserve_vram and ctx['all_upscaled_latents'][upscale_idx].shape[0] > 1:
                clear_memory(debug=debug, deep=True, force=True, timer_name=f"upscale_all_batches - batch {upscale_idx+1} - deep")
            
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
                debug.log(f"  BlockSwap overhead: {total_time:.2f}ms", category="blockswap")
                debug.log(f"  Total swaps: {swap_summary['total_swaps']}", category="blockswap")
                
                # Show block swap details
                if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                    avg_ms = swap_summary.get('block_avg_ms', 0)
                    total_ms = swap_summary.get('block_total_ms', 0)
                    min_ms = swap_summary.get('block_min_ms', 0)
                    max_ms = swap_summary.get('block_max_ms', 0)
                    
                    debug.log(f"  Block swaps: {swap_summary['block_swaps']} "
                            f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                            category="blockswap")
                    
                    # Show most frequently swapped block
                    if 'most_swapped_block' in swap_summary:
                        debug.log(f"  Most swapped: Block {swap_summary['most_swapped_block']} "
                                f"({swap_summary['most_swapped_count']} times)", category="blockswap")
                
                # Show I/O swap details if present
                if 'io_swaps' in swap_summary and swap_summary['io_swaps'] > 0:
                    debug.log(f"  I/O swaps: {swap_summary['io_swaps']} "
                            f"(avg: {swap_summary.get('io_avg_ms', 0):.2f}ms, total: {swap_summary.get('io_total_ms', 0):.2f}ms)", 
                            category="blockswap")

        # Cleanup DiT as it's no longer needed after upscaling
        cleanup_dit(runner=runner, debug=debug, keep_model_in_ram=cache_model)
        
        # Cleanup text embeddings as they're no longer needed after upscaling
        cleanup_text_embeddings(ctx, debug)
        
        clear_memory(debug=debug, deep=True, force=True, timer_name="upscale_all_batches_finally")
    
    debug.end_timer("phase2_upscaling", "Phase 2: DiT upscaling complete", show_breakdown=True)
    debug.log_memory_state("After phase 2 (DiT upscaling)", show_tensors=False)
    
    return ctx


def decode_all_batches(runner: 'VideoDiffusionInfer', ctx: Optional[Dict[str, Any]] = None, 
                       preserve_vram: bool = False, debug: Optional['Debug'] = None, 
                       progress_callback: Optional[Callable[[int, int, int, str], None]] = None, 
                       cache_model: bool = False) -> Dict[str, Any]:
    """
    Phase 3: VAE Decoding.
    
    Decodes all upscaled latents back to pixel space.
    Requires context from upscale_all_batches with upscaled latents.
    
    Args:
        runner: VideoDiffusionInfer instance with loaded models (required)
        ctx: Context from upscale_all_batches containing upscaled latents (required)
        preserve_vram: If True, offload VAE between operations
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        cache_model: If True, keep VAE model in RAM for reuse instead of deleting it
        
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
    
    # Ensure precision is initialized
    _ensure_precision_initialized(ctx, runner, debug)
    
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
            materialize_model(runner, "vae", str(ctx['vae_device']), runner.config, 
                            preserve_vram, debug)
        
        # Move VAE to GPU for decoding
        manage_model_device(model=runner.vae, target_device=str(ctx['vae_device']), 
                          model_name="VAE", preserve_vram=False, debug=debug,
                          runner=runner)
        
        debug.log_memory_state("After VAE loading", detailed_tensors=False)
        
        for batch_idx, upscaled_latent in enumerate(ctx['all_upscaled_latents']):
            if upscaled_latent is None:
                continue
            
            check_interrupt(ctx)
            
            debug.log(f"Decoding batch {decode_idx+1}/{num_valid_latents}", category="vae", force=True)
            debug.start_timer(f"decode_batch_{decode_idx+1}")
            
            # Move latent to device with correct dtype for decoding
            upscaled_latent = upscaled_latent.to(ctx['vae_device'], dtype=ctx['compute_dtype'], non_blocking=False)
            
            # Decode latent
            debug.start_timer("vae_decode")
            samples = runner.vae_decode([upscaled_latent], preserve_vram=preserve_vram)
            debug.end_timer("vae_decode", "VAE decode")
            
            # Convert to Float16 for efficiency
            if samples and len(samples) > 0 and samples[0].dtype != torch.float16:
                debug.log(f"Converting from {samples[0].dtype} to Float16", category="precision")
                samples = [sample.to(torch.float16, non_blocking=False) for sample in samples]
            
            # Process samples
            samples = optimized_video_rearrange(samples)
            
            # Store decoded samples for post-processing
            ctx['batch_samples'][decode_idx] = samples
            
            # Free the upscaled latent
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
        # Always restore VAE to RGB (3-channel) state before cleanup
        if hasattr(runner, 'vae') and runner.vae is not None:
            if hasattr(runner.vae, 'encoder') and runner.vae.encoder.conv_in.in_channels == 4:
                runner.vae = restore_vae_rgb_only(runner.vae, debug)
        
        # Cleanup VAE as it's no longer needed
        cleanup_vae(runner=runner, debug=debug, keep_model_in_ram=cache_model)
        
        # Clean up upscaled latents storage
        if 'all_upscaled_latents' in ctx:
            del ctx['all_upscaled_latents']
        
    debug.end_timer("phase3_decoding", "Phase 3: VAE decoding complete", show_breakdown=True)
    debug.log_memory_state("After phase 3 (VAE decoding)", show_tensors=False)
    
    return ctx


def postprocess_all_batches(ctx: Optional[Dict[str, Any]] = None,
                         debug: Optional['Debug'] = None,
                         progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
                         color_correction: str = "wavelet") -> Dict[str, Any]:
    """
    Phase 4: Post-processing and Final Video Assembly.
    
    Applies color correction and assembles the final video from decoded batches.
    Requires context from decode_all_batches with decoded samples.
    
    Args:
        ctx: Context from decode_all_batches containing batch_samples (required)
        debug: Debug instance for logging (required)
        progress_callback: Optional callback(current, total, frames, phase_name)
        color_correction: Color correction method - "wavelet", "adain", or "none" (default: "wavelet")
        
    Returns:
        dict: Updated context containing:
            - final_video: Assembled video tensor [T, H, W, C] in float16, range [0,1]
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
    
    # Count valid samples
    num_valid_samples = len([s for s in ctx['batch_samples'] if s is not None])
    processed_samples = []
    
    try:
        for batch_idx, samples in enumerate(ctx['batch_samples']):
            if samples is None:
                continue
                
            check_interrupt(ctx)
            
            debug.log(f"Post-processing batch {batch_idx+1}/{num_valid_samples}", category="video", force=True)
            debug.start_timer(f"postprocess_batch_{batch_idx+1}")
            
            # Post-process each sample in the batch
            for i, sample in enumerate(samples):
                # Get original length for trimming (always available)
                video_idx = min(batch_idx, len(ctx['all_ori_lengths']) - 1)
                ori_length = ctx['all_ori_lengths'][video_idx] if 'all_ori_lengths' in ctx else sample.shape[0]
                
                # Trim to original length if necessary
                if ori_length < sample.shape[0]:
                    sample = sample[:ori_length]
                
                # Apply color correction if enabled (RGB only)
                if color_correction != "none" and ctx.get('all_transformed_videos') is not None:
                    # Check if RGBA (samples are in T, C, H, W format at this point)
                    has_alpha = ctx.get('is_rgba', False)
                    alpha_channel = None
                    
                    if has_alpha:
                        debug.log("  RGBA color correction: preserving Alpha channel", category="video")
                        # Split in channels-first format: (T, 4, H, W) -> (T, 3, H, W) + (T, 1, H, W)
                        rgb_sample = sample[:, :3, :, :]  # RGB channels
                        alpha_channel = sample[:, 3:4, :, :]  # Alpha channel
                        sample = rgb_sample
                        
                    # Get transformed video from CPU
                    transformed_video = ctx['all_transformed_videos'][video_idx]
                    
                    # Move to GPU for color correction
                    transformed_video = transformed_video.to(ctx['vae_device'], non_blocking=False)
                    
                    # Adjust transformed video to handle cases when padding was applied during encoding
                    if transformed_video.shape[1] != sample.shape[0]:
                        debug.log(f"  Trimming reference from {transformed_video.shape[1]} to {sample.shape[0]} frames", 
                                category="video")
                        transformed_video = transformed_video[:, :sample.shape[0]]
                    
                    # Split reference video to RGB if it's RGBA (for color correction)
                    if has_alpha and transformed_video.shape[0] == 4:  # Check channel dimension (C, T, H, W)
                        transformed_video = transformed_video[:3, ...]  # Keep only RGB channels
                    
                    input_video = [optimized_single_video_rearrange(transformed_video)]
                    
                    # Apply selected color correction method
                    debug.start_timer(f"color_correction_{color_correction}")
                    
                    if color_correction == "wavelet":
                        debug.log("  Applying wavelet color reconstruction", category="video")
                        sample = wavelet_reconstruction(sample, input_video[0], debug)
                    elif color_correction == "adain":
                        debug.log("  Applying AdaIN color correction", category="video")
                        sample = adaptive_instance_normalization(sample, input_video[0])
                    else:
                        debug.log(f"  Unknown color correction method: {color_correction}", level="WARNING", category="video", force=True)
                    
                    debug.end_timer(f"color_correction_{color_correction}", f"Color correction ({color_correction})")
                    
                    # Free the transformed video
                    ctx['all_transformed_videos'][video_idx] = None
                    del input_video, transformed_video

                    # Recombine with Alpha if it was present in input
                    if has_alpha and alpha_channel is not None:
                        # Concatenate in channels-first: (T, 3, H, W) + (T, 1, H, W) -> (T, 4, H, W)
                        sample = torch.cat([sample, alpha_channel], dim=1)
                
                else:
                    debug.log("  Color correction disabled (set to none)", category="video")
                
                # Free the original length entry
                if 'all_ori_lengths' in ctx and video_idx < len(ctx['all_ori_lengths']):
                    ctx['all_ori_lengths'][video_idx] = None
                
                # Convert to final format
                sample = optimized_sample_to_image_format(sample)
                sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
                sample_cpu = sample.to(torch.float16).to("cpu")
                del sample
                
                # Store processed sample
                processed_samples.append(sample_cpu)
            
            debug.end_timer(f"postprocess_batch_{batch_idx+1}", f"Post-processed batch {batch_idx+1}")
            
            if progress_callback:
                progress_callback(batch_idx+1, num_valid_samples,
                                1, "Phase 4: Post-processing")

        debug.log("Assembling final video from processed batches...", category="video")

        # Merge all processed samples into final video
        if processed_samples:
            total_frames = sum(batch.shape[0] for batch in processed_samples)
            
            if total_frames > 0:
                sample_shape = processed_samples[0].shape
                H, W, C = sample_shape[1], sample_shape[2], sample_shape[3]
                
                debug.log(f"Total frames: {total_frames}, shape per frame: {H}x{W}x{C}", category="info")
                
                # Pre-allocate final tensor
                ctx['final_video'] = torch.empty((total_frames, H, W, C), dtype=torch.float16)
                
                # Copy batch results into final tensor
                current_idx = 0
                for batch in processed_samples:
                    batch_frames = batch.shape[0]
                    ctx['final_video'][current_idx:current_idx + batch_frames] = batch
                    current_idx += batch_frames
                
                final_shape = ctx['final_video'].shape
                Hf, Wf, Cf = final_shape[1], final_shape[2], final_shape[3]

                # Format channel info for readability
                channels_str = "RGBA" if Cf == 4 else "RGB" if Cf == 3 else f"{Cf}-channel"
                debug.log(f"Final video assembled: Total frames: {total_frames}, Resolution: {Wf}x{Hf}px, Channels: {channels_str}", category="generation", force=True)
            else:
                ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=torch.float16)
                debug.log("No frame to assemble", level="WARNING", category="video", force=True)
        else:
            ctx['final_video'] = torch.empty((0, 0, 0, 0), dtype=torch.float16)
            debug.log("No samples to assemble", level="WARNING", category="video", force=True)
            
    except Exception as e:
        debug.log(f"Error in Phase 4 (Post-processing): {e}", level="ERROR", category="error", force=True)
        raise
    finally:
        # Clean up post-processing intermediate storage
        if 'batch_samples' in ctx:
            ctx['batch_samples'].clear()
            del ctx['batch_samples']
        
        # Clean up video transform
        if ctx.get('video_transform'):
            for transform in ctx['video_transform'].transforms:
                if hasattr(transform, '__dict__'):
                    transform.__dict__.clear()
            ctx['video_transform'] = None
            
        # Clean up any remaining intermediate storage
        for storage_key in ['all_latents', 'all_transformed_videos', 'all_ori_lengths']:
            if storage_key in ctx:
                del ctx[storage_key]
        
        # Final deep memory clear to ensure everything is cleaned
        clear_memory(debug=debug, deep=True, force=True, timer_name="final_memory_clear")
        
    debug.end_timer("phase4_postprocess", "Phase 4: Post-processing complete", show_breakdown=True)
    debug.log_memory_state("After phase 4 (Post-processing)", show_tensors=False)
    
    return ctx