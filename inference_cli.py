#!/usr/bin/env python3
"""
SeedVR2 Video Upscaler - Standalone CLI Interface

Multi-GPU capable command-line interface for high-quality video upscaling using
SeedVR2 diffusion models. Supports both single and multi-GPU processing with
temporal overlap blending.

Features:
    - Multi-GPU parallel processing with automatic workload distribution
    - Temporal overlap blending for smooth transitions between chunks
    - BlockSwap memory optimization for limited VRAM scenarios
    - VAE tiling for large resolution processing
    - Comprehensive dtype management (BFloat16 compute pipeline)
    - Torch.compile integration for 20-40% speedup
    - Multiple output formats (video/PNG sequences)
    - Advanced color correction methods

Usage:
    python inference_cli.py --input input.mp4|image.png|directory --resolution 1080 --model <model_name>

Requirements:
    - Python 3.12+
    - PyTorch 2.0+
    - CUDA 12.0+ (for NVIDIA GPUs)
"""

import sys
import os
import argparse
import time
import platform
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set environment variable so all spawned processes can find modules
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# -------------------------------------------------------------
# 1) Configure VRAM management (cudaMallocAsync) before heavy imports
if platform.system() != "Darwin":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    # 2) Pre-parse command line to configure CUDA_VISIBLE_DEVICES early
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--cuda_device", type=str, default=None)
    _pre_args, _ = _pre_parser.parse_known_args()
    if _pre_args.cuda_device is not None:
        device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
        if len(device_list_env) == 1:
            # Single GPU: restrict visibility now
            os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# -------------------------------------------------------------
# Import heavy dependencies (torch, etc.) after environment configuration
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils.downloads import download_weight
from src.utils.debug import Debug
from src.utils.model_registry import get_available_dit_models, DEFAULT_DIT, DEFAULT_VAE
from src.utils.constants import SEEDVR2_FOLDER_NAME
debug = Debug(enabled=False)  # Default to disabled, can be enabled via CLI


# =============================================================================
# Device Management Helpers
# =============================================================================

def _get_platform_type() -> str:
    """Determine the platform device type (cuda/mps/cpu)."""
    if platform.system() == "Darwin":
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _device_id_to_name(device_id: str, platform_type: str = None) -> str:
    """
    Convert device ID to full device name.
    
    Args:
        device_id: Device ID ("0", "1") or special value ("cpu", "none")
        platform_type: Override platform type ("cuda", "mps", "cpu")
    
    Returns:
        Full device name ("cuda:0", "mps:0", "cpu", "none")
    """
    if device_id in ("cpu", "none"):
        return device_id
    
    if platform_type is None:
        platform_type = _get_platform_type()
    
    # MPS typically doesn't use indices
    if platform_type == "mps":
        return "mps"
    
    return f"{platform_type}:{device_id}"


def _parse_offload_device(offload_arg: str, platform_type: str = None, cache_enabled: bool = False) -> Optional[str]:
    """
    Parse offload device argument to full device name.
    
    Args:
        offload_arg: Offload device argument ("none", "cpu", "0", "1", or "cuda:1")
        platform_type: Override platform type
        cache_enabled: If True and offload_arg is "none", default to "cpu"
    
    Returns:
        Full device name or None
    """
    if offload_arg == "none":
        # If caching enabled but no offload device specified, default to CPU
        return "cpu" if cache_enabled else None
    
    if offload_arg == "cpu":
        return "cpu"
    
    # If already a full device name (cuda:1, mps:0), return as-is
    if ":" in offload_arg:
        return offload_arg
    
    # Otherwise treat as device ID
    return _device_id_to_name(offload_arg, platform_type)


# =============================================================================
# Video I/O Functions
# =============================================================================
# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

def get_media_files(directory: str) -> List[str]:
    """Get all video and image files from directory, sorted."""
    files = []
    for ext in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted([str(f) for f in files])

def extract_frames_from_image(image_path: str) -> Tuple[torch.Tensor, float]:
    """Extract single frame from image file."""
    debug.log(f"Loading image: {image_path}", category="file")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize
    frame = frame.astype(np.float32) / 255.0
    
    # Convert to tensor [1, H, W, C]
    frames_tensor = torch.from_numpy(frame[None, ...]).to(torch.float16)
    
    debug.log(f"Image tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")
    
    return frames_tensor, 30.0  # Default FPS for images

def get_input_type(input_path: str) -> str:
    """Determine input type: 'video', 'image', 'directory', or 'unknown'."""
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if path.is_dir():
        return 'directory'
    
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    else:
        return "unknown"

def generate_output_path(input_path: str, output_format: str, output_dir: Optional[str] = None, 
                        input_type: Optional[str] = None) -> str:
    """
    Generate output path based on input path and format.
    
    Args:
        input_path: Source file path
        output_format: "mp4" or "png"
        output_dir: Optional output directory
        input_type: Optional input type ("image", "video", "directory")
    
    Returns:
        Output path (file for single image/video, directory for sequences)
    """
    input_name = Path(input_path).stem
    
    if output_format == "png":
        # Single image → single PNG file
        if input_type == "image":
            if output_dir:
                return str(Path(output_dir) / f"{input_name}_upscaled.png")
            return f"output/{input_name}_upscaled.png"
        # Video/sequence → directory of numbered PNGs
        else:
            if output_dir:
                return str(Path(output_dir) / f"{input_name}_upscaled")
            return f"output/{input_name}_upscaled"
    else:
        # Video format always returns file path
        if output_dir:
            return str(Path(output_dir) / f"{input_name}_upscaled.mp4")
        return f"output/{input_name}_upscaled.mp4"

def process_single_file(input_path: str, args: argparse.Namespace, device_list: List[str], 
                       output_path: Optional[str] = None, format_auto_detected: bool = False,
                       runner_cache: Optional[Dict[str, Any]] = None) -> int:
    """
    Process a single video or image file.
    
    Args:
        input_path: Path to input file
        args: Command-line arguments
        device_list: List of GPU device IDs
        output_path: Optional explicit output path
        format_auto_detected: Whether output format was auto-detected
    
    Returns:
        Number of frames processed
    """
    input_type = get_input_type(input_path)
    
    if input_type == "unknown":
        debug.log(f"Skipping unsupported file: {input_path}", level="WARNING", category="file", force=True)
        return 0
    
    debug.log(f"Processing {input_type}: {Path(input_path).name}", category="generation", force=True)
    
    # Extract frames
    if input_type == "video":
        start_time = time.time()
        frames_tensor, original_fps = extract_frames_from_video(
            input_path, args.skip_first_frames, args.load_cap, args.prepend_frames
        )
        debug.log(f"Frame extraction time: {time.time() - start_time:.2f}s", category="timing")
    else:
        frames_tensor, original_fps = extract_frames_from_image(input_path)
    
    # Track frames before processing (for FPS calculation)
    input_frame_count = len(frames_tensor)
    
    # Generate output path if not provided
    output_path = output_path or generate_output_path(input_path, args.output_format, input_type=input_type)
    
    # Show format with auto-detection indicator
    format_prefix = "Auto-detected" if format_auto_detected else "Output"
    debug.log(f"{format_prefix} output format: {args.output_format}", category="info", force=True, indent_level=1)
    
    # Process frames
    processing_start = time.time()
    # Use direct processing if caching enabled
    if runner_cache is not None:
        # Direct single-GPU processing with model caching
        result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache)
    else:
        # Multi-GPU or non-cached processing via worker processes
        result = _gpu_processing(frames_tensor, device_list, args)
    debug.log(f"Processing time: {time.time() - processing_start:.2f}s", category="timing")

    # Save results
    is_png_format = args.output_format == "png"
    is_single_image = input_type == "image"
    
    if is_png_format and is_single_image:
        # Single PNG file
        os.makedirs(Path(output_path).parent, exist_ok=True)
        frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
    
    elif is_png_format:
        # PNG sequence (save_frames_to_png creates directory internally)
        save_frames_to_png(result, output_path, base_name=Path(input_path).stem)
    
    else:
        # Video file
        os.makedirs(Path(output_path).parent, exist_ok=True)
        save_frames_to_video(result, output_path, original_fps)
    
    # Log appropriate save message based on format
    if is_png_format and not is_single_image:
        debug.log(f"PNG frames saved in directory: {output_path}", category="file", force=True)
    else:
        debug.log(f"Output saved to: {output_path}", category="file", force=True)
    
    return input_frame_count

def extract_frames_from_video(
    video_path: str, 
    skip_first_frames: int = 0, 
    load_cap: Optional[int] = None, 
    prepend_frames: int = 0
) -> Tuple[torch.Tensor, float]:
    """
    Extract frames from video file and convert to tensor format.
    
    Reads video using OpenCV, converts BGR to RGB, normalizes to [0,1] range,
    and optionally prepends reversed frames to reduce initial artifacts.
    
    Args:
        video_path: Path to input video file
        skip_first_frames: Number of initial frames to skip (default: 0)
        load_cap: Maximum number of frames to load, None loads all (default: None)
        prepend_frames: Number of frames to prepend (reversed from start) to reduce
                       initial artifacts (default: 0)
        
    Returns:
        Tuple containing:
            - frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
            - fps: Original video frames per second
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or no frames extracted
    """
    debug.log(f"Extracting frames from video: {video_path}", category="file")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    debug.log(f"Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS", category="info")
    if skip_first_frames:
        debug.log(f"Will skip first {skip_first_frames} frames", category="info")
    if load_cap:
        debug.log(f"Will load maximum {load_cap} frames", category="info")
    if prepend_frames:
        debug.log(f"Will prepend {prepend_frames} frames to the video", category="info")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frame if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            continue

        if skip_first_frames > 0 and frame_idx == skip_first_frames:
            debug.log(f"Skipped first {skip_first_frames} frames", category="info") 

        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            debug.log(f"Reached load cap of {load_cap} frames", category="info")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug.enabled and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            debug.log(f"Extracted {frames_loaded}/{total_to_load} frames", category="file")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    debug.log(f"Extracted {len(frames)} frames", category="success")

    # Convert to tensor (will be cast to compute_dtype in worker process)
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float32)
    
    # Apply prepend frames using shared function
    if prepend_frames > 0:
        from src.core.generation_utils import prepend_video_frames
        frames_tensor = prepend_video_frames(frames_tensor, prepend_frames, debug)
    
    debug.log(f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")

    return frames_tensor, fps


def save_frames_to_video(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0
) -> None:
    """
    Save frames tensor to MP4 video file.
    
    Converts tensor from Float32 [0,1] to uint8 [0,255], RGB to BGR for OpenCV,
    and writes to video file using mp4v codec.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_path: Output video file path (will be created if doesn't exist)
        fps: Frames per second for output video (default: 30.0)
    
    Raises:
        ValueError: If video writer cannot be initialized
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames to video: {output_path}", category="file")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if debug.enabled and (i + 1) % 100 == 0:
            debug.log(f"Saved {i + 1}/{T} frames", category="file")

    out.release()
    
    debug.log(f"Video saved successfully: {output_path}", category="success")


def save_frames_to_png(
    frames_tensor: torch.Tensor, 
    output_dir: str, 
    base_name: str
) -> None:
    """
    Save frames tensor as sequential PNG image files.
    
    Each frame saved as {base_name}_{index:05d}.png with zero-padded indices.
    Converts Float32 [0,1] to uint8 [0,255] and RGB to BGR for OpenCV.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C], Float32, range [0,1]
        output_dir: Directory to save PNG files (created if doesn't exist)
        base_name: Base name for output files (e.g., "frame" → "frame_00000.png")
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames as PNGs to directory: {output_dir}", category="file")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy uint8 RGB
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    digits = max(5, len(str(total)))  # at least 5 digits

    for idx, frame in enumerate(frames_np):
        filename = f"{base_name}_{idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, frame_bgr)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} PNGs", category="file")

    debug.log(f"PNG saving completed: {total} files in '{output_dir}'", category="success")


# =============================================================================
# Core Processing Logic
# =============================================================================

def _process_frames_core(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    debug: 'Debug',
    runner_cache: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Core frame processing logic shared between worker and direct processing.
    
    Executes the complete 4-phase pipeline: encode → upscale → decode → postprocess.
    Supports both cached (direct) and non-cached (worker) execution modes.
    
    Args:
        frames_tensor: Input frames [T, H, W, C], Float16/Float32, range [0,1]
        args: Command-line arguments with all processing settings
        device_id: Device ID for inference ("0", "1", etc.)
        debug: Debug instance for logging
        runner_cache: Optional cache dict for model reuse (direct mode only)
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
    """
    from src.core.generation_utils import setup_generation_context, prepare_runner
    from src.core.generation_phases import (
        encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches
    )
    
    # Determine platform and convert device IDs to full names
    platform_type = _get_platform_type()
    inference_device = _device_id_to_name(device_id, platform_type)
    
    # Parse offload devices (with caching defaults)
    cache_dit = args.cache_dit if runner_cache is not None else False
    cache_vae = args.cache_vae if runner_cache is not None else False
    
    dit_offload = _parse_offload_device(args.dit_offload_device, platform_type, cache_dit)
    vae_offload = _parse_offload_device(args.vae_offload_device, platform_type, cache_vae)
    tensor_offload = _parse_offload_device(args.tensor_offload_device, platform_type, False)
    
    # Setup or reuse generation context
    if runner_cache is not None and 'ctx' in runner_cache:
        ctx = runner_cache['ctx']
        # Clear previous run data but keep device config
        keys_to_keep = {'dit_device', 'vae_device', 'dit_offload_device', 
                       'vae_offload_device', 'tensor_offload_device', 'compute_dtype'}
        for key in list(ctx.keys()):
            if key not in keys_to_keep:
                del ctx[key]
    else:
        ctx = setup_generation_context(
            dit_device=inference_device,
            vae_device=inference_device,
            dit_offload_device=dit_offload,
            vae_offload_device=vae_offload,
            tensor_offload_device=tensor_offload,
            debug=debug
        )
        if runner_cache is not None:
            runner_cache['ctx'] = ctx
    
    # Build torch compile args
    torch_compile_args_dit = None
    torch_compile_args_vae = None
    if args.compile_dit:
        torch_compile_args_dit = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    if args.compile_vae:
        torch_compile_args_vae = {
            "backend": args.compile_backend,
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": args.compile_dynamic,
            "dynamo_cache_size_limit": args.compile_dynamo_cache_size_limit,
            "dynamo_recompile_limit": args.compile_dynamo_recompile_limit,
        }
    
    # Prepare runner with caching support
    model_dir = args.model_dir if args.model_dir is not None else f"./models/{SEEDVR2_FOLDER_NAME}"
    
    # Use fixed IDs for CLI caching when enabled
    dit_id = "cli_dit" if cache_dit else None
    vae_id = "cli_vae" if cache_vae else None
    
    runner, cache_context = prepare_runner(
        dit_model=args.model,
        vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug,
        ctx=ctx,
        dit_cache=cache_dit,
        vae_cache=cache_vae,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config={
            'blocks_to_swap': args.blocks_to_swap,
            'swap_io_components': args.swap_io_components,
            'offload_device': dit_offload,
        },
        encode_tiled=args.vae_encode_tiling_enabled,
        encode_tile_size=args.vae_encode_tile_size,
        encode_tile_overlap=args.vae_encode_tile_overlap,
        decode_tiled=args.vae_decode_tiling_enabled,
        decode_tile_size=args.vae_decode_tile_size,
        decode_tile_overlap=args.vae_decode_tile_overlap,
        tile_debug=args.tile_debug.lower() if args.tile_debug else "false",
        attention_mode=args.attention_mode,
        torch_compile_args_dit=torch_compile_args_dit,
        torch_compile_args_vae=torch_compile_args_vae
    )
    
    ctx['cache_context'] = cache_context
    if runner_cache is not None:
        runner_cache['runner'] = runner
    
    # Phase 1: Encode
    ctx = encode_all_batches(
        runner, ctx=ctx, images=frames_tensor,
        debug=debug, 
        batch_size=args.batch_size,
        seed=args.seed,
        progress_callback=None, 
        temporal_overlap=args.temporal_overlap,
        resolution=args.resolution,
        max_resolution=args.max_resolution,
        input_noise_scale=args.input_noise_scale,
        color_correction=args.color_correction
    )
    
    # Phase 2: Upscale
    ctx = upscale_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        seed=args.seed,
        latent_noise_scale=args.latent_noise_scale,
        cache_model=cache_dit
    )
    
    # Phase 3: Decode
    ctx = decode_all_batches(
        runner, ctx=ctx, debug=debug, progress_callback=None,
        cache_model=cache_vae
    )
    
    # Phase 4: Post-process
    ctx = postprocess_all_batches(
        ctx=ctx, debug=debug, progress_callback=None,
        color_correction=args.color_correction,
        prepend_frames=0,  # Worker mode handles this in main process
        temporal_overlap=args.temporal_overlap,
        batch_size=args.batch_size
    )
    
    result_tensor = ctx['final_video']
    
    # Convert to CPU and compatible dtype
    if result_tensor.is_cuda or result_tensor.is_mps:
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
    
    return result_tensor


def _worker_process(
    proc_idx: int, 
    device_id: int, 
    frames_np: np.ndarray, 
    shared_args: Dict[str, Any], 
    return_queue: mp.Queue
) -> None:
    """
    Worker process for multi-GPU upscaling.
    
    Sets up isolated CUDA environment and calls core processing logic.
    Results returned via multiprocessing queue as numpy arrays.
    """
    if platform.system() != "Darwin":
        # Limit CUDA visibility to the chosen GPU BEFORE importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import torch
    
    # Create debug instance for this worker
    worker_debug = Debug(enabled=shared_args["debug"])
    
    # Convert numpy back to tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16)
    
    # Create args namespace from shared_args
    args = argparse.Namespace(**shared_args)
    
    # Process frames (no caching in worker mode)
    result_tensor = _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id="0",  # Always "0" in worker (CUDA_VISIBLE_DEVICES set)
        debug=worker_debug,
        runner_cache=None  # No caching in multiprocessing mode
    )
    
    # Send back result as numpy array
    return_queue.put((proc_idx, result_tensor.numpy()))


def _single_gpu_direct_processing(
    frames_tensor: torch.Tensor,
    args: argparse.Namespace,
    device_id: str,
    runner_cache: Dict[str, Any]
) -> torch.Tensor:
    """
    Direct single-GPU processing with model caching support.
    
    Uses main process and shared runner cache for efficient multi-file processing.
    """
    return _process_frames_core(
        frames_tensor=frames_tensor,
        args=args,
        device_id=device_id,
        debug=debug,
        runner_cache=runner_cache
    )


def _gpu_processing(
    frames_tensor: torch.Tensor, 
    device_list: List[str], 
    args: argparse.Namespace
) -> torch.Tensor:
    """
    Orchestrate multi-GPU parallel video upscaling with temporal overlap blending.
    
    Splits input frames across multiple GPUs with optional temporal overlap,
    spawns worker processes for parallel processing, and reassembles results
    with smooth blending of overlapping regions.
    
    Processing flow:
        1. Split frames into chunks (with overlap if enabled)
        2. Spawn worker processes on each GPU
        3. Wait for all workers to complete
        4. Blend overlapping regions using Hann window crossfade
        5. Remove prepended frames from final result
    
    Args:
        frames_tensor: Input frames [T, H, W, C], Float32, range [0,1]
        device_list: List of device IDs as strings (e.g., ["0", "1"])
        args: Parsed command-line arguments containing all processing settings
    
    Returns:
        Upscaled frames tensor [T', H', W', C], Float32, range [0,1]
        where T' may be less than T if prepend_frames were removed
    
    Note:
        - Single GPU: Can use multiprocessing or direct processing
        - Multi-GPU with overlap: Chunks sized to multiples of batch_size for
          proper temporal blending
        - Prepended frames removed after all GPU workers complete (multi-GPU safe)
    """
    num_devices = len(device_list)
    total_frames = frames_tensor.shape[0]
    
    # Create overlapping chunks (for multi GPU); ensures every chunk is 
    # a multiple of batch_size (except last one) to avoid blending issues
    if args.temporal_overlap > 0 and num_devices > 1:
        chunk_with_overlap = total_frames // num_devices + args.temporal_overlap
        if args.batch_size > 1:
            chunk_with_overlap = ((chunk_with_overlap + args.batch_size - 1) // args.batch_size) * args.batch_size
        base_chunk_size = chunk_with_overlap - args.temporal_overlap

        chunks = []
        for i in range(num_devices):
            start_idx = i * base_chunk_size
            if i == num_devices - 1: # last chunk/device
                end_idx = total_frames
            else:
                end_idx = min(start_idx + chunk_with_overlap, total_frames)
            chunks.append(frames_tensor[start_idx:end_idx])
    else:
        chunks = torch.chunk(frames_tensor, num_devices, dim=0)

    # Use direct Queue with explicit unlimited size for large video chunks
    return_queue = mp.Queue(maxsize=0)  # 0 = unlimited (explicit)
    workers = []

    # Convert args namespace to dict for serialization
    shared_args = vars(args).copy()

    # Start all workers
    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    # Collect results before joining to prevent deadlock
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1
    
    # Now safe to join
    for p in workers:
        p.join()

    # Concatenate results with overlap blending using shared function
    if args.temporal_overlap > 0 and num_devices > 1:
        from src.core.generation_utils import blend_overlapping_frames
        
        overlap = args.temporal_overlap
        result_tensor = None
        
        for idx, res_np in enumerate(results_np):
            chunk_tensor = torch.from_numpy(res_np).to(torch.float32)
            
            if idx == 0:
                # First chunk: keep all frames
                result_tensor = chunk_tensor
            else:
                # Subsequent chunks: blend overlapping region with accumulated result
                if chunk_tensor.shape[0] > overlap and result_tensor.shape[0] >= overlap:
                    # Get overlapping regions
                    prev_tail = result_tensor[-overlap:]  # Last N frames from accumulated result
                    cur_head = chunk_tensor[:overlap]      # First N frames from current chunk
                    
                    # Blend using shared function
                    blended = blend_overlapping_frames(prev_tail, cur_head, overlap)
                    
                    # Replace tail of result with blended frames, then append rest of chunk
                    result_tensor = torch.cat([
                        result_tensor[:-overlap],           # Everything except the tail
                        blended,                            # Blended overlapping frames
                        chunk_tensor[overlap:]              # Non-overlapping part of current chunk
                    ], dim=0)
                else:
                    # Edge case: chunk too small, just append non-overlapping part
                    if chunk_tensor.shape[0] > overlap:
                        result_tensor = torch.cat([result_tensor, chunk_tensor[overlap:]], dim=0)
        
        if result_tensor is None:
            result_tensor = torch.from_numpy(results_np[0]).to(torch.float32)
    else:
        # Simple concatenation without overlap
        result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float32)

    # Handle prepend_frames removal (multi-GPU safe - done after all workers complete)
    if args.prepend_frames > 0:
        if args.prepend_frames < result_tensor.shape[0]:
            debug.log(f"Removing {args.prepend_frames} prepended frames from output", category="generation")
            result_tensor = result_tensor[args.prepend_frames:]
        else:
            debug.log(f"Warning: prepend_frames ({args.prepend_frames}) >= total frames ({result_tensor.shape[0]}), skipping removal", 
                     level="WARNING", category="generation")
    
    return result_tensor

# =============================================================================
# Argument Parsing
# =============================================================================

class OneOrTwoValues(argparse.Action):
    """
    Custom argparse action for tile size arguments accepting 1 or 2 integers.
    
    Allows flexible input formats:
        - Single integer: --tile_size 1024 → (1024, 1024)
        - Two integers: --tile_size 1024 768 → (1024, 768)
        - Comma-separated: --tile_size 1024,768 → (1024, 768)
    
    Used for VAE tiling parameters where height and width can be specified
    separately or as a single value applied to both dimensions.
    """
    
    def __call__(
        self, 
        parser: argparse.ArgumentParser, 
        namespace: argparse.Namespace, 
        values: List[str], 
        option_string: Optional[str] = None
    ) -> None:
        """Parse and validate tile size arguments."""
        if len(values) not in [1, 2]:
            parser.error(f"{option_string} requires 1 or 2 arguments")

        if len(values) == 1:
            values = values[0]
            if ',' in values:
                values = [v.strip() for v in values.split(',') if v.strip()]
            else:
                values = values.split()
        
        try:
            result = tuple(int(v) for v in values)
            if len(result) == 1:
                result = (result[0], result[0])  # Convert single value to (h, w)
            setattr(namespace, self.dest, result)
        except ValueError:
            parser.error(f"{option_string} arguments must be integers")


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for SeedVR2 CLI.
    
    Configures all available options including model selection, processing parameters,
    memory optimization settings, and output configuration. Uses custom action classes
    for complex argument types (e.g., OneOrTwoValues for tile sizes).
    
    Returns:
        Parsed arguments namespace with all CLI parameters
    
    Note:
        - cuda_device argument only available on non-macOS systems
        - Default model directory resolves to "models/SEEDVR2" if not specified
        - Tile size/overlap arguments use OneOrTwoValues for flexible input
    """
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input video file, image file, or directory containing videos/images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation (default: 42)")
    parser.add_argument("--resolution", type=int, default=1080,
                        help="Target resolution of the short side (default: 1080)")
    parser.add_argument("--max_resolution", type=int, default=0,
                        help="Maximum resolution for any edge. Scales down proportionally if exceeded after --resolution is applied. 0 = no limit (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 1)")
    parser.add_argument("--model", type=str, default=DEFAULT_DIT,
                        choices=get_available_dit_models(),
                        help="Model to use (default: 3B FP8)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help=f"Directory containing the model files (default: models/{SEEDVR2_FOLDER_NAME})")
    parser.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip the first frames during processing")
    parser.add_argument("--load_cap", type=int, default=0,
                        help="Maximum number of frames to load from video (default: load all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated, if output_format is png, it will be a directory)")
    parser.add_argument("--output_format", type=str, default=None, 
                        choices=["mp4", "png", None],
                        help="Output format: 'mp4' video or 'png' images (default: auto-detect from input)")
    parser.add_argument("--color_correction", type=str, default="lab", 
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    help="Color correction method: 'lab' (full perceptual color matching with detail preservation, recommended), 'wavelet' (frequency-based natural colors, preserves details), 'wavelet_adaptive' (wavelet base + targeted saturation correction), 'hsv' (hue-conditional saturation matching), 'adain' (statistical style transfer), 'none' (no correction)")
    parser.add_argument("--input_noise_scale", type=float, default=0.0,
                        help="Input noise scale (0.0-1.0) to reduce artifacts at high resolutions. (default: 0.0)")
    parser.add_argument("--latent_noise_scale", type=float, default=0.0,
                        help="Latent space noise scale (0.0-1.0). Adds noise during diffusion, can soften details. Use if input_noise doesn't help (default: 0.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    if platform.system() != "Darwin":
        parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device id(s). Single id (e.g., '0') or comma-separated list '0,1' for multi-GPU")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Number of transformer blocks to swap for VRAM optimization (default: 0, disabled). "
                             "Up to 32 for 3B model, 36 for 7B model. Requires --dit_offload_device to be set.")
    parser.add_argument("--temporal_overlap", type=int, default=0,
                        help="Temporal overlap for processing (default: 0, no temporal overlap)")
    parser.add_argument("--prepend_frames", type=int, default=0,
                        help="Number of frames to prepend to the video (default: 0). This can help with artifacts at the start of the video and are automatically removed after processing")
    parser.add_argument("--swap_io_components", action="store_true",
                        help="Offload DiT input/output embeddings and normalization layers for additional VRAM savings. "
                             "Requires --dit_offload_device to be set. Can be used alone or with --blocks_to_swap.")
    parser.add_argument("--dit_offload_device", type=str, default="none",
                        help="Device to offload DiT model when not in use (default: none). "
                             "Options: 'none' (keep on inference device), 'cpu' (offload to RAM/system memory), "
                             "or GPU device ID like '1' (offload to another GPU). "
                             "Required when BlockSwap is enabled (blocks_to_swap > 0 or swap_io_components = True). "
                             "Multi-GPU example: --cuda_device 0 --dit_offload_device 1 distributes memory across GPUs.")
    parser.add_argument("--vae_offload_device", type=str, default="none",
                        help="Device to offload VAE when not in use (default: none). "
                             "Options: 'none' (keep on inference device), 'cpu' (offload to RAM/system memory), "
                             "or GPU device ID like '1' (offload to another GPU). "
                             "Use 'cpu' or another GPU to free VRAM between encode/decode phases.")
    parser.add_argument("--tensor_offload_device", type=str, default="cpu",
                        help="Device to offload intermediate tensors between phases (default: cpu). "
                             "Options: 'cpu' (offload to RAM/system memory - recommended), 'none' (keep on inference device), "
                             "or GPU device ID like '1' (offload to another GPU). "
                             "Use 'cpu' to prevent VRAM accumulation for long videos.")
    parser.add_argument("--cache_dit", action="store_true",
                        help="Cache DiT model between files for faster multi-file processing (single GPU only). Model cached on device specified by --dit_offload_device (default: cpu)")
    parser.add_argument("--cache_vae", action="store_true",
                        help="Cache VAE model between files for faster multi-file processing (single GPU only). Model cached on device specified by --vae_offload_device (default: cpu)")
    parser.add_argument("--vae_encode_tiling_enabled", action="store_true",
                        help="Enable VAE encode tiling for VRAM reduction during encoding. Disabled by default.")
    parser.add_argument("--vae_encode_tile_size", action=OneOrTwoValues, nargs='+', default=(1024, 1024),
                        help="VAE encode tile size in pixels (default: 1024). Only used when encode tiling is enabled. Adjust based on available VRAM. Use single integer or two integers 'h w'.")
    parser.add_argument("--vae_encode_tile_overlap", action=OneOrTwoValues, nargs='+', default=(128, 128),
                        help="VAE encode tile overlap in pixels (default: 128). Only used when encode tiling is enabled. Higher values improve blending at the cost of slower processing. Use single integer or two integers 'h w'.")
    parser.add_argument("--vae_decode_tiling_enabled", action="store_true",
                        help="Enable VAE decode tiling for VRAM reduction during decoding. Disabled by default.")
    parser.add_argument("--vae_decode_tile_size", action=OneOrTwoValues, nargs='+', default=(1024, 1024),
                        help="VAE decode tile size in pixels (default: 1024). Only used when decode tiling is enabled. Adjust based on available VRAM. Use single integer or two integers 'h w'.")
    parser.add_argument("--vae_decode_tile_overlap", action=OneOrTwoValues, nargs='+', default=(128, 128),
                        help="VAE decode tile overlap in pixels (default: 128). Only used when decode tiling is enabled. Higher values improve blending at the cost of slower processing. Use single integer or two integers 'h w'.")
    parser.add_argument("--tile_debug", type=str, default="false", 
                        choices=["false", "encode", "decode"],
                        help="Enable tile debug visualization: 'false' (default, no overlay), 'encode' (show encode tiles), 'decode' (show decode tiles). Only works when respective tiling is enabled.")
    parser.add_argument("--attention_mode", type=str, default="sdpa",
                        choices=["sdpa", "flash_attn"],
                        help="Attention computation backend: 'sdpa' (default, always available) or 'flash_attn' (requires flash-attn package, faster)")
    parser.add_argument("--compile_dit", action="store_true", 
                        help="Enable torch.compile for DiT model (20-40%% speedup, requires PyTorch 2.0+)")
    parser.add_argument("--compile_vae", action="store_true",
                        help="Enable torch.compile for VAE model (15-25%% speedup, requires PyTorch 2.0+)")
    parser.add_argument("--compile_backend", type=str, default="inductor", choices=["inductor", "cudagraphs"],
                        help="Torch compile backend (default: inductor)")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Torch compile mode (default: default)")
    parser.add_argument("--compile_fullgraph", action="store_true",
                        help="Compile entire model as single graph. False allows graph breaks (more compatible), True enforces no breaks (maximum optimization but fragile). Default: False")
    parser.add_argument("--compile_dynamic", action="store_true",
                        help="Handle varying input shapes without recompilation. False specializes for exact shapes, True creates dynamic kernels. Default: False")
    parser.add_argument("--compile_dynamo_cache_size_limit", type=int, default=64,
                        help="Max cached compiled versions per function. Higher = more memory, lower = more recompilation. Default: 64")
    parser.add_argument("--compile_dynamo_recompile_limit", type=int, default=128,
                        help="Max recompilation attempts before falling back to uncompiled. Safety limit to prevent infinite loops. Default: 128")
    return parser.parse_args()

# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """
    Main entry point for SeedVR2 Video Upscaler CLI.
    
    Orchestrates the complete upscaling workflow:
        1. Parse and validate command-line arguments
        2. Extract frames from input video/image(s)
        3. Download required models if not cached
        4. Process frames on single or multiple GPUs
        5. Save results as video or PNG sequence
        6. Report timing and FPS (calculated from total wall-clock time)
    
    Error handling:
        - Validates tile configuration before processing
        - Provides detailed error messages with traceback
        - Ensures proper cleanup on exit (VRAM automatically freed)
    
    Raises:
        SystemExit: On argument validation failure or processing error
    """
    # print header
    debug.print_header(cli=True)
    
    # Parse arguments
    args = parse_arguments()
    debug.enabled = args.debug
    
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"{key}: {value}", category="none", indent_level=1)

    if args.vae_encode_tiling_enabled and (args.vae_encode_tile_overlap[0] >= args.vae_encode_tile_size[0] or args.vae_encode_tile_overlap[1] >= args.vae_encode_tile_size[1]):
        debug.log(f"VAE encode tile overlap {args.vae_encode_tile_overlap} must be smaller than tile size {args.vae_encode_tile_size}", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    if args.vae_decode_tiling_enabled and (args.vae_decode_tile_overlap[0] >= args.vae_decode_tile_size[0] or args.vae_decode_tile_overlap[1] >= args.vae_decode_tile_size[1]):
        debug.log(f"VAE decode tile overlap {args.vae_decode_tile_overlap} must be smaller than tile size {args.vae_decode_tile_size}", level="ERROR", category="vae", force=True)
        sys.exit(1)
    
    # Validate BlockSwap configuration - either blocks_to_swap or swap_io_components requires dit_offload_device
    blockswap_enabled = args.blocks_to_swap > 0 or args.swap_io_components
    if blockswap_enabled and args.dit_offload_device == "none":
        config_details = []
        if args.blocks_to_swap > 0:
            config_details.append(f"blocks_to_swap={args.blocks_to_swap}")
        if args.swap_io_components:
            config_details.append("swap_io_components=True")
        
        debug.log(
            f"BlockSwap enabled ({', '.join(config_details)}) but dit_offload_device='none'. "
            "BlockSwap requires dit_offload_device to be set (typically 'cpu'). "
            "Either set --dit_offload_device cpu or disable BlockSwap "
            "(--blocks_to_swap 0 and do not use --swap_io_components)",
            level="ERROR", category="blockswap", force=True
        )
        sys.exit(1)
    
    # Inform about caching defaults
    if args.cache_dit and args.dit_offload_device == "none":
        offload_target = "system memory (CPU)" if _get_platform_type() != "mps" else "unified memory"
        debug.log(
            f"DiT caching enabled: Using default {offload_target} for offload. "
            "Set --dit_offload_device explicitly to use a different device.",
            category="cache", force=True
        )
    
    if args.cache_vae and args.vae_offload_device == "none":
        offload_target = "system memory (CPU)" if _get_platform_type() != "mps" else "unified memory"
        debug.log(
            f"VAE caching enabled: Using default {offload_target} for offload. "
            "Set --vae_offload_device explicitly to use a different device.",
            category="cache", force=True
        )

    if args.debug:
        if platform.system() == "Darwin":
            debug.log("You are running on macOS and will use the MPS backend!", category="info", force=True)
        else:
            # Show actual CUDA device visibility
            debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}", category="device")
            if torch.cuda.is_available():
                debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
                debug.log(f"Using device index 0 inside script (mapped to selected GPU)", category="device")
    
    try:
        start_time = time.time()
        
        # Parse GPU list
        if platform.system() == "Darwin":
            device_list = ["0"]
        else:
            if args.cuda_device:
                device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()]
            else:
                device_list = ["0"]
        if args.debug:
            debug.log(f"Using devices: {device_list}", category="device")
        
        # Download models once before processing
        if not download_weight(dit_model=args.model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug):
            debug.log("Failed to download required models. Check console output above.", level="ERROR", category="download", force=True)
            sys.exit(1)
        
        # Determine input type and process accordingly
        input_type = get_input_type(args.input)

        # Track total frames for FPS calculation (time tracked via start_time)
        total_frames_processed = 0
        
        # Track if output format was user-specified or auto-detected
        format_auto_detected = args.output_format is None
        
        if input_type == 'directory':
            media_files = get_media_files(args.input)
            if not media_files:
                debug.log(f"No video or image files found in directory: {args.input}", 
                        level="ERROR", category="file", force=True)
                sys.exit(1)
            
            debug.log(f"Found {len(media_files)} media files to process", category="file", force=True)
            
            # Validate caching with multi-GPU (not supported in CLI - would need shared memory)
            if (args.cache_dit or args.cache_vae) and len(device_list) > 1:
                debug.log(
                    "Model caching requires single GPU selection (you selected multiple GPUs). "
                    "Disabling caching for this run.", 
                    level="WARNING", category="cache", force=True
                )
                args.cache_dit = False
                args.cache_vae = False
            
            # Initialize runner cache if caching enabled
            runner_cache = {} if (args.cache_dit or args.cache_vae) else None
            
            for idx, file_path in enumerate(media_files, 1):
                # Visual separation between files (except before first file)
                if idx > 1:
                    debug.log("", category="none", force=True)
                    debug.log("━" * 60, category="none", force=True)
                    debug.log("", category="none", force=True)
                
                debug.log(f"Processing file {idx}/{len(media_files)}", category="generation", force=True)
                
                # Auto-detect format per file if not user-specified
                if format_auto_detected:
                    file_type = get_input_type(file_path)
                    file_output_format = "mp4" if file_type == "video" else "png"
                else:
                    file_output_format = args.output_format
                
                # Temporarily override args.output_format for this file
                original_format = args.output_format
                args.output_format = file_output_format
                
                # generate_output_path handles None gracefully with "outputs" default
                output_path = generate_output_path(file_path, file_output_format, args.output, 
                                   input_type=get_input_type(file_path))
                
                # Process with explicit output path and runner cache
                frames = process_single_file(file_path, args, device_list, output_path, 
                                            format_auto_detected=format_auto_detected,
                                            runner_cache=runner_cache)
                total_frames_processed += frames
                
                # Restore original format
                args.output_format = original_format

        elif input_type in ("video", "image"):
            # Auto-detect output format for single file if not specified
            if format_auto_detected:
                args.output_format = "mp4" if input_type == "video" else "png"
            
            # Validate caching for single file (would provide no benefit but shouldn't error)
            if (args.cache_dit or args.cache_vae):
                if len(device_list) > 1:
                    debug.log(
                        "Model caching requires single GPU selection (you selected multiple GPUs). "
                        "Disabling caching for this run.",
                        level="WARNING", category="cache", force=True
                    )
                    args.cache_dit = False
                    args.cache_vae = False
                else:
                    debug.log(
                        "Model caching has no benefit for single file processing (only useful for directories). "
                        "Consider removing --cache_dit/--cache_vae for single files.",
                        category="tip", force=True
                    )
            
            # No caching for single file (no benefit)
            frames = process_single_file(args.input, args, device_list, args.output,
                                        format_auto_detected=format_auto_detected,
                                        runner_cache=None)
            total_frames_processed += frames
        
        else:
            debug.log(f"Unsupported input type: {args.input}", level="ERROR", category="file", force=True)
            sys.exit(1)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        debug.log("", category="none", force=True)
        debug.log(f"All upscaling processes completed successfully in {total_time:.2f}s", category="success", force=True)
        
        # Calculate and display FPS based on overall wall-clock time
        if total_time > 0 and total_frames_processed > 0:
            fps = total_frames_processed / total_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
        
    except Exception as e:
        debug.log(f"Error during processing: {e}", level="ERROR", category="generation", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)

        # print footer
        debug.print_footer()

if __name__ == "__main__":
    main()