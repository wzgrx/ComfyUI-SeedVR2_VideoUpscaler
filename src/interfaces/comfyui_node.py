# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any, Optional

from ..utils.constants import get_base_cache_dir, get_script_directory
from ..utils.downloads import download_weight
from ..utils.model_registry import (
    get_available_vae_models,
    get_available_dit_models,
    DEFAULT_DIT,
    DEFAULT_VAE
)
from ..utils.debug import Debug
from ..core.model_manager import configure_runner
from ..core.generation import (
    setup_device_environment,
    prepare_generation_context, 
    prepare_runner,
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches,
    postprocess_all_batches,
    prepare_video_transforms
)
from ..optimization.memory_manager import (
    cleanup_text_embeddings,
    complete_cleanup, 
    get_device_list
)

# Import ComfyUI progress reporting
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

script_directory = get_script_directory()

class SeedVR2:
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """
    
    def __init__(self):
        """Initialize SeedVR2 node"""
        self.runner = None
        self.ctx = None
        self.debug = None
        self._dit_model_name = ""
        self._vae_model_name = ""


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define ComfyUI input parameter types and constraints
        
        Returns:
            Dictionary defining input parameters, types, and validation
        """
        return {
            "required": {
                "pixels": ("IMAGE", {
                    "tooltip": "Input video frames. Accepts both RGB (3-channel) and RGBA (4-channel) images. Output will match input format."
                }),
                "dit": ("SEEDVR2_DIT", {
                    "tooltip": "DiT model configuration from SeedVR2 Load DiT Model node"
                }),
                "vae": ("SEEDVR2_VAE", {
                    "tooltip": "VAE model configuration from SeedVR2 Load VAE Model node"
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1, 
                    "step": 1,
                    "tooltip": "Random seed for generation. Same seed = same output."
                }),
            },
            "optional": {
                "new_resolution": ("INT", {
                    "default": 1072, 
                    "min": 16, 
                    "max": 16384, 
                    "step": 16,
                    "tooltip": "Target resolution for the shortest edge. Maintains aspect ratio."
                }),
                "batch_size": ("INT", {
                    "default": 5,
                    "min": 1, 
                    "max": 16384,
                    "step": 4,
                    "tooltip": "Frames processed per batch. At least 5 for temporal consistency. Higher = better quality but more VRAM."
                }),
                "color_correction": (["wavelet", "adain", "none"], {
                    "default": "wavelet",
                    "tooltip": "Color correction method: wavelet=natural, adain=stylistic, none=raw output"
                }),
                "input_noise_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.001,
                    "tooltip": "Add noise to input to reduce high-res artifacts."
                }),
                "latent_noise_scale": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.00,
                    "max": 1.00,
                    "step": 0.001,
                    "tooltip": "Add noise during diffusion for detail softening (use sparingly)"
                }),
                "preserve_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models between pipeline steps (slower but saves VRAM)"
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed memory usage and timing information during generation, useful for troubleshooting."
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, pixels: torch.Tensor, dit: Dict[str, Any], vae: Dict[str, Any], 
                seed: int, new_resolution: int = 1072, batch_size: int = 5,
                color_correction: str = "wavelet", input_noise_scale: float = 0.0, 
                latent_noise_scale: float = 0.0, preserve_vram: bool = False,
                enable_debug: bool = False) -> Tuple[torch.Tensor]:
        """
        Execute SeedVR2 video upscaling with progress reporting
        
        Main entry point for ComfyUI node execution.
        Automatically detects and preserves input format (RGB or RGBA). Handles model downloads,
        configuration unpacking, and delegates to internal execution pipeline.
        
        Args:
            pixels: Input video frames as tensor (N, H, W, C) in [0, 1] range
            dit: DiT model configuration from SeedVR2LoadDiTModel node containing:
                - model: Model filename
                - device: Target device
                - blocks_to_swap: BlockSwap configuration
                - swap_io_components: I/O component swap flag
                - cache_in_ram: Model caching flag
                - torch_compile_args: Optional compilation settings
            vae: VAE model configuration from SeedVR2LoadVAEModel node containing:
                - model: Model filename
                - device: Target device
                - encode_tiled: Enable tiled encoding
                - encode_tile_size: Encoding tile size
                - encode_tile_overlap: Encoding tile overlap
                - decode_tiled: Enable tiled decoding
                - decode_tile_size: Decoding tile size
                - decode_tile_overlap: Decoding tile overlap
                - cache_in_ram: Model caching flag
                - torch_compile_args: Optional compilation settings
            seed: Random seed for reproducible generation
            new_resolution: Target resolution for shortest edge (maintains aspect ratio)
            batch_size: Frames per batch (minimum 5 for temporal consistency)
            color_correction: Color correction method ("wavelet", "adain", or "none")
            input_noise_scale: Input noise injection scale [0.0-1.0]
            latent_noise_scale: Latent noise injection scale [0.0-1.0]
            preserve_vram: Offload models between pipeline phases to save VRAM
            enable_debug: Enable detailed logging and memory tracking
            
        Returns:
            Tuple containing upscaled video tensor (N, H', W', C) in [0, 1] range
            
        Raises:
            ValueError: If model files cannot be downloaded or configuration is invalid
            RuntimeError: If generation pipeline fails
            
        Note:
            - Models are automatically downloaded on first use
            - Minimum batch_size of 5 recommended for temporal consistency
            - Higher batch_size improves quality but requires more VRAM
        """
        
        # Unpack DiT configuration
        dit_model = dit["model"]
        dit_device = dit["device"]
        blocks_to_swap = dit.get("blocks_to_swap", 0)
        swap_io_components = dit.get("swap_io_components", False)
        cache_model_dit = dit.get("cache_in_ram", False)
        dit_torch_compile_args = dit.get("torch_compile_args", None)

        # Unpack VAE configuration
        vae_model = vae["model"]
        vae_device = vae["device"]
        encode_tiled = vae.get("encode_tiled", False)
        encode_tile_size = vae.get("encode_tile_size", 512)
        encode_tile_overlap = vae.get("encode_tile_overlap", 64)
        decode_tiled = vae.get("decode_tiled", False) 
        decode_tile_size = vae.get("decode_tile_size", 512)
        decode_tile_overlap = vae.get("decode_tile_overlap", 64)
        cache_model_vae = vae.get("cache_in_ram", False)
        vae_torch_compile_args = vae.get("torch_compile_args", None)
        
        # Create block_swap_config if blocks_to_swap > 0
        block_swap_config = None
        if blocks_to_swap > 0:
            block_swap_config = {
                "blocks_to_swap": blocks_to_swap,
                "swap_io_components": swap_io_components,
            }
        
        # Fixed parameters (could be exposed in future)
        temporal_overlap = 0
        
        # Initialize debug
        self.debug = Debug(enabled=enable_debug)
        debug = self.debug

        self.debug.log("", category="none", force=True)
        self.debug.log(" ╔══════════════════════════════════════════════════════════╗", category="none", force=True)
        self.debug.log(" ║ ███████ ███████ ███████ ██████  ██    ██ ██████  ███████ ║", category="none", force=True)
        self.debug.log(" ║ ██      ██      ██      ██   ██ ██    ██ ██   ██      ██ ║", category="none", force=True)
        self.debug.log(" ║ ███████ █████   █████   ██   ██ ██    ██ ██████  █████   ║", category="none", force=True)
        self.debug.log(" ║      ██ ██      ██      ██   ██  ██  ██  ██   ██ ██      ║", category="none", force=True)
        self.debug.log(" ║ ███████ ███████ ███████ ██████    ████   ██   ██ ███████ ║", category="none", force=True)
        self.debug.log(" ║                         © ByteDance Seed · NumZ · AInVFX ║", category="none", force=True)
        self.debug.log(" ╚══════════════════════════════════════════════════════════╝", category="none", force=True)
        self.debug.log("", category="none", force=True)

        self.debug.start_timer("total_execution", force=True)

        self.debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        self.debug.log_memory_state("Before model preparation", show_tensors=False, detailed_tensors=False)
        self.debug.start_timer("model_preparation")

        # Check if download succeeded
        if not download_weight(dit_model=dit_model, vae_model=vae_model, debug=self.debug):
            raise RuntimeError(
                f"Failed to download required model files. "
                f"DiT model: {dit_model}, VAE model: {vae_model}. "
                "Please check the console output above for specific file failures and manual download instructions."
            )

        cfg_scale = 1.0
        try:
            # Detect input format and handle alpha channel convention
            if pixels.shape[-1] == 4:
                # RGBA detected: ComfyUI inverts alpha when loading images
                # (ComfyUI mask convention: 1=hidden, 0=visible vs standard alpha: 1=opaque, 0=transparent)
                # Invert alpha in-place: multiply by -1 then add 1 (equivalent to 1.0 - alpha)
                self.debug.log("RGBA input detected - inverting alpha channel to match mask convention", category="info")
                pixels[..., 3].mul_(-1).add_(1)
            
            return self._internal_execute(pixels, dit_model, vae_model, seed, new_resolution, cfg_scale, 
                                        batch_size, color_correction, input_noise_scale, latent_noise_scale, 
                                        encode_tiled, encode_tile_size, encode_tile_overlap, 
                                        decode_tiled, decode_tile_size, decode_tile_overlap, 
                                        preserve_vram, temporal_overlap, 
                                        cache_model_dit, cache_model_vae, dit_device, vae_device, 
                                        dit_torch_compile_args, vae_torch_compile_args, block_swap_config)
        except Exception as e:
            self.cleanup(cache_model_dit=cache_model_dit, cache_model_vae=cache_model_vae, debug=self.debug)
            raise e
        

    def cleanup(self, cache_model_dit: bool = False, cache_model_vae: bool = False, 
               debug: Optional['Debug'] = None) -> None:
        """
        Cleanup runner and free memory
        
        Args:
            cache_model_dit: If True, keep DiT model in RAM for future runs
            cache_model_vae: If True, keep VAE model in RAM for future runs
            debug: Optional debug instance for logging cleanup operations
        """
        # Clear progress bar if it exists
        if hasattr(self, '_pbar') and self._pbar is not None:
            self._pbar.update_absolute(0, 100)
            self._pbar = None
            
        # Get debug from runner if not provided
        if debug is None and self.runner and hasattr(self.runner, 'debug'):
            debug = self.runner.debug
        
        # Use complete_cleanup for all cleanup operations
        if self.runner:
            complete_cleanup(runner=self.runner, debug=debug, keep_dit_in_ram=cache_model_dit, keep_vae_in_ram=cache_model_vae)
            
            # Delete runner only if neither model is cached
            if not (cache_model_dit or cache_model_vae):
                del self.runner
                self.runner = None
                self._dit_model_name = ""
                self._vae_model_name = ""
        
        # Clean up context text embeddings if they exist
        if self.ctx:
            cleanup_text_embeddings(self.ctx, debug)
            self.ctx = None


    def _internal_execute(
        self, 
        pixels: torch.Tensor, 
        dit_model: str, 
        vae_model: str, 
        seed: int, 
        new_resolution: int, 
        cfg_scale: float, 
        batch_size: int,
        color_correction: str, 
        input_noise_scale: float, 
        latent_noise_scale: float, 
        encode_tiled: bool, 
        encode_tile_size: int, 
        encode_tile_overlap: int, 
        decode_tiled: bool, 
        decode_tile_size: int, 
        decode_tile_overlap: int, 
        preserve_vram: bool, 
        temporal_overlap: int, 
        cache_model_dit: bool, 
        cache_model_vae: bool, 
        dit_device: str, 
        vae_device: str,
        dit_torch_compile_args: Optional[Dict[str, Any]], 
        vae_torch_compile_args: Optional[Dict[str, Any]], 
        block_swap_config: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor]:
        """
        Internal execution logic with progress tracking and four-phase pipeline
        
        Implements the complete video upscaling pipeline:
        1. Model preparation and device setup
        2. Four-phase batch processing (encode → upscale → decode → postprocess)
        3. Progress tracking and memory management
        4. Cleanup and result finalization
        
        Args:
            pixels: Input video frames tensor (N, H, W, C)
            dit_model: DiT model filename
            vae_model: VAE model filename
            seed: Random seed for generation
            new_resolution: Target resolution for shortest edge
            cfg_scale: Classifier-free guidance scale (fixed at 1.0)
            batch_size: Number of frames per batch
            color_correction: Color correction method
            input_noise_scale: Input noise injection scale
            latent_noise_scale: Latent noise injection scale
            encode_tiled: Enable VAE tiled encoding
            encode_tile_size: VAE encoding tile size in pixels
            encode_tile_overlap: VAE encoding tile overlap in pixels
            decode_tiled: Enable VAE tiled decoding
            decode_tile_size: VAE decoding tile size in pixels
            decode_tile_overlap: VAE decoding tile overlap in pixels
            preserve_vram: Offload models between phases
            temporal_overlap: Overlap frames between batches (fixed at 0)
            cache_model_dit: Keep DiT model in RAM after use
            cache_model_vae: Keep VAE model in RAM after use
            dit_device: Device string for DiT model
            vae_device: Device string for VAE model
            dit_torch_compile_args: Optional torch.compile settings for DiT
            vae_torch_compile_args: Optional torch.compile settings for VAE
            block_swap_config: Optional BlockSwap configuration for DiT
            
        Returns:
            Tuple containing upscaled video tensor (N, H', W', C)
            
        Raises:
            RuntimeError: If any pipeline phase fails
            
        Pipeline Phases:
            Phase 1: VAE encoding - converts pixels to latent space
            Phase 2: DiT upscaling - diffusion-based upscaling in latent space
            Phase 3: VAE decoding - converts latents back to pixels
            Phase 4: Post-processing - color correction and final assembly
            
        Note:
            This method manages the complete lifecycle including model loading,
            inference, and cleanup based on caching preferences.
        """
        
        debug = self.debug
        
        # Initialize progress bar
        if ProgressBar is not None:
            self._pbar = ProgressBar(100)
        else:
            self._pbar = None

        # Setup device environment
        dit_device, vae_device = setup_device_environment(dit_device, vae_device, debug)

        # Create generation context with the configured device
        ctx = prepare_generation_context(dit_device=dit_device, vae_device=vae_device, debug=debug)
        self.ctx = ctx

        # Prepare runner with model state management
        self.runner, model_changed = prepare_runner(
            dit_model=dit_model, 
            vae_model=vae_model, 
            model_dir=get_base_cache_dir(),
            preserve_vram=preserve_vram,
            debug=debug,
            cache_model_dit=cache_model_dit,
            cache_model_vae=cache_model_vae,
            block_swap_config=block_swap_config,
            encode_tiled=encode_tiled,
            encode_tile_size=(encode_tile_size, encode_tile_size),
            encode_tile_overlap=(encode_tile_overlap, encode_tile_overlap),
            decode_tiled=decode_tiled,
            decode_tile_size=(decode_tile_size, decode_tile_size),
            decode_tile_overlap=(decode_tile_overlap, decode_tile_overlap),
            torch_compile_args_dit=dit_torch_compile_args,
            torch_compile_args_vae=vae_torch_compile_args,
            cached_runner=self.runner if (cache_model_dit or cache_model_vae) else None
        )
        self._dit_model_name = dit_model
        self._vae_model_name = vae_model

        debug.log_memory_state("After model preparation", show_tensors=False, detailed_tensors=False)
        debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

        debug.log("", category="none", force=True)
        debug.log("Starting video upscaling generation...", category="generation", force=True)
        debug.start_timer("generation")
       
        # Track total frames and resolutions
        total_frames = len(pixels)
        input_h, input_w = pixels.shape[1], pixels.shape[2]
        channels_info = "RGBA" if pixels.shape[-1] == 4 else "RGB"
        
        # Create transform pipeline early and apply to first frame to get exact output dimensions
        video_transform = prepare_video_transforms(new_resolution)
        
        # Apply transform to first frame to get exact output dimensions (reusing actual transform logic)
        sample_frame = pixels[0].permute(2, 0, 1)  # HWC -> CHW for transform
        sample_frame = sample_frame.unsqueeze(0)  # Add time dimension: CHW -> TCHW (T=1)
        transformed_sample = video_transform(sample_frame)  # Apply full pipeline
        # Transform outputs CTHW format, get dimensions from last two axes
        output_h, output_w = transformed_sample.shape[-2:]  # Get actual output size
        
        debug.log(f"   Total frames: {total_frames}, Input: {input_w}x{input_h}px → Output: {output_w}x{output_h}px, Batch size: {batch_size}, Channels: {channels_info}", category="generation", force=True)
        
        # Store transform in context for reuse during encoding (avoids recreating it)
        ctx['video_transform'] = video_transform
        
        # Phase 1: Encode all batches
        ctx = encode_all_batches(
            self.runner, 
            ctx=ctx, 
            images=pixels, 
            batch_size=batch_size, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback, 
            temporal_overlap=temporal_overlap, 
            res_w=new_resolution,
            input_noise_scale=input_noise_scale,
            color_correction=color_correction
        )

        # Phase 2: Upscale all batches
        ctx = upscale_all_batches(
            self.runner, 
            ctx=ctx, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback, 
            cfg_scale=cfg_scale, 
            seed=seed,
            latent_noise_scale=latent_noise_scale,
            cache_model=cache_model_dit
        )

        # Phase 3: Decode all batches
        ctx = decode_all_batches(
            self.runner, 
            ctx=ctx, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback,
            cache_model=cache_model_vae
        )

        # Phase 4: Post-processing and final assembly
        ctx = postprocess_all_batches(
            ctx=ctx,
            debug=debug,
            progress_callback=self._progress_callback,
            color_correction=color_correction
        )

        # Get final result - preserve channel format from input
        sample = ctx['final_video']
        
        # Ensure output is on CPU (ComfyUI expects CPU tensors)
        if torch.is_tensor(sample) and sample.is_cuda:
            sample = sample.cpu()

        debug.log("", category="none", force=True)
        debug.log("Video upscaling completed successfully!", category="generation", force=True)

        debug.end_timer("generation", "Video generation")

        # Final cleanup
        debug.start_timer("final_cleanup")
        self.cleanup(cache_model_dit=cache_model_dit, cache_model_vae=cache_model_vae, debug=debug)
        debug.end_timer("final_cleanup", "Final cleanup")

        # Log final memory state after ALL cleanup is done by the phases
        debug.log_memory_state("After all phases complete", show_tensors=False, detailed_tensors=False)
        
        # Final timing summary
        debug.log("", category="none")
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")
        child_times = {
            "Model preparation": debug.timer_durations.get("model_preparation", 0),
            "Video generation": debug.timer_durations.get("generation", 0),
            "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
        }
        # Add phase breakdowns as separate items (they'll appear at the same level)
        if "phase1_encoding" in debug.timer_durations:
            child_times["  Phase 1: VAE encoding"] = debug.timer_durations.get("phase1_encoding", 0)
        if "phase2_upscaling" in debug.timer_durations:
            child_times["  Phase 2: DiT upscaling"] = debug.timer_durations.get("phase2_upscaling", 0)
        if "phase3_decoding" in debug.timer_durations:
            child_times["  Phase 3: VAE decoding"] = debug.timer_durations.get("phase3_decoding", 0)
        if "phase4_postprocess" in debug.timer_durations:
            child_times["  Phase 4: Post-processing"] = debug.timer_durations.get("phase4_postprocess", 0)

        total_execution_time = debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
        
        # Calculate and log FPS
        if total_execution_time > 0:
            fps = total_frames / total_execution_time
            debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)
            
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")

        # Clear history for next run (do this last, after all logging)
        debug.clear_history()

        # Clear progress bar and context reference
        self._pbar = None
        self.ctx = None

        return (sample,)

    def _progress_callback(self, current_step: int, total_steps: int, 
                          current_batch_frames: int, phase_name: str = "") -> None:
        """
        Progress callback for generation phases
        
        Args:
            current_step: Current step number within the phase
            total_steps: Total steps in the current phase
            current_batch_frames: Number of frames in current batch
            phase_name: Name of the current phase (e.g., "Phase 1: Encoding")
        """
        
        if not hasattr(self, '_pbar') or self._pbar is None:
            return
        
        # Calculate overall progress across all phases
        # Phase weights: Encode=20%, Upscale=60%, Decode=20%
        phase_weights = {"Encoding": 0.2, "Upscaling": 0.6, "Decoding": 0.2}
        phase_offset = {"Encoding": 0.0, "Upscaling": 0.2, "Decoding": 0.8}
        
        # Extract the phase type from "Phase X: Type" format
        if ":" in phase_name:
            phase_key = phase_name.split(":")[1].strip()
        else:
            phase_key = "Upscaling"
        
        weight = phase_weights.get(phase_key, 1.0)
        offset = phase_offset.get(phase_key, 0.0)
        
        # Calculate weighted progress
        phase_progress = (current_step / total_steps) if total_steps > 0 else 0
        overall_progress = offset + (phase_progress * weight)
        
        # Update the progress bar with the overall progress
        progress_value = int(overall_progress * 100)
        self._pbar.update_absolute(progress_value, 100)

    def __del__(self):
        """Destructor"""
        try:
            # Store debug reference
            debug = self.debug if hasattr(self, 'debug') else None
            
            # Full cleanup
            if hasattr(self, 'cleanup'):
                self.cleanup(cache_model_dit=False, cache_model_vae=False, debug=debug)
            
            # Clear all remaining references
            for attr in ['runner', '_model_name', 'debug', 'ctx']:
                if hasattr(self, attr):
                    delattr(self, attr)
        except:
            pass


class SeedVR2LoadDiTModel:
    """
    Configure DiT (Diffusion Transformer) model loader with memory optimization
    
    Provides configuration for:
    - Model selection and device placement
    - BlockSwap memory optimization for limited VRAM
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_DIT configuration dictionary for main upscaler node
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        devices = get_device_list()
        return {
            "required": {
                "model": (get_available_dit_models(), {
                    "default": DEFAULT_DIT,
                    "tooltip": "DiT model for upscaling. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                }),
                "device": (devices, {
                    "default": devices[0],
                    "tooltip": "Device to use for DiT upscaling"
                }),
            },
            "optional": {
                "blocks_to_swap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 36,
                    "step": 1,
                    "tooltip": "BlockSwap: Number of transformer blocks to offload (0=disabled, 16=balanced, 32=max savings for 3b model, 36=max savings for 7b model)"
                }),
                "swap_io_components": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload embeddings/IO layers to CPU for additional VRAM savings"
                }),
                "cache_in_ram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep DiT model in RAM between runs for faster batch processing"
                }),
                "torch_compile_args": ("TORCH_COMPILE_ARGS", {
                    "tooltip": "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node"
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_TYPES = ("SEEDVR2_DIT",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure DiT model loading and memory optimization settings"
    
    def create_config(self, model: str, device: str, blocks_to_swap: int = 0, 
                     swap_io_components: bool = False, cache_in_ram: bool = False, 
                     torch_compile_args: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any]]:
        """
        Create DiT model configuration dictionary
        
        Args:
            model: DiT model filename (e.g., "seedvr2_ema_3b_fp16.safetensors")
            device: Target device for DiT model (e.g., "cuda:0", "cpu")
            blocks_to_swap: Number of transformer blocks to offload for BlockSwap (0=disabled)
            swap_io_components: Whether to offload input/output layers to CPU
            cache_in_ram: Whether to keep model in RAM between runs
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            Tuple containing configuration dictionary for SeedVR2 main node
        """
        config = {
            "model": model,
            "device": device,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "cache_in_ram": cache_in_ram,
            "torch_compile_args": torch_compile_args,
        }
        return (config,)


class SeedVR2LoadVAEModel:
    """
    Configure VAE (Variational Autoencoder) model loader with tiling support
    
    Provides configuration for:
    - Model selection and device placement
    - Tiled encoding/decoding for VRAM reduction
    - Tile size and overlap control
    - Model caching between runs
    - Optional torch.compile integration
    
    Returns:
        SEEDVR2_VAE configuration dictionary for main upscaler node
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        devices = get_device_list()
        return {
            "required": {
                "model": (get_available_vae_models(), {
                    "default": DEFAULT_VAE,
                    "tooltip": "VAE model for encoding/decoding. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                }),
                "device": (devices, {
                    "default": devices[0],
                    "tooltip": "Device to use for VAE processing"
                }),
            },
            "optional": {
                "encode_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled encoding to reduce VRAM during encoding"
                }),
                "encode_tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "step": 32,
                    "tooltip": "Size of encoding tiles in pixels. Smaller = less VRAM but more seams/artifacts and slower. Larger = more VRAM but better quality and faster."
                }),
                "encode_tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 32,
                    "tooltip": "Pixel overlap between encoding tiles to reduce visible seams. Higher = better blending but slower processing."
                }),
                "decode_tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled decoding to reduce VRAM during decoding"
                }),
                "decode_tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "step": 32,
                    "tooltip": "Size of decoding tiles in pixels. Smaller = less VRAM but more seams/artifacts and slower. Larger = more VRAM but better quality and faster."
                }),
                "decode_tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 32,
                    "tooltip": "Pixel overlap between decoding tiles to reduce visible seams. Higher = better blending but slower processing."
                }),
                "cache_in_ram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep VAE model in RAM between runs for faster batch processing"
                }),
                "torch_compile_args": ("TORCH_COMPILE_ARGS", {
                    "tooltip": "Optional torch.compile optimization settings from SeedVR2 Torch Compile Settings node"
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_TYPES = ("SEEDVR2_VAE",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure VAE settings and tiling options for memory optimization"
    
    def create_config(self, model: str, device: str, encode_tiled: bool = False, 
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64, 
                     decode_tiled: bool = False, decode_tile_size: int = 512,
                     decode_tile_overlap: int = 64, cache_in_ram: bool = False, 
                     torch_compile_args: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any]]:
        """
        Create VAE model configuration dictionary
        
        Args:
            model: VAE model filename (e.g., "ema_vae_fp16.safetensors")
            device: Target device for VAE model (e.g., "cuda:0", "cpu")
            encode_tiled: Enable tiled encoding to reduce VRAM usage
            encode_tile_size: Size of encoding tiles in pixels
            encode_tile_overlap: Overlap between encoding tiles to reduce seams
            decode_tiled: Enable tiled decoding to reduce VRAM usage
            decode_tile_size: Size of decoding tiles in pixels
            decode_tile_overlap: Overlap between decoding tiles to reduce seams
            cache_in_ram: Whether to keep model in RAM between runs
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            Tuple containing configuration dictionary for SeedVR2 main node
        """
        config = {
            "model": model,
            "device": device,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "cache_in_ram": cache_in_ram,
            "torch_compile_args": torch_compile_args,
        }
        return (config,)

class SeedVR2TorchCompileSettings:
    """Configure torch.compile optimization for DiT and VAE models"""
    
class SeedVR2TorchCompileSettings:
    """Configure torch.compile optimization for DiT and VAE models"""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "backend": (["inductor", "cudagraphs"], {
                    "default": "inductor",
                    "tooltip": (
                        "Compilation backend:\n"
                        "• inductor: Full optimization with Triton kernel generation and fusion\n"
                        "• cudagraphs: Lightweight, only wraps model with CUDA graphs, no kernel optimization"
                    )
                }),
                "mode": (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], {
                    "default": "default",
                    "tooltip": (
                        "Optimization level (compilation time vs runtime speed):\n"
                        "• default: Fast compile, good speedup, lowest memory\n"
                        "• reduce-overhead: Fast compile, better speedup with CUDA graphs, +10-20% memory\n"
                        "• max-autotune: Slow compile, best speedup, highest memory\n"
                        "• max-autotune-no-cudagraphs: Like max-autotune but without CUDA graphs (more compatible)"
                    )
                }),
                "fullgraph": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Compile entire model as single graph:\n"
                        "• False: Allows graph breaks, compiles what it can, more compatible\n"
                        "• True: Enforces no graph breaks, errors if any found, maximum optimization but fragile"
                    )
                }),
                "dynamic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Handle varying input shapes without recompilation:\n"
                        "• False: Specialized for exact shapes, recompiles if shape changes\n"
                        "• True: Creates dynamic kernels upfront, slower but handles shape variations"
                    )
                }),
                "dynamo_cache_size_limit": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "tooltip": (
                        "Max cached compiled versions per function (prevents memory bloat):\n"
                        "Higher value = more variations cached = more memory used\n"
                        "Lower value = more recompilation if inputs vary\n"
                        "Default 64 is good for most cases. Increase if you see cache_size_limit warnings"
                    )
                }),
            },
            "optional": {
                "dynamo_recompile_limit": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "tooltip": (
                        "Max recompilation attempts before giving up (falls back to no compilation):\n"
                        "Safety limit to prevent infinite recompilation loops\n"
                        "Default 128 is sufficient. Only increase if you see recompile_limit warnings"
                    )
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_TYPES = ("TORCH_COMPILE_ARGS",)
    FUNCTION = "create_args"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = (
        "Configure torch.compile optimization for DiT and VAE speedup. "
        "Connect to DiT and/or VAE model loader. Requires PyTorch 2.0+ and Triton."
    )
    
    def create_args(self, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int = 128) -> Tuple[Dict[str, Any]]:
        """
        Create torch.compile configuration for model optimization
        
        Args:
            backend: Compilation backend ("inductor" or "cudagraphs")
            mode: Optimization mode ("default", "reduce-overhead", "max-autotune", etc.)
            fullgraph: Whether to compile entire model as single graph
            dynamic: Whether to handle varying input shapes without recompilation
            dynamo_cache_size_limit: Maximum cached compiled versions per function
            dynamo_recompile_limit: Maximum recompilation attempts before fallback
            
        Returns:
            Tuple containing torch.compile configuration dictionary
        """
        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
        }
        return (compile_args,)
        
# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
    "SeedVR2LoadDiTModel": SeedVR2LoadDiTModel,
    "SeedVR2LoadVAEModel": SeedVR2LoadVAEModel,
    "SeedVR2TorchCompileSettings": SeedVR2TorchCompileSettings,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
    "SeedVR2LoadDiTModel": "SeedVR2 (Down)Load DiT Model",
    "SeedVR2LoadVAEModel": "SeedVR2 (Down)Load VAE Model",
    "SeedVR2TorchCompileSettings": "SeedVR2 Torch Compile Settings",
}

# Export version and metadata
__version__ = "2.0.0-modular"
__author__ = "SeedVR2 Team"
__description__ = "High-quality video upscaling using advanced diffusion models"

# Additional exports for introspection
__all__ = [
    'SeedVR2',
    'SeedVR2LoadDiTModel',
    'SeedVR2LoadVAEModel',
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS'
]
