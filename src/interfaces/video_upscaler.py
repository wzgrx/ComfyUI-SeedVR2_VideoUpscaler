"""
SeedVR2 Video Upscaler Node
Main ComfyUI node for high-quality video upscaling using diffusion models
"""

import torch
from typing import Tuple, Dict, Any, Optional

from ..utils.constants import get_base_cache_dir
from ..utils.downloads import download_weight
from ..utils.debug import Debug
from ..core.generation import (
    setup_device_environment,
    setup_video_transform,
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
    complete_cleanup
)

# Import ComfyUI progress reporting
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


class SeedVR2VideoUpscaler:
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
        """Initialize SeedVR2VideoUpscaler node"""
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
                    "default": 42,
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
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {
                    "default": "lab",
                    "tooltip": "Color correction method:\n• lab - Full perceptual color matching with detail preservation (recommended)\n• wavelet - Frequency-based transfer preserving high-frequency details\n• wavelet_adaptive - Wavelet base + targeted saturation correction for oversaturated regions\n• hsv - Hue-conditional saturation histogram matching\n• adain - Statistical style transfer (mean/std matching)\n• none - No color correction applied"
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
            dit: DiT model configuration from SeedVR2LoadDiTModel node
            vae: VAE model configuration from SeedVR2LoadVAEModel node
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
            RuntimeError: If generation fails
        """
        # Initialize debug instance
        self.debug = Debug(enabled=enable_debug)
        debug = self.debug
        
        # Extract configuration from dict inputs
        dit_model = dit["model"]
        vae_model = vae["model"]
        dit_device = dit["device"]
        vae_device = vae["device"]
        dit_cache = dit["cache_in_ram"]
        vae_cache = vae["cache_in_ram"]
        dit_id = dit.get("node_id")
        vae_id = vae.get("node_id")
        dit_torch_compile_args = dit.get("torch_compile_args")
        vae_torch_compile_args = vae.get("torch_compile_args")
        
        # Extract VAE tiling configuration
        encode_tiled = vae["encode_tiled"]
        encode_tile_size = vae["encode_tile_size"]
        encode_tile_overlap = vae["encode_tile_overlap"]
        decode_tiled = vae["decode_tiled"]
        decode_tile_size = vae["decode_tile_size"]
        decode_tile_overlap = vae["decode_tile_overlap"]
        
        # Extract DiT BlockSwap configuration
        block_swap_config = None
        if dit.get("blocks_to_swap", 0) > 0 or dit.get("swap_io_components", False):
            block_swap_config = {
                "blocks_to_swap": dit["blocks_to_swap"],
                "swap_io_components": dit["swap_io_components"],
            }
        
        # Fixed parameters
        temporal_overlap = 0
        
        # Intro logo
        debug.log("", category="none", force=True)
        debug.log(" ╔══════════════════════════════════════════════════════════╗", category="none", force=True)
        debug.log(" ║ ███████ ███████ ███████ ██████  ██    ██ ██████  ███████ ║", category="none", force=True)
        debug.log(" ║ ██      ██      ██      ██   ██ ██    ██ ██   ██      ██ ║", category="none", force=True)
        debug.log(" ║ ███████ █████   █████   ██   ██ ██    ██ ██████  █████   ║", category="none", force=True)
        debug.log(" ║      ██ ██      ██      ██   ██  ██  ██  ██   ██ ██      ║", category="none", force=True)
        debug.log(" ║ ███████ ███████ ███████ ██████    ████   ██   ██ ███████ ║", category="none", force=True)
        debug.log(" ║                         © ByteDance Seed · NumZ · AInVFX ║", category="none", force=True)
        debug.log(" ╚══════════════════════════════════════════════════════════╝", category="none", force=True)
        debug.log("", category="none", force=True)

        debug.start_timer("total_execution", force=True)

        debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        debug.log_memory_state("Before model preparation", show_tensors=False, detailed_tensors=False)
        debug.start_timer("model_preparation")

        # Check if download succeeded
        if not download_weight(dit_model=dit_model, vae_model=vae_model, debug=debug):
            raise RuntimeError(
                f"Failed to download required model files. "
                f"DiT model: {dit_model}, VAE model: {vae_model}. "
                "Please check the console output above for specific file failures and manual download instructions."
            )

        cfg_scale = 1.0
        try:
            return self._internal_execute(
                pixels=pixels,
                dit_model=dit_model,
                vae_model=vae_model,
                seed=seed,
                new_resolution=new_resolution,
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                color_correction=color_correction,
                input_noise_scale=input_noise_scale,
                latent_noise_scale=latent_noise_scale,
                encode_tiled=encode_tiled,
                encode_tile_size=encode_tile_size,
                encode_tile_overlap=encode_tile_overlap,
                decode_tiled=decode_tiled,
                decode_tile_size=decode_tile_size,
                decode_tile_overlap=decode_tile_overlap,
                preserve_vram=preserve_vram,
                temporal_overlap=temporal_overlap,
                dit_cache=dit_cache,
                vae_cache=vae_cache,
                dit_id=dit_id,
                vae_id=vae_id,
                dit_device=dit_device,
                vae_device=vae_device,
                dit_torch_compile_args=dit_torch_compile_args,
                vae_torch_compile_args=vae_torch_compile_args,
                block_swap_config=block_swap_config
            )
        except Exception as e:
            self.cleanup(dit_cache=dit_cache, vae_cache=vae_cache, debug=debug)
            raise e

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
        dit_cache: bool, 
        vae_cache: bool, 
        dit_id: int,
        vae_id: int,
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
            dit_cache: Keep DiT model in RAM after use
            vae_cache: Keep VAE model in RAM after use
            dit_id: Node instance ID for DiT model caching
            vae_id: Node instance ID for VAE model caching
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

        # Prepare runner with model state management and global cache
        self.runner, model_changed, cache_context = prepare_runner(
            dit_model=dit_model, 
            vae_model=vae_model, 
            model_dir=get_base_cache_dir(),
            preserve_vram=preserve_vram,
            debug=debug,
            dit_cache=dit_cache,
            vae_cache=vae_cache,
            dit_id=dit_id,
            vae_id=vae_id,
            block_swap_config=block_swap_config,
            encode_tiled=encode_tiled,
            encode_tile_size=(encode_tile_size, encode_tile_size),
            encode_tile_overlap=(encode_tile_overlap, encode_tile_overlap),
            decode_tiled=decode_tiled,
            decode_tile_size=(decode_tile_size, decode_tile_size),
            decode_tile_overlap=(decode_tile_overlap, decode_tile_overlap),
            torch_compile_args_dit=dit_torch_compile_args,
            torch_compile_args_vae=vae_torch_compile_args
        )
        self._dit_model_name = dit_model
        self._vae_model_name = vae_model
        
        # Store cache context in ctx for use in generation phases
        ctx['cache_context'] = cache_context

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
        setup_video_transform(ctx, new_resolution, debug)

        # Apply transform to first frame to get exact output dimensions (reusing actual transform logic)
        sample_frame = pixels[0].permute(2, 0, 1)  # HWC -> CHW for transform
        sample_frame = sample_frame.unsqueeze(0)  # Add time dimension: CHW -> TCHW (T=1)
        transformed_sample = ctx['video_transform'](sample_frame)  # Apply full pipeline
        # Transform outputs CTHW format, get dimensions from last two axes
        output_h, output_w = transformed_sample.shape[-2:]  # Get actual output size
        
        debug.log(f"  Total frames: {total_frames}, Input: {input_w}x{input_h}px → Output: {output_w}x{output_h}px, Batch size: {batch_size}, Channels: {channels_info}", category="generation", force=True)

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
            cache_model=dit_cache
        )

        # Phase 3: Decode all batches
        ctx = decode_all_batches(
            self.runner, 
            ctx=ctx, 
            preserve_vram=preserve_vram,
            debug=debug, 
            progress_callback=self._progress_callback,
            cache_model=vae_cache
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
        
        # Convert to float32 for maximum ComfyUI compatibility
        if torch.is_tensor(sample) and sample.dtype != torch.float32:
            sample = sample.to(torch.float32)
        
        # Ensure output is on CPU (ComfyUI expects CPU tensors)
        if torch.is_tensor(sample) and sample.is_cuda:
            sample = sample.cpu()

        debug.log("", category="none", force=True)
        debug.log("Video upscaling completed successfully!", category="success", force=True)

        debug.end_timer("generation", "Video generation")

        # Final cleanup
        debug.start_timer("final_cleanup")
        self.cleanup(dit_cache=dit_cache, vae_cache=vae_cache, debug=debug)
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
        
        # Footer with support info
        debug.log("", category="none", force=True)
        debug.log("━━━━━━━━━━━━━━━━━━", category="none", force=True)
        debug.log("Questions? Updates? Watch the videos, star the repo & join us!", category="dialogue", force=True)
        debug.log("https://www.youtube.com/@AInVFX", category="generation", force=True)
        debug.log("https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler", category="star", force=True)

        # Clear history for next run (do this last, after all logging)
        debug.clear_history()

        # Clear progress bar and context reference
        self._pbar = None
        self.ctx = None

        return (sample,)

    def _progress_callback(self, current_step: int, total_steps: int, 
                          current_frames: int, phase_name: str) -> None:
        """
        Update progress bar based on pipeline phase
        
        Args:
            current_step: Current step within phase
            total_steps: Total steps in phase
            current_frames: Number of frames being processed
            phase_name: Name of current phase
        """
        if self._pbar is None:
            return
        
        # Define phase weights and offsets for overall progress
        phase_weights = {
            "Phase 1: Encoding": 0.25,
            "Phase 2: Upscaling": 0.50,
            "Phase 3: Decoding": 0.20,
            "Phase 4: Post-processing": 0.05
        }
        
        phase_offset = {
            "Phase 1: Encoding": 0.0,
            "Phase 2: Upscaling": 0.25,
            "Phase 3: Decoding": 0.75,
            "Phase 4: Post-processing": 0.95
        }
        
        # Extract phase key from phase_name
        phase_key = phase_name.split(" (")[0] if " (" in phase_name else phase_name
        
        # Get weight and offset
        weight = phase_weights.get(phase_key, 1.0)
        offset = phase_offset.get(phase_key, 0.0)
        
        # Calculate weighted progress
        phase_progress = (current_step / total_steps) if total_steps > 0 else 0
        overall_progress = offset + (phase_progress * weight)
        
        # Update the progress bar with the overall progress
        progress_value = int(overall_progress * 100)
        self._pbar.update_absolute(progress_value, 100)

    def cleanup(self, dit_cache: bool = False, vae_cache: bool = False, 
               debug: Optional[Debug] = None) -> None:
        """
        Cleanup resources after upscaling.
        
        Args:
            dit_cache: If True, keep DiT model in RAM for reuse
            vae_cache: If True, keep VAE model in RAM for reuse
            debug: Debug instance (uses self.debug if not provided)
        """
        # Use existing debug from runner if not provided
        if debug is None and self.runner and hasattr(self.runner, 'debug'):
            debug = self.runner.debug
        
        # Use complete_cleanup for all cleanup operations
        if self.runner:
            complete_cleanup(runner=self.runner, debug=debug, keep_dit_in_ram=dit_cache, keep_vae_in_ram=vae_cache)
            
            # Delete runner only if neither model is cached
            if not (dit_cache or vae_cache):
                del self.runner
                self.runner = None
                self._dit_model_name = ""
                self._vae_model_name = ""
        
        # Clean up context text embeddings if they exist
        if self.ctx:
            cleanup_text_embeddings(self.ctx, debug)
            self.ctx = None

    def __del__(self):
        """Destructor"""
        try:
            # Store debug reference
            debug = self.debug if hasattr(self, 'debug') else None
            
            # Full cleanup
            if hasattr(self, 'cleanup'):
                self.cleanup(dit_cache=False, vae_cache=False, debug=debug)
            
            # Clear all remaining references
            for attr in ['runner', '_model_name', 'debug', 'ctx']:
                if hasattr(self, attr):
                    delattr(self, attr)
        except:
            pass