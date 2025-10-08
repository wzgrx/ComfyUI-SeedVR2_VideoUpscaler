# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any

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
                "pixels": ("IMAGE", ),
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
    RETURN_NAMES = ("IMAGE", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, pixels: torch.Tensor, dit: Dict[str, Any], vae: Dict[str, Any], 
                seed: int, new_resolution: int = 1072, batch_size: int = 5,
                color_correction: str = "wavelet", input_noise_scale: float = 0.0, 
                latent_noise_scale: float = 0.0, preserve_vram: bool = False,
                enable_debug: bool = False) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling with progress reporting"""
        
        # Unpack DiT configuration
        dit_model = dit["model"]
        dit_device = dit["device"]
        blocks_to_swap = dit.get("blocks_to_swap", 0)
        offload_io_components = dit.get("offload_io_components", False)
        cache_model_dit = dit.get("cache_in_ram", False)
        
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
        
        # Create block_swap_config if blocks_to_swap > 0
        block_swap_config = None
        if blocks_to_swap > 0:
            block_swap_config = {
                "blocks_to_swap": blocks_to_swap,
                "offload_io_components": offload_io_components,
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
            return self._internal_execute(pixels, dit_model, vae_model, seed, new_resolution, cfg_scale, 
                                        batch_size, color_correction, input_noise_scale, latent_noise_scale, 
                                        encode_tiled, encode_tile_size, encode_tile_overlap, 
                                        decode_tiled, decode_tile_size, decode_tile_overlap, 
                                        preserve_vram, temporal_overlap, 
                                        cache_model_dit, cache_model_vae, dit_device, vae_device, block_swap_config)
        except Exception as e:
            self.cleanup(cache_model_dit=cache_model_dit, cache_model_vae=cache_model_vae, debug=self.debug)
            raise e
        

    def cleanup(self, cache_model_dit: bool = False, cache_model_vae: bool = False, debug=None):
        """
        Cleanup runner and free memory
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


    def _internal_execute(self, pixels, dit_model, vae_model, seed, new_resolution, cfg_scale, batch_size,
                color_correction, input_noise_scale, latent_noise_scale, encode_tiled, encode_tile_size, 
                encode_tile_overlap, decode_tiled, decode_tile_size, decode_tile_overlap, 
                preserve_vram, temporal_overlap, cache_model_dit, cache_model_vae, dit_device, vae_device, block_swap_config) -> Tuple[torch.Tensor]:
        """Internal execution logic with progress tracking"""
        
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
            cached_runner=self.runner if (cache_model_dit or cache_model_vae) else None
        )
        self._dit_model_name = dit_model
        self._vae_model_name = vae_model

        debug.log_memory_state("After model preparation", show_tensors=False, detailed_tensors=False)
        debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

        debug.log("", category="none", force=True)
        debug.log("Starting video upscaling generation...", category="generation", force=True)
        debug.start_timer("generation")
       
        # Track total frames for FPS calculation
        total_frames = len(pixels)
        debug.log(f"   Total frames: {total_frames}, New resolution: {new_resolution}, Batch size: {batch_size}", category="generation", force=True)
        
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

        # Get final result
        sample = ctx['final_video']

        # Ensure sample is on CPU (ComfyUI expects CPU tensors)
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

    def _progress_callback(self, current_step, total_steps, current_batch_frames, phase_name=""):
        """Progress callback for generation phases"""
        
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
    """Configure DiT model and memory optimization settings"""
    
    @classmethod
    def INPUT_TYPES(cls):
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
                "offload_io_components": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload embeddings/IO layers to CPU for additional VRAM savings"
                }),
                "cache_in_ram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep DiT model in RAM between runs for faster batch processing"
                }),
            }
        }
    
    RETURN_TYPES = ("SEEDVR2_DIT",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure DiT model loading and memory optimization settings"
    
    def create_config(self, model, device, blocks_to_swap=0, offload_io_components=False, cache_in_ram=False):
        config = {
            "model": model,
            "device": device,
            "blocks_to_swap": blocks_to_swap,
            "offload_io_components": offload_io_components,
            "cache_in_ram": cache_in_ram,
        }
        return (config,)


class SeedVR2LoadVAEModel:
    """Configure VAE settings and tiling options"""
    
    @classmethod
    def INPUT_TYPES(cls):
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
            }
        }
    
    RETURN_TYPES = ("SEEDVR2_VAE",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure VAE settings and tiling options for memory optimization"
    
    def create_config(self, model, device, encode_tiled=False, encode_tile_size=512, 
                     encode_tile_overlap=64, decode_tiled=False, decode_tile_size=512,
                     decode_tile_overlap=64, cache_in_ram=False):
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
        }
        return (config,)
        
# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
    "SeedVR2LoadDiTModel": SeedVR2LoadDiTModel,
    "SeedVR2LoadVAEModel": SeedVR2LoadVAEModel,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
    "SeedVR2LoadDiTModel": "SeedVR2 (Down)Load DiT Model",
    "SeedVR2LoadVAEModel": "SeedVR2 (Down)Load VAE Model",
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
