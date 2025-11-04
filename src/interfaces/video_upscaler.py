"""
SeedVR2 Video Upscaler Node
Main ComfyUI node for high-quality video upscaling using diffusion models
"""

import torch
from comfy_api.latest import io
from typing import Tuple, Dict, Any, Optional
from ..utils.constants import get_base_cache_dir
from ..utils.downloads import download_weight
from ..utils.debug import Debug
from ..core.generation_phases import (
    encode_all_batches, 
    upscale_all_batches, 
    decode_all_batches,
    postprocess_all_batches
)
from ..core.generation_utils import (
    setup_video_transform,
    setup_generation_context, 
    prepare_runner,
    prepare_video_transforms,
    prepend_video_frames
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


class SeedVR2VideoUpscaler(io.ComfyNode):
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """

    @classmethod
    def define_schema(cls) -> io.Schema:        
        return io.Schema(
            node_id="SeedVR2VideoUpscaler",
            display_name="SeedVR2 Video Upscaler",
            category="SEEDVR2",
            description=(
                "High-quality video upscaling using diffusion models. "
                "Supports RGB and RGBA formats, temporal consistency, and adaptive VRAM management."
            ),
            inputs=[
                io.Image.Input("image",
                    tooltip="Input video frames. Accepts both RGB (3-channel) and RGBA (4-channel) images. Output will match input format."
                ),
                io.Custom("SEEDVR2_DIT").Input("dit",
                    tooltip="DiT model configuration from SeedVR2 Load DiT Model node"
                ),
                io.Custom("SEEDVR2_VAE").Input("vae",
                    tooltip="VAE model configuration from SeedVR2 Load VAE Model node"
                ),
                io.Int.Input("seed",
                    default=42,
                    min=0,
                    max=2**32 - 1,
                    step=1,
                    tooltip="Random seed for generation. Same seed = same output."
                ),
                io.Int.Input("new_resolution",
                    default=1080,
                    min=16,
                    max=16384,
                    step=2,
                    tooltip="Target resolution for the shortest edge. Maintains aspect ratio."
                ),
                io.Int.Input("max_resolution",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    tooltip="Maximum resolution for any edge. If any dimension exceeds this after new_resolution is applied, both dimensions are scaled down proportionally. 0 = no limit (default)."
                ),
                io.Int.Input("batch_size",
                    default=5,
                    min=1,
                    max=16384,
                    step=4,
                    tooltip="Frames processed per batch. Minimum 5 for temporal consistency. Higher = better quality but more VRAM."
                ),
                io.Int.Input("temporal_overlap",
                    default=0,
                    min=0,
                    max=16,
                    step=1,
                    optional=True,
                    tooltip="Overlapping frames between batches for smoother transitions. 0 = disabled. Values 1-4 work well for temporal consistency."
                ),
                io.Int.Input("prepend_frames",
                    default=0,
                    min=0,
                    max=32,
                    step=1,
                    optional=True,
                    tooltip="Number of frames to prepend to the video (reversed from start). This can help with artifacts at the start of the video and are automatically removed after processing."
                ),
                io.Combo.Input("color_correction",
                    options=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    default="lab",
                    tooltip="Color correction method: 'lab' (full perceptual color matching with detail preservation, recommended), 'wavelet' (frequency-based natural colors, preserves details), 'wavelet_adaptive' (wavelet base + targeted saturation correction), 'hsv' (hue-conditional saturation matching), 'adain' (statistical style transfer), 'none' (no correction)"
                ),
                io.Float.Input("input_noise_scale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip="Input noise injection scale [0.0-1.0]. Adds variation to input. 0.0 = disabled."
                ),
                io.Float.Input("latent_noise_scale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip="Latent noise injection scale [0.0-1.0]. Adds variation to latent space. 0.0 = disabled."
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="cpu",
                    optional=True,
                    tooltip="Device to offload intermediate tensors. 'cpu' prevents VRAM accumulation for long videos. 'none' keeps all tensors on inference device (faster but increases VRAM usage)."
                ),
                io.Boolean.Input("enable_debug",
                    default=False,
                    optional=True,
                    tooltip="Show detailed memory usage and timing information during generation, useful for troubleshooting."
                ),
            ],
            outputs=[
                io.Image.Output()
            ]
        )
    
    @classmethod
    def execute(cls, image: torch.Tensor, dit: Dict[str, Any], vae: Dict[str, Any], 
                seed: int, new_resolution: int = 1080, max_resolution: int = 0, batch_size: int = 5,
                temporal_overlap: int = 0, prepend_frames: int = 0,
                color_correction: str = "wavelet", input_noise_scale: float = 0.0,
                latent_noise_scale: float = 0.0, offload_device: str = "none", 
                enable_debug: bool = False) -> io.NodeOutput:
        """
        Execute SeedVR2 video upscaling with progress reporting
        
        Main entry point for ComfyUI node execution.
        Automatically detects and preserves input format (RGB or RGBA). Handles model downloads,
        configuration unpacking, and delegates to upscaling pipeline.
        
        Args:
            image: Input video frames as tensor (N, H, W, C) in [0, 1] range
            dit: DiT model configuration from SeedVR2LoadDiTModel node
            vae: VAE model configuration from SeedVR2LoadVAEModel node
            seed: Random seed for reproducible generation
            new_resolution: Target resolution for shortest edge (maintains aspect ratio)
            max_resolution: Maximum resolution for any edge (0 = no limit)
            batch_size: Frames per batch (minimum 5 for temporal consistency)
            temporal_overlap: Overlapping frames between batches (0-16)
            prepend_frames: Frames to prepend (0-32) to reduce initial artifacts.
            color_correction: Color correction method
            input_noise_scale: Input noise injection scale [0.0-1.0]
            latent_noise_scale: Latent noise injection scale [0.0-1.0]
            offload_device: Device to offload intermediate tensors
            enable_debug: Enable detailed logging and memory tracking
            
        Returns:
            NodeOutput containing upscaled video tensor (N, H', W', C) in [0, 1] range
            
        Raises:
            ValueError: If model files cannot be downloaded or configuration is invalid
            RuntimeError: If generation fails
        """
        # Initialize debug (stateless - stored in local variable)
        debug = Debug(enabled=enable_debug)
        
        # Track execution state in local variables (not instance)
        runner = None
        ctx = None
        pbar = None
        
        # Define progress callback as local closure
        def progress_callback(current_step: int, total_steps: int, 
                            current_frames: int, phase_name: str) -> None:
            """
            Update progress bar based on pipeline phase
            
            Args:
                current_step: Current step within phase
                total_steps: Total steps in phase
                current_frames: Number of frames being processed
                phase_name: Name of current phase
            """
            if pbar is None:
                return
            
            phase_weights = {
                "Phase 1: Encoding": 0.2,
                "Phase 2: Upscaling": 0.25,
                "Phase 3: Decoding": 0.5,
                "Phase 4: Post-processing": 0.05
            }
            phase_offset = {
                "Phase 1: Encoding": 0.0,
                "Phase 2: Upscaling": 0.2,
                "Phase 3: Decoding": 0.45,
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
            pbar.update_absolute(progress_value, 100)
        
        # Define cleanup as local function
        def cleanup(dit_cache: bool = False, vae_cache: bool = False) -> None:
            """Cleanup resources after upscaling"""
            nonlocal runner, ctx
            
            # Use complete_cleanup for all cleanup operations
            if runner:
                complete_cleanup(runner=runner, debug=debug, 
                               dit_cache=dit_cache, vae_cache=vae_cache)
                
                # Delete runner only if neither model is cached
                if not (dit_cache or vae_cache):
                    runner = None
            
            # Clean up context text embeddings if they exist
            if ctx:
                cleanup_text_embeddings(ctx, debug)
                ctx = None
        
        # Extract configuration from dict inputs
        dit_model = dit["model"]
        vae_model = vae["model"]
        dit_device = torch.device(dit["device"])
        vae_device = torch.device(vae["device"])
        dit_id = dit["node_id"]
        vae_id = vae["node_id"]

        # OPTIONAL inputs - use .get() with defaults
        dit_cache = dit.get("cache_model", False)
        attention_mode = dit.get("attention_mode", "sdpa")
        vae_cache = vae.get("cache_model", False)

        # BlockSwap configuration - construct from individual values
        blocks_to_swap = dit.get("blocks_to_swap", 0)
        swap_io_components = dit.get("swap_io_components", False)
        dit_offload_str = dit.get("offload_device", "none")

        block_swap_config = None
        if blocks_to_swap > 0 or swap_io_components:
            # Convert offload device string to torch.device for BlockSwap
            if dit_offload_str != "none":
                block_swap_config = {
                    "blocks_to_swap": blocks_to_swap,
                    "swap_io_components": swap_io_components,
                    "offload_device": torch.device(dit_offload_str)
                }

        # Device configuration for offloading - convert "none" to None, else torch.device
        vae_offload_str = vae.get("offload_device", "none")
        dit_offload_device = torch.device(dit_offload_str) if dit_offload_str != "none" else None
        vae_offload_device = torch.device(vae_offload_str) if vae_offload_str != "none" else None
        tensor_offload_device = torch.device(offload_device) if offload_device != "none" else None

        # VAE tiling configuration
        encode_tiled = vae.get("encode_tiled", False)
        encode_tile_size = vae.get("encode_tile_size", 512)
        encode_tile_overlap = vae.get("encode_tile_overlap", 64)
        decode_tiled = vae.get("decode_tiled", False)
        decode_tile_size = vae.get("decode_tile_size", 512)
        decode_tile_overlap = vae.get("decode_tile_overlap", 64)
        tile_debug = vae.get("tile_debug", False)

        # TorchCompile args (optional connection, can be None)
        dit_torch_compile_args = dit.get("torch_compile_args")
        vae_torch_compile_args = vae.get("torch_compile_args")
        
        # Print header
        debug.print_header()

        debug.start_timer("total_execution", force=True)

        debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        debug.log_memory_state("Before model preparation", show_tensors=False, detailed_tensors=False)
        debug.start_timer("model_preparation")

        # Check if download succeeded
        debug.log("Checking and downloading models if needed...", category="download")
        if not download_weight(dit_model=dit_model, vae_model=vae_model, debug=debug):
            raise RuntimeError(
                f"Failed to download required model files. "
                f"DiT model: {dit_model}, VAE model: {vae_model}. "
                "Please check the console output above for specific file failures and manual download instructions."
            )
        
        try:
            # Initialize ComfyUI progress bar if available
            if ProgressBar is not None:
                pbar = ProgressBar(100)
            
            # Setup generation context with device configuration
            ctx = setup_generation_context(
                dit_device=dit_device,
                vae_device=vae_device,
                dit_offload_device=dit_offload_device,
                vae_offload_device=vae_offload_device,
                tensor_offload_device=tensor_offload_device,
                debug=debug
            )

            # Prepare runner with model state management and global cache
            runner, cache_context = prepare_runner(
                dit_model=dit_model, 
                vae_model=vae_model, 
                model_dir=get_base_cache_dir(),
                debug=debug,
                ctx=ctx,
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
                tile_debug=tile_debug,
                attention_mode=attention_mode,
                torch_compile_args_dit=dit_torch_compile_args,
                torch_compile_args_vae=vae_torch_compile_args
            )

            # Store cache context in ctx for use in generation phases
            ctx['cache_context'] = cache_context

            debug.log_memory_state("After model preparation", show_tensors=False, detailed_tensors=False)
            debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

            debug.log("", category="none", force=True)
            debug.log("Starting upscaling generation...", category="generation", force=True)
            debug.start_timer("generation")
           
            # Track input frames before any modifications
            input_frames = len(image)
            input_h, input_w = image.shape[1], image.shape[2]
            channels_info = "RGBA" if image.shape[-1] == 4 else "RGB"
            
            # Handle prepend_frames if requested
            if prepend_frames > 0:
                image = prepend_video_frames(image, prepend_frames, debug)
            
            # Track total frames after prepending
            total_frames = len(image)
            ctx['total_frames'] = total_frames
            
            # Setup transform and compute dimensions
            sample_frame = image[0].permute(2, 0, 1).unsqueeze(0)
            true_h, true_w, padded_h, padded_w = setup_video_transform(ctx, new_resolution, max_resolution, debug, sample_frame)
            
            # Build concise parameter info
            params_info = f"Batch size: {batch_size}"
            if prepend_frames > 0:
                params_info += f", Prepend frames: {prepend_frames}"
            if temporal_overlap > 0:
                params_info += f", Temporal overlap: {temporal_overlap}"
            params_info += f", Seed: {seed}, Channels: {channels_info}"
            
            # Build resolution constraint info
            res_constraint = f"shortest edge: {new_resolution}px"
            if max_resolution > 0:
                res_constraint += f", max edge: {max_resolution}px"
            
            # Log dimension flow with full context
            if true_h > 0:
                frame_text = "frame" if input_frames <= 1 else "frames"
                if true_h == padded_h and true_w == padded_w:
                    debug.log(f"Input: {input_frames} {frame_text}, {input_w}x{input_h}px → Output: {true_w}x{true_h}px ({res_constraint})", 
                             category="generation", force=True, indent_level=1)
                else:
                    debug.log(f"Input: {input_frames} {frame_text}, {input_w}x{input_h}px → Padded: {padded_w}x{padded_h}px → Output: {true_w}x{true_h}px ({res_constraint})", 
                             category="generation", force=True, indent_level=1)
            
            debug.log(f"{params_info}", category="generation", force=True, indent_level=1)
            del sample_frame
            
            # Phase 1: Encode
            ctx = encode_all_batches(
                runner,
                ctx=ctx,
                images=image,
                debug=debug,
                batch_size=batch_size,
                seed=seed,
                progress_callback=progress_callback,
                temporal_overlap=temporal_overlap,
                resolution=new_resolution,
                max_resolution=max_resolution,
                input_noise_scale=input_noise_scale,
                color_correction=color_correction
            )

            # Phase 2: Upscale
            ctx = upscale_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                seed=seed,
                latent_noise_scale=latent_noise_scale,
                cache_model=dit_cache
            )

            # Phase 3: Decode
            ctx = decode_all_batches(
                runner,
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                cache_model=vae_cache
            )

            # Phase 4: Post-processing
            ctx = postprocess_all_batches(
                ctx=ctx,
                debug=debug,
                progress_callback=progress_callback,
                color_correction=color_correction,
                prepend_frames=prepend_frames,
                temporal_overlap=temporal_overlap,
                batch_size=batch_size
            )

            sample = ctx['final_video']
            
            # Ensure CPU tensor in float32 for maximum ComfyUI compatibility
            if torch.is_tensor(sample):
                if sample.is_cuda or sample.is_mps:
                    sample = sample.cpu()
                if sample.dtype != torch.float32:
                    sample = sample.to(torch.float32)

            debug.log("", category="none", force=True)
            debug.log("Upscaling completed successfully!", category="success", force=True)
            debug.end_timer("generation", "Video generation")

            # Final cleanup
            debug.start_timer("final_cleanup")
            cleanup(dit_cache=dit_cache, vae_cache=vae_cache)
            debug.end_timer("final_cleanup", "Final cleanup")

            debug.log_memory_state("After all phases complete", show_tensors=False, detailed_tensors=False)
            
            # Final timing summary
            debug.log("", category="none")
            debug.log("────────────────────────", category="none")
            child_times = {
                "Model preparation": debug.timer_durations.get("model_preparation", 0),
                "Video generation": debug.timer_durations.get("generation", 0),
                "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
            }
            if "phase1_encoding" in debug.timer_durations:
                child_times["  Phase 1: VAE encoding"] = debug.timer_durations.get("phase1_encoding", 0)
            if "phase2_upscaling" in debug.timer_durations:
                child_times["  Phase 2: DiT upscaling"] = debug.timer_durations.get("phase2_upscaling", 0)
            if "phase3_decoding" in debug.timer_durations:
                child_times["  Phase 3: VAE decoding"] = debug.timer_durations.get("phase3_decoding", 0)
            if "phase4_postprocess" in debug.timer_durations:
                child_times["  Phase 4: Post-processing"] = debug.timer_durations.get("phase4_postprocess", 0)

            total_execution_time = debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
            
            if total_execution_time > 0:
                fps = total_frames / total_execution_time
                debug.log(f"Average FPS: {fps:.2f} frames/sec", category="timing", force=True)

            # Print footer
            debug.print_footer()

            debug.clear_history()
            pbar = None
            ctx = None

            # V3-compatible return with optional UI preview
            return io.NodeOutput(sample)
            
        except Exception as e:
            cleanup(dit_cache=dit_cache, vae_cache=vae_cache)
            raise e