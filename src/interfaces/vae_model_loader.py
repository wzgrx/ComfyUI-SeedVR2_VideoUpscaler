"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support
"""

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from typing import Dict, Any, Tuple
from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE
from ..optimization.memory_manager import get_device_list


class SeedVR2LoadVAEModel(io.ComfyNode):
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
    def define_schema(cls) -> io.Schema:        
        devices = get_device_list()
        vae_models = get_available_vae_models()
        
        return io.Schema(
            node_id="SeedVR2LoadVAEModel",
            display_name="SeedVR2 (Down)Load VAE Model",
            category="SEEDVR2",
            description=(
                "Configure VAE model for SeedVR2 encoding/decoding. Supports tiled processing for VRAM reduction, "
                "model caching, and torch.compile optimization. Connect output to SeedVR2 Video Upscaler."
            ),
            inputs=[
                io.Combo.Input("model",
                    options=vae_models,
                    default=DEFAULT_VAE,
                    tooltip="VAE model for encoding/decoding. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                ),
                io.Combo.Input("device",
                    options=devices,
                    default=devices[0],
                    tooltip="Device for VAE inference (encoding/decoding)"
                ),
                io.Boolean.Input("encode_tiled",
                    default=True,
                    optional=True,
                    tooltip="Enable tiled encoding (ON by default to prevent noise artifacts at high resolution). Disable only for low-resolution inputs to improve speed."
                ),
                io.Int.Input("encode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip="Encoding tile size in pixels (default: 1024). Can be reduced if running out of memory, but increasing above 1024 is not recommended as it causes noise artifacts in the latent."
                ),
                io.Int.Input("encode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip="Pixel overlap between encoding tiles to reduce visible seams (default: 128). Higher values improve blending at the cost of slower processing."
                ),
                io.Boolean.Input("decode_tiled",
                    default=False,
                    optional=True,
                    tooltip="Enable tiled decoding to reduce VRAM during decoding"
                ),
                io.Int.Input("decode_tile_size",
                    default=1024,
                    min=64,
                    step=32,
                    optional=True,
                    tooltip="Decoding tile size in pixels (default: 1024). Adjust based on available VRAM."
                ),
                io.Int.Input("decode_tile_overlap",
                    default=128,
                    min=0,
                    step=32,
                    optional=True,
                    tooltip="Pixel overlap between decoding tiles to reduce visible seams (default: 128). Higher values improve blending at the cost of slower processing."
                ),
                io.Combo.Input("offload_device",
                    options=get_device_list(include_none=True, include_cpu=True),
                    default="none",
                    optional=True,
                    tooltip="Device to offload VAE model to when not in use. Select 'none' to keep on inference device, or 'cpu' for CPU offload (slower but reduces VRAM usage). Required for BlockSwap."
                ),
                io.Boolean.Input("cache_model",
                    default=False,
                    optional=True,
                    tooltip="Keep VAE model loaded on offload_device between workflow runs. Useful for batch processing."
                ),
                io.Custom("TORCH_COMPILE_ARGS").Input("torch_compile_args",
                    optional=True,
                    tooltip="Optional torch.compile settings from SeedVR2 Torch Compile Settings node for speedup"
                ),
            ],
            outputs=[
                io.Custom("SEEDVR2_VAE").Output()
            ]
        )
    
    @classmethod
    def execute(cls, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, encode_tiled: bool = False,
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64,
                     decode_tiled: bool = False, decode_tile_size: int = 512, 
                     decode_tile_overlap: int = 64, torch_compile_args: Dict[str, Any] = None
                     ) -> io.NodeOutput:
        """
        Create VAE model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is invalid
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached VAE model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "torch_compile_args": torch_compile_args,
            "node_id": get_executing_context().node_id,
        }
        return io.NodeOutput(config)