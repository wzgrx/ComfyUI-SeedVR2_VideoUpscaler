"""
SeedVR2 VAE Model Loader Node
Configure VAE (Variational Autoencoder) model with tiling support
"""

from typing import Dict, Any, Tuple

from ..utils.model_registry import get_available_vae_models, DEFAULT_VAE
from ..optimization.memory_manager import get_device_list


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
                    "tooltip": "Keep model in RAM between runs for faster reuse. Useful for batch processing."
                }),
                "torch_compile_args": ("TORCH_COMPILE_ARGS", {
                    "tooltip": "Optional torch.compile settings from SeedVR2 Torch Compile Settings node for speedup"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("SEEDVR2_VAE",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = (
        "Configure VAE model for SeedVR2 encoding/decoding. Supports tiled processing for VRAM reduction, "
        "model caching, and torch.compile optimization. Connect output to SeedVR2 Video Upscaler."
    )
    
    def create_config(self, model: str, device: str, encode_tiled: bool = False, 
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64,
                     decode_tiled: bool = False, decode_tile_size: int = 512, 
                     decode_tile_overlap: int = 64, cache_in_ram: bool = False,
                     torch_compile_args: Dict[str, Any] = None, unique_id: int = 0
                     ) -> Tuple[Dict[str, Any]]:
        """
        Create VAE model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            cache_in_ram: Whether to keep model in RAM between runs
            torch_compile_args: Optional torch.compile configuration from settings node
            unique_id: Node instance ID for caching
            
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
            "node_id": unique_id,
        }
        return (config,)