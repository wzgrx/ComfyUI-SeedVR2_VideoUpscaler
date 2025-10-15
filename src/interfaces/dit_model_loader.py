"""
SeedVR2 DiT Model Loader Node
Configure DiT (Diffusion Transformer) model with memory optimization
"""

from typing import Dict, Any, Tuple

from ..utils.model_registry import get_available_dit_models, DEFAULT_DIT
from ..optimization.memory_manager import get_device_list


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
                    "tooltip": "Device to use for DiT processing"
                }),
            },
            "optional": {
                "blocks_to_swap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 36,
                    "step": 1,
                    "tooltip": "Number of transformer blocks to swap to CPU. 0=disabled (fastest, most VRAM). Higher values save VRAM but are slower. 3B model: 0-32 blocks, 7B model: 0-36 blocks."
                }),
                "swap_io_components": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload input/output embeddings and norm layers to CPU for additional VRAM savings (slower)"
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
    
    RETURN_TYPES = ("SEEDVR2_DIT",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = (
        "Configure DiT model for SeedVR2 upscaling. Supports BlockSwap for limited VRAM, "
        "model caching, and torch.compile optimization. Connect output to SeedVR2 Video Upscaler."
    )
    
    def create_config(self, model: str, device: str, blocks_to_swap: int = 0, 
                     swap_io_components: bool = False, cache_in_ram: bool = False,
                     torch_compile_args: Dict[str, Any] = None, unique_id: int = 0
                     ) -> Tuple[Dict[str, Any]]:
        """
        Create DiT model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            blocks_to_swap: Number of transformer blocks to swap to CPU (0=disabled)
            swap_io_components: Whether to offload I/O components to CPU
            cache_in_ram: Whether to keep model in RAM between runs
            torch_compile_args: Optional torch.compile configuration from settings node
            unique_id: Node instance ID for caching
            
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
            "node_id": unique_id,
        }
        return (config,)