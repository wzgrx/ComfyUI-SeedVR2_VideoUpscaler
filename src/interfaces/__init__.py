"""
SeedVR2 ComfyUI Nodes
Central registry for all SeedVR2 nodes
"""

from .video_upscaler import SeedVR2VideoUpscaler
from .dit_model_loader import SeedVR2LoadDiTModel
from .vae_model_loader import SeedVR2LoadVAEModel
from .torch_compile_settings import SeedVR2TorchCompileSettings

# ComfyUI Node Mappings -
NODE_CLASS_MAPPINGS = {
    "SeedVR2VideoUpscaler": SeedVR2VideoUpscaler,
    "SeedVR2LoadDiTModel": SeedVR2LoadDiTModel,
    "SeedVR2LoadVAEModel": SeedVR2LoadVAEModel,
    "SeedVR2TorchCompileSettings": SeedVR2TorchCompileSettings,
}

# Human-readable node display names - unchanged
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2VideoUpscaler": "SeedVR2 Video Upscaler",
    "SeedVR2LoadDiTModel": "SeedVR2 (Down)Load DiT Model",
    "SeedVR2LoadVAEModel": "SeedVR2 (Down)Load VAE Model",
    "SeedVR2TorchCompileSettings": "SeedVR2 Torch Compile Settings",
}

__version__ = "2.0.0"
__author__ = "numz, adrientoupet"
__description__ = "ComfyUI integration of ByteDance-Seed's SeedVR2: One-step diffusion-based video/image upscaling with memory-efficient inference"

__all__ = [
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
    'SeedVR2LoadVAEModel',
    'SeedVR2TorchCompileSettings',
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS'
]