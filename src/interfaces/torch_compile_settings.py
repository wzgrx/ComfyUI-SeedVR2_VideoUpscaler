"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from typing import Dict, Any, Tuple


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
                        "• default: Fast compilation, good speedup\n"
                        "• reduce-overhead: Lower overhead, better for smaller models\n"
                        "• max-autotune: Slowest compilation, best runtime (recommended for production)\n"
                        "• max-autotune-no-cudagraphs: Like max-autotune but without cudagraphs"
                    )
                }),
                "fullgraph": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compile entire model as single graph (faster but less flexible). May fail with dynamic shapes."
                }),
                "dynamic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Handle varying input shapes without recompilation. Useful for different resolutions/batch sizes."
                }),
                "dynamo_cache_size_limit": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Maximum cached compiled versions per function. Increase if using many different input shapes."
                }),
                "dynamo_recompile_limit": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Maximum recompilation attempts before fallback to eager mode. Only increase if you see recompile_limit warnings"
                }),
            }
        }
    
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