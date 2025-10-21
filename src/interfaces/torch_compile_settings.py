"""
SeedVR2 Torch Compile Settings Node
Configure torch.compile optimization for DiT and VAE models
"""

from comfy_api.latest import io
from typing import Dict, Any, Tuple


class SeedVR2TorchCompileSettings(io.ComfyNode):
    """Configure torch.compile optimization for DiT and VAE models"""
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeedVR2TorchCompileSettings",
            display_name="SeedVR2 Torch Compile Settings",
            category="SEEDVR2",
            description=(
                "Configure torch.compile optimization for DiT and VAE speedup. "
                "Connect to DiT and/or VAE model loader. Requires PyTorch 2.0+ and Triton."
            ),
            inputs=[
                io.Combo.Input("backend", 
                    options=["inductor", "cudagraphs"],
                    default="inductor",
                    tooltip=(
                        "Compilation backend:\n"
                        "• inductor: Full optimization with Triton kernel generation and fusion\n"
                        "• cudagraphs: Lightweight, only wraps model with CUDA graphs, no kernel optimization"
                    )
                ),
                io.Combo.Input("mode",
                    options=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    default="default",
                    tooltip=(
                        "Optimization level (compilation time vs runtime speed):\n"
                        "• default: Fast compilation, good speedup\n"
                        "• reduce-overhead: Lower overhead, better for smaller models\n"
                        "• max-autotune: Slowest compilation, best runtime (recommended for production)\n"
                        "• max-autotune-no-cudagraphs: Like max-autotune but without cudagraphs"
                    )
                ),
                io.Boolean.Input("fullgraph",
                    default=False,
                    tooltip="Compile entire model as single graph (faster but less flexible). May fail with dynamic shapes."
                ),
                io.Boolean.Input("dynamic",
                    default=False,
                    tooltip="Handle varying input shapes without recompilation. Useful for different resolutions/batch sizes."
                ),
                io.Int.Input("dynamo_cache_size_limit",
                    default=64,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip="Maximum cached compiled versions per function. Increase if using many different input shapes."
                ),
                io.Int.Input("dynamo_recompile_limit",
                    default=128,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip="Maximum recompilation attempts before fallback to eager mode. Only increase if you see recompile_limit warnings"
                ),
            ],
            outputs=[
                io.Custom("TORCH_COMPILE_ARGS").Output()
            ]
        )
    
    @classmethod
    def execute(cls, backend: str, mode: str, fullgraph: bool, dynamic: bool, 
                   dynamo_cache_size_limit: int, dynamo_recompile_limit: int) -> io.NodeOutput:
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
            NodeOutput containing torch.compile configuration dictionary
        """
        compile_args = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
        }
        return io.NodeOutput(compile_args)