"""
GGUF Quantization Operations Module for SeedVR2

Provides runtime quantization support with proper debug logging and memory management
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple
from ..utils.debug import Debug

# Import GGUF with fallback
try:
    import gguf
    from .gguf_dequant import dequantize_tensor, is_torch_compatible, is_quantized as is_gguf_quantized
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None
    
    # Fallback functions when GGUF not available
    def is_torch_compatible(tensor: torch.Tensor) -> bool:
        return True
    
    def is_gguf_quantized(tensor: torch.Tensor) -> bool:
        return False
    
    def dequantize_tensor(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None, 
                     dequant_dtype: Optional[torch.dtype] = None, 
                     debug: Optional[Debug] = None) -> torch.Tensor:
        if dtype is not None:
            return tensor.to(dtype)
        return tensor


class _GGUFQuantizedBase(nn.Module):
    """Base class for GGUF quantized layers with shared dequantization logic"""
    
    def __init__(self, debug: Optional[Debug] = None):
        super().__init__()
        self.weight = None
        self.bias = None
        self.quantized_weight = None
        self.weight_qtype = None
        self.weight_shape = None
        self.debug = debug
    
    def load_quantized_weight(self, weight_tensor: torch.Tensor, bias_tensor: Optional[torch.Tensor] = None) -> None:
        """Load quantized weight tensor with debug logging"""
        if hasattr(weight_tensor, 'tensor_type') and hasattr(weight_tensor, 'tensor_shape'):
            # This is a quantized tensor
            self.quantized_weight = weight_tensor
            self.weight_qtype = weight_tensor.tensor_type
            self.weight_shape = weight_tensor.tensor_shape
        else:
            # Regular tensor
            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
        
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
    
    def dequantize_weight(self, device: Optional[str] = None, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize weight tensor on-the-fly with multiple fallback paths"""
        if self.quantized_weight is None:
            return self.weight
        
        # Start timer for any dequantization path
        timer_started = False
        if self.debug:
            self.debug.start_timer("gguf_dequant")
            timer_started = True
        
        try:
            # Check if already unquantized
            if GGUF_AVAILABLE and self.weight_qtype in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                return self.quantized_weight.to(device, dtype)
            
            # Primary path: Use gguf.quants.dequantize for GGUF tensors
            if GGUF_AVAILABLE and self.weight_qtype is not None and hasattr(gguf, 'quants'):
                try:
                    numpy_data = self.quantized_weight.cpu().numpy()
                    dequantized = gguf.quants.dequantize(numpy_data, self.weight_qtype)
                    result = torch.from_numpy(dequantized).to(device, dtype)
                    result.requires_grad_(False)
                    return result.reshape(self.weight_shape)
                except Exception as e:
                    if self.debug:
                        self.debug.log(f"GGUF dequantization failed, trying fallbacks: {e}", 
                                    level="WARNING", category="precision")
            
            # Fallback: Check for custom dequantization methods
            if hasattr(self.quantized_weight, 'gguf_dequantize'):
                return self.quantized_weight.gguf_dequantize(device, dtype)
            elif hasattr(self.quantized_weight, 'dequantize'):
                # PyTorch's dequantize() takes no arguments
                dequantized = self.quantized_weight.dequantize()
                return dequantized.to(device, dtype)
            
            # Try our dequantize_tensor function
            try:
                result = dequantize_tensor(self.quantized_weight, dtype=dtype, 
                                        dequant_dtype=dtype, debug=self.debug)
                if device:
                    result = result.to(device)
                return result
            except Exception as e:
                if self.debug:
                    self.debug.log(f"dequantize_tensor failed: {e}", 
                                level="WARNING", category="precision")
            
            # Last resort - return as-is but warn about potential issues
            if self.debug:
                self.debug.log(f"Returning quantized weight without dequantization - this may cause errors", 
                            level="ERROR", category="precision", force=True)
            return self.quantized_weight.to(device, dtype)
            
        finally:
            # Always end timer if it was started
            if timer_started and self.debug:
                self.debug.end_timer("gguf_dequant", "GGUF weight dequantization")


class GGUFQuantizedLinear(_GGUFQuantizedBase):
    """Quantized Linear layer with on-the-fly dequantization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, debug: Optional[Debug] = None):
        super().__init__(debug)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        dtype = input.dtype
        
        # Dequantize weight on-the-fly
        weight = self.dequantize_weight(device, dtype)
        
        # Handle shape mismatch for GGUF compressed weights
        if hasattr(self, 'weight_shape') and self.weight_shape != weight.shape:
            # The GGUF weight is compressed, we need to expand it
            target_shape = (self.out_features, self.in_features)
            if weight.shape[0] != target_shape[0] or weight.shape[1] != target_shape[1]:
                # Adjust weight shape - GGUF uses a compressed representation
                # The actual dequantization should restore the proper shape
                if self.debug:
                    self.debug.log(f"Adjusting weight shape from {weight.shape} to {target_shape}", 
                                category="precision")
                # For now, pad with zeros if needed (proper dequantization should handle this)
                if weight.shape[0] < target_shape[0]:
                    padding = torch.zeros((target_shape[0] - weight.shape[0], weight.shape[1]), 
                                        device=device, dtype=dtype)
                    weight = torch.cat([weight, padding], dim=0)
                if weight.shape[1] < target_shape[1]:
                    padding = torch.zeros((weight.shape[0], target_shape[1] - weight.shape[1]), 
                                        device=device, dtype=dtype)
                    weight = torch.cat([weight, padding], dim=1)
        
        # PyTorch linear expects weight to be [out_features, in_features]
        # but GGUF provides [in_features, out_features], so we need to transpose
        if len(weight.shape) == 2 and weight.shape != (self.out_features, self.in_features):
            weight = weight.transpose(0, 1)
        
        return torch.nn.functional.linear(input, weight, self.bias)


class GGUFQuantizedConv2d(_GGUFQuantizedBase):
    """Quantized Conv2d layer with on-the-fly dequantization"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None,
                 debug: Optional[Debug] = None):
        super().__init__(debug)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        dtype = input.dtype
        
        # Dequantize weight on-the-fly
        weight = self.dequantize_weight(device, dtype)
        
        # Standard conv2d operation
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride, 
                                         self.padding, self.dilation, self.groups)


def is_quantized_tensor(tensor) -> bool:
    """Check if a tensor is GGUF quantized"""
    if not GGUF_AVAILABLE:
        return False
    
    # Check multiple indicators
    if hasattr(tensor, 'tensor_type') and hasattr(tensor, 'tensor_shape'):
        return True
    if hasattr(tensor, '_gguf_quantized'):
        return True
    if 'GGUFTensor' in str(type(tensor)):
        return True
    return is_gguf_quantized(tensor) if callable(is_gguf_quantized) else False


def replace_linear_with_quantized(module, debug: Optional[Debug] = None, prefix=""):
    """Replace Linear and Conv2d layers with quantized versions if they have GGUF quantized weights"""
    if not GGUF_AVAILABLE:
        return 0
    
    replacements_made = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if is_quantized_tensor(child.weight):
                # Create quantized linear layer
                quantized_linear = GGUFQuantizedLinear(
                    child.in_features, 
                    child.out_features,
                    bias=child.bias is not None,
                    debug=debug
                )
                quantized_linear.load_quantized_weight(child.weight, child.bias)
                setattr(module, name, quantized_linear)
                replacements_made += 1
                
        elif isinstance(child, nn.Conv2d):
            if is_quantized_tensor(child.weight):
                # Create quantized conv2d layer
                quantized_conv = GGUFQuantizedConv2d(
                    child.in_channels,
                    child.out_channels, 
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    debug=debug
                )
                quantized_conv.load_quantized_weight(child.weight, child.bias)
                setattr(module, name, quantized_conv)
                replacements_made += 1
        else:
            # Recursively replace in child modules
            full_name = f"{prefix}.{name}" if prefix else name
            replacements_made += replace_linear_with_quantized(child, debug, full_name)
    
    return replacements_made