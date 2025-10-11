"""
VAE RGBA Adapter - Dynamic channel adaptation for RGBA support

This module provides utilities to dynamically adapt the VAE's input and output
convolution layers to support RGBA (4-channel) processing while preserving
all pretrained intermediate layer weights and custom layer functionality.
"""

import torch
import torch.nn as nn
from typing import Optional, Any


def adapt_vae_for_rgba(vae: nn.Module, debug: Optional[Any] = None) -> nn.Module:
    """
    Adapt VAE encoder and decoder to handle 4 channels (RGBA) instead of 3 (RGB).
    
    This function modifies only the first and last convolution layers by expanding
    their weight tensors in-place:
    - Encoder conv_in: 3 → 4 input channels
    - Decoder conv_out: 3 → 4 output channels
    
    All intermediate layers and their pretrained weights remain unchanged.
    The 4th channel weights are initialized from the mean of RGB channels.
    
    IMPORTANT: This preserves the custom conv layer type (causal conv) and all
    its special functionality like memory_state support.
    
    Args:
        vae: The VAE model to adapt
        debug: Optional debug instance for logging
        
    Returns:
        Modified VAE with RGBA support
    """
    if debug:
        debug.log("Adapting VAE to RGBA mode", category="setup", force=True)
    
    # Adapt encoder conv_in: 3 → 4 input channels
    if hasattr(vae, 'encoder') and hasattr(vae.encoder, 'conv_in'):
        conv_in = vae.encoder.conv_in
        
        # Check if already adapted
        if conv_in.in_channels == 4:
            if debug:
                debug.log("Encoder conv_in already adapted for 4 channels", category="info")
        else:
            old_weight = conv_in.weight.data  # Shape: (out_ch, 3, k, k, k)
            
            # Create new weight tensor with 4 input channels
            # Shape: (out_ch, 4, k, k, k)
            new_weight = torch.zeros(
                (old_weight.shape[0], 4, *old_weight.shape[2:]),
                dtype=old_weight.dtype,
                device=old_weight.device
            )
            
            # Copy RGB weights
            new_weight[:, :3, ...] = old_weight
            
            # Initialize 4th channel (Alpha) as mean of RGB channels
            new_weight[:, 3:4, ...] = old_weight.mean(dim=1, keepdim=True)
            
            # Expand the conv layer in-place
            conv_in.in_channels = 4
            conv_in.weight = nn.Parameter(new_weight, requires_grad=conv_in.weight.requires_grad)
            
            if debug:
                debug.log("Adapted encoder conv_in: 3→4 channels", category="success")
    
    # Adapt decoder conv_out: 3 → 4 output channels  
    if hasattr(vae, 'decoder') and hasattr(vae.decoder, 'conv_out'):
        conv_out = vae.decoder.conv_out
        
        # Check if already adapted
        if conv_out.out_channels == 4:
            if debug:
                debug.log("Decoder conv_out already adapted for 4 channels", category="info")
        else:
            old_weight = conv_out.weight.data  # Shape: (3, in_ch, k, k, k)
            old_bias = conv_out.bias.data if conv_out.bias is not None else None
            
            # Create new weight tensor with 4 output channels
            # Shape: (4, in_ch, k, k, k)
            new_weight = torch.zeros(
                (4, *old_weight.shape[1:]),
                dtype=old_weight.dtype,
                device=old_weight.device
            )
            
            # Copy RGB weights
            new_weight[:3, ...] = old_weight
            
            # Initialize 4th channel (Alpha) output as mean of RGB channels
            new_weight[3:4, ...] = old_weight.mean(dim=0, keepdim=True)
            
            # Expand the conv layer in-place
            conv_out.out_channels = 4
            conv_out.weight = nn.Parameter(new_weight, requires_grad=conv_out.weight.requires_grad)
            
            # Expand bias if it exists
            if old_bias is not None:
                new_bias = torch.cat([
                    old_bias,
                    old_bias.mean().unsqueeze(0)  # Alpha bias as mean of RGB biases
                ])
                conv_out.bias = nn.Parameter(new_bias, requires_grad=conv_out.bias.requires_grad)
            
            if debug:
                debug.log("Adapted decoder conv_out: 3→4 channels", category="success")
    
    if debug:
        debug.log("Adapted VAE to RGBA mode", category="success")
    
    return vae


def restore_vae_rgb_only(vae: nn.Module, debug: Optional[Any] = None) -> nn.Module:
    """
    Restore VAE back to 3-channel (RGB) mode by keeping only RGB weights.
    
    This is useful if you want to cache the model in its original 3-channel form.
    
    Args:
        vae: The adapted VAE model to restore
        debug: Optional debug instance for logging
        
    Returns:
        VAE restored to 3-channel mode
    """
    if debug:
        debug.log("Restoring VAE to RGB-only mode", category="setup", force=True)
    
    # Restore encoder conv_in to 3 channels
    if hasattr(vae, 'encoder') and hasattr(vae.encoder, 'conv_in'):
        conv_in = vae.encoder.conv_in
        if conv_in.in_channels == 4:
            # Keep only first 3 channels
            new_weight = conv_in.weight.data[:, :3, ...].clone()
            
            conv_in.in_channels = 3
            conv_in.weight = nn.Parameter(new_weight, requires_grad=conv_in.weight.requires_grad)
            
            if debug:
                debug.log("Restored encoder conv_in to 3 channels", category="success")
    
    # Restore decoder conv_out to 3 channels
    if hasattr(vae, 'decoder') and hasattr(vae.decoder, 'conv_out'):
        conv_out = vae.decoder.conv_out
        if conv_out.out_channels == 4:
            # Keep only first 3 channels
            new_weight = conv_out.weight.data[:3, ...].clone()
            new_bias = conv_out.bias.data[:3].clone() if conv_out.bias is not None else None
            
            conv_out.out_channels = 3
            conv_out.weight = nn.Parameter(new_weight, requires_grad=conv_out.weight.requires_grad)
            
            if new_bias is not None:
                conv_out.bias = nn.Parameter(new_bias, requires_grad=conv_out.bias.requires_grad)
            
            if debug:
                debug.log("Restored decoder conv_out to 3 channels", category="success")
    
    if debug:
        debug.log("Restored VAE to RGB-only mode", category="success")
    
    return vae