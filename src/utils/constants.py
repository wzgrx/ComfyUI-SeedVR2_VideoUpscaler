"""
Shared constants and utilities for SeedVR2
Only includes constants actually used in the codebase
"""

import os
import warnings
import inspect
from typing import Optional

# Model folder names
SEEDVR2_FOLDER_NAME = "SEEDVR2" # Physical folder name on disk
SEEDVR2_MODEL_TYPE = "seedvr2" # Model type identifier for ComfyUI

# Supported model file formats
SUPPORTED_MODEL_EXTENSIONS = {'.safetensors', '.gguf'}

# GGUF Quantization Constants
QK_K = 256
K_SCALE_SIZE = 12
GGUF_BLOCK_SIZE = 32
GGUF_TYPE_SIZE = 64

# Download configuration
HUGGINGFACE_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
DOWNLOAD_CHUNK_SIZE = 8192 * 1024  # 8MB chunks for hash calculation
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 2  # seconds

def get_script_directory() -> str:
    """Get the root script directory path (3 levels up from this file)"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_base_cache_dir() -> str:
    """Get or create the model cache directory"""
    try:
        import folder_paths # only works if comfyui is available
        cache_dir = os.path.join(folder_paths.models_dir, SEEDVR2_FOLDER_NAME)
        folder_paths.add_model_folder_path(SEEDVR2_MODEL_TYPE, cache_dir)
    except:
        cache_dir = f"./{SEEDVR2_MODEL_TYPE}_models"
    
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_all_model_paths() -> list:
    """Get all registered model paths including those from extra_model_paths.yaml"""
    try:
        import folder_paths
        # Ensure default path is registered first
        get_base_cache_dir()
        # Get all paths registered for seedvr2 model type
        paths = folder_paths.get_folder_paths(SEEDVR2_MODEL_TYPE)
        return paths if paths else [get_base_cache_dir()]
    except:
        return [get_base_cache_dir()]


def get_all_model_files() -> dict:
    """
    Get a mapping of all model files to their full paths across all registered directories.
    
    Returns:
        dict: Mapping of filename -> full path for all discovered model files
    """
    model_files = {}
    all_paths = get_all_model_paths()
    
    for path in all_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if is_supported_model_file(file):
                    # Only keep first occurrence of each file (priority order)
                    if file not in model_files:
                        model_files[file] = os.path.join(path, file)
    
    return model_files


def find_model_file(filename: str, fallback_dir: Optional[str] = None) -> str:
    """
    Find a model file in any registered path.
    
    Args:
        filename: Name of the model file to find
        fallback_dir: Directory to use if file not found in any registered path
        
    Returns:
        str: Full path to the model file
    """
    # Get all model files
    model_files = get_all_model_files()
    
    # Return path if found
    if filename in model_files:
        return model_files[filename]
    
    # Fallback to specified directory or base cache dir
    if fallback_dir:
        return os.path.join(fallback_dir, filename)
    else:
        return os.path.join(get_base_cache_dir(), filename)


def get_validation_cache_path() -> str:
    """Get path to model validation cache file"""
    cache_dir = get_base_cache_dir()
    return os.path.join(cache_dir, ".validation_cache.json")


def is_supported_model_file(filename: str) -> bool:
    """Check if a file has a supported model extension"""
    return any(filename.endswith(ext) for ext in SUPPORTED_MODEL_EXTENSIONS)


def suppress_tensor_warnings() -> None:
    """
    Suppress common tensor conversion and numpy array warnings that are expected behavior
    when working with GGUF tensors and numpy arrays.
    """
    warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)


def get_unique_id() -> Optional[int]:
    """
    Extract node's unique_id from ComfyUI execution stack.
    
    In ComfyUI V3 schema, unique_id is not passed as a parameter to execute()
    but is available in the execution stack frames. This function walks the
    stack to reliably extract it.
    
    The unique_id is used for node-specific caching and state management,
    ensuring each node instance maintains separate cache entries even when
    using the same configuration.
    
    Returns:
        Node's unique_id as integer if found, None otherwise
        
    Note:
        Temp solution until we find a better approach/the schema gets updated...
    """
    frame = inspect.currentframe()
    try:
        # Check up to 10 frames (execution.py is typically 2-4 frames up)
        for _ in range(10):
            frame = frame.f_back
            if frame is None:
                break
            
            # Look for unique_id in frame locals
            if 'unique_id' in frame.f_locals:
                unique_id = frame.f_locals['unique_id']
                # Verify it's a valid value
                if isinstance(unique_id, (int, str)) and unique_id is not None:
                    return int(unique_id) if isinstance(unique_id, str) else unique_id
        
        return None
    finally:
        del frame  # Prevent reference cycles