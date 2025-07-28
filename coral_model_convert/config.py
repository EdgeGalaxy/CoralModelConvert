import os
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Create necessary directories
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API Configuration
API_TITLE = "Coral Model Convert API"
API_DESCRIPTION = "A unified model conversion service supporting RKNN and other formats"
API_VERSION = "0.1.0"

# File upload limits
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".onnx", ".pb", ".tflite"}

# RKNN specific configuration
RKNN_SUPPORTED_PLATFORMS = ["rk3562", "rk3566", "rk3568", "rk3588"]
RKNN_SUPPORTED_QUANT_ALGOS = ["normal", "mmse", "kl_divergence"]