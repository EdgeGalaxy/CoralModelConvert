"""Unified adapter interface for model conversion services"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from pathlib import Path
from enum import Enum


class ModelFormat(Enum):
    """Supported model formats"""
    ONNX = "onnx"
    RKNN = "rknn"
    TFLITE = "tflite"
    TENSORFLOW = "pb"


class ConversionTarget(Enum):
    """Supported conversion targets"""
    RKNN = "rknn"
    # Future targets can be added here
    # TENSORRT = "tensorrt"
    # OPENVINO = "openvino"


class BaseModelConverter(ABC):
    """Base class for all model converters"""
    
    @abstractmethod
    def convert(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model to target format
        
        Returns:
            Dict containing conversion results including output path, metadata, etc.
        """
        pass
    
    @abstractmethod
    def validate_input(self, model_path: Union[str, Path]) -> bool:
        """
        Validate input model format
        
        Args:
            model_path: Path to the input model
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> list[ModelFormat]:
        """
        Get list of supported input formats
        
        Returns:
            List of supported ModelFormat enums
        """
        pass


class ModelConverterAdapter:
    """Unified adapter for different model conversion services"""
    
    def __init__(self):
        self._converters = {}
        self._register_converters()
    
    def _register_converters(self):
        """Register available converters"""
        from .converters.rknn_converter import RKNNConverter
        
        # Register RKNN converter for ONNX to RKNN conversion
        self._converters[(ModelFormat.ONNX, ConversionTarget.RKNN)] = RKNNConverter
    
    def get_available_conversions(self) -> Dict[str, list]:
        """Get available conversion paths"""
        conversions = {}
        for (source, target), converter_class in self._converters.items():
            if source.value not in conversions:
                conversions[source.value] = []
            conversions[source.value].append(target.value)
        return conversions
    
    def convert_model(
        self,
        source_format: Union[str, ModelFormat],
        target_format: Union[str, ConversionTarget],
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert model using appropriate converter
        
        Args:
            source_format: Source model format
            target_format: Target conversion format
            model_path: Path to source model
            output_dir: Output directory for converted model
            **kwargs: Additional parameters for specific converters
            
        Returns:
            Conversion result dictionary
        """
        # Normalize formats
        if isinstance(source_format, str):
            source_format = ModelFormat(source_format)
        if isinstance(target_format, str):
            target_format = ConversionTarget(target_format)
        
        # Find appropriate converter
        converter_key = (source_format, target_format)
        if converter_key not in self._converters:
            raise ValueError(
                f"No converter available for {source_format.value} -> {target_format.value}"
            )
        
        converter_class = self._converters[converter_key]
        
        # Handle RKNN converter specifically
        if target_format == ConversionTarget.RKNN:
            converter = converter_class(
                onnx_model_path=model_path,
                output_dir=output_dir,
                **kwargs
            )
            return converter.convert()
        
        # Future converters can be handled here
        raise NotImplementedError(f"Converter for {target_format.value} not implemented")
    
    def validate_conversion_params(
        self,
        source_format: Union[str, ModelFormat],
        target_format: Union[str, ConversionTarget],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate conversion parameters for specific converter
        
        Returns:
            Dictionary with validation results and any parameter adjustments
        """
        # Normalize formats
        if isinstance(source_format, str):
            source_format = ModelFormat(source_format)
        if isinstance(target_format, str):
            target_format = ConversionTarget(target_format)
        
        if target_format == ConversionTarget.RKNN:
            return self._validate_rknn_params(**kwargs)
        
        return {"valid": True, "errors": []}
    
    def _validate_rknn_params(self, **kwargs) -> Dict[str, Any]:
        """Validate RKNN conversion parameters"""
        from .config import RKNN_SUPPORTED_PLATFORMS, RKNN_SUPPORTED_QUANT_ALGOS
        
        errors = []
        warnings = []
        
        # Validate platform
        platform = kwargs.get("target_platform", "rk3588")
        if platform not in RKNN_SUPPORTED_PLATFORMS:
            errors.append(f"Unsupported platform: {platform}")
        
        # Validate quantization algorithm
        quant_algo = kwargs.get("quantized_algorithm", "normal")
        if quant_algo not in RKNN_SUPPORTED_QUANT_ALGOS:
            errors.append(f"Unsupported quantization algorithm: {quant_algo}")
        
        # Validate step
        step = kwargs.get("step", "onestep")
        if step not in ["onestep", "step1", "step2"]:
            errors.append(f"Unsupported step: {step}")
        
        # Check hybrid quantization consistency
        hybrid_quant = kwargs.get("hybrid_quant", True)
        dataset_file = kwargs.get("dataset_file")
        
        if hybrid_quant and not dataset_file:
            warnings.append("Hybrid quantization requires dataset file for best results")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }