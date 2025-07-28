"""Pydantic models for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ConversionStatus(str, Enum):
    """Conversion status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RKNNConversionRequest(BaseModel):
    """Request model for RKNN conversion"""
    target_platform: str = Field(default="rk3588", description="Target RKNN platform")
    hybrid_quant: bool = Field(default=True, description="Use hybrid quantization")
    quantized_algorithm: str = Field(default="normal", description="Quantization algorithm")
    optimization_level: int = Field(default=3, ge=0, le=3, description="Optimization level")
    rknn_batchsize: Optional[int] = Field(default=None, description="RKNN batch size")
    with_acc_analysis: bool = Field(default=False, description="Perform accuracy analysis")
    step: str = Field(default="onestep", description="Conversion step")


class RKNNConversionURLRequest(BaseModel):
    """Request model for RKNN conversion via URL"""
    model_url: str = Field(description="URL of the ONNX model to convert")
    target_platform: str = Field(default="rk3588", description="Target RKNN platform")
    hybrid_quant: bool = Field(default=True, description="Use hybrid quantization")
    quantized_algorithm: str = Field(default="normal", description="Quantization algorithm")
    optimization_level: int = Field(default=3, ge=0, le=3, description="Optimization level")
    rknn_batchsize: Optional[int] = Field(default=None, description="RKNN batch size")
    with_acc_analysis: bool = Field(default=False, description="Perform accuracy analysis")
    step: str = Field(default="onestep", description="Conversion step")


class ConversionResponse(BaseModel):
    """Response model for conversion requests"""
    task_id: str = Field(description="Unique task identifier")
    status: ConversionStatus = Field(description="Current conversion status")
    message: str = Field(description="Status message")
    created_at: str = Field(description="Task creation timestamp")


class ConversionResult(BaseModel):
    """Model for conversion results"""
    task_id: str
    status: ConversionStatus
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for parameter validation"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class AvailableConversionsResponse(BaseModel):
    """Response model for available conversions"""
    conversions: Dict[str, List[str]] = Field(description="Available conversion paths")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str