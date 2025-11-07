"""API routers for model conversion endpoints"""

import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from ..models import (
    ConversionResponse,
    ConversionResult,
    ValidationResponse,
    AvailableConversionsResponse,
    RKNNConversionRequest,
    RKNNConversionURLRequest,
    ConversionStatus,
    DownloadTestRequest,
    DownloadTestResponse,
)
from ..tasks import task_manager
from ..config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS, TEMP_DIR, OUTPUT_DIR
from ..exceptions import ModelConversionError
from ..utils.cloud import download_file_from_cloud
import uuid

router = APIRouter(prefix="/api/v1", tags=["conversion"])


@router.get("/conversions", response_model=AvailableConversionsResponse)
async def get_available_conversions():
    """Get available conversion paths"""
    conversions = task_manager.adapter.get_available_conversions()
    return AvailableConversionsResponse(conversions=conversions)


@router.post("/validate/rknn", response_model=ValidationResponse)
async def validate_rknn_params(request: RKNNConversionRequest):
    """Validate RKNN conversion parameters"""
    validation_result = task_manager.adapter.validate_conversion_params(
        source_format="onnx",
        target_format="rknn",
        **request.model_dump()
    )
    return ValidationResponse(**validation_result)


@router.post("/convert/rknn", response_model=ConversionResponse)
async def convert_to_rknn(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    dataset_file: Optional[UploadFile] = File(None),
    task_id: Optional[str] = Form(None),
    target_platform: str = Form("rk3588"),
    hybrid_quant: bool = Form(True),
    quantized_algorithm: str = Form("normal"),
    optimization_level: int = Form(3),
    rknn_batchsize: Optional[int] = Form(None),
    with_acc_analysis: bool = Form(False),
    step: str = Form("onestep")
):
    """Convert ONNX model to RKNN format"""
    
    # Validate file size
    if model_file.size and model_file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate file extension
    model_ext = Path(model_file.filename).suffix.lower()
    if model_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create conversion request
    request = RKNNConversionRequest(
        target_platform=target_platform,
        hybrid_quant=hybrid_quant,
        quantized_algorithm=quantized_algorithm,
        optimization_level=optimization_level,
        rknn_batchsize=rknn_batchsize,
        with_acc_analysis=with_acc_analysis,
        step=step
    )
    
    # Validate parameters
    validation_result = task_manager.adapter.validate_conversion_params(
        source_format="onnx",
        target_format="rknn",
        **request.model_dump()
    )
    
    if not validation_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid parameters: {', '.join(validation_result['errors'])}"
        )
    
    # Create task (use provided task_id if any)
    task_id = task_manager.create_task(task_id=task_id)
    
    # Save uploaded files
    model_path = TEMP_DIR / f"{task_id}_{model_file.filename}"
    
    try:
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        # Handle dataset file if provided
        dataset_path = None
        if dataset_file:
            dataset_path = TEMP_DIR / f"{task_id}_{dataset_file.filename}"
            with open(dataset_path, "wb") as f:
                content = await dataset_file.read()
                f.write(content)
        
        # Create output directory for this task
        task_output_dir = OUTPUT_DIR / task_id
        task_output_dir.mkdir(exist_ok=True)
        
        # Start conversion in background
        conversion_params = request.model_dump()
        if dataset_path:
            conversion_params["dataset_file"] = str(dataset_path)
        
        # Directly schedule on the event loop to avoid BackgroundTasks issues
        asyncio.create_task(
            task_manager.run_conversion(
                task_id=task_id,
                source_format="onnx",
                target_format="rknn",
                model_path=str(model_path),
                output_dir=str(task_output_dir),
                **conversion_params,
            )
        )
        
        return ConversionResponse(
            task_id=task_id,
            status=ConversionStatus.PENDING,
            message="Conversion task created successfully",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        # Cleanup on error
        if model_path.exists():
            model_path.unlink()
        if dataset_path and dataset_path.exists():
            dataset_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Failed to create conversion task: {str(e)}")


@router.get("/tasks/{task_id}", response_model=ConversionResult)
async def get_task_status(task_id: str):
    """Get conversion task status"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.post("/convert/rknn/url", response_model=ConversionResponse)
async def convert_to_rknn_from_url(
    request: RKNNConversionURLRequest,
    background_tasks: BackgroundTasks,
    is_sync: bool = False,
):
    """Convert ONNX model to RKNN format from URL"""
    
    try:
        # Validate OSS key
        if not request.model_oss_key:
            raise HTTPException(
                status_code=400,
                detail="model_oss_key is required"
            )
        
        if request.callback_url:
            if not request.callback_url.startswith(('http://', 'https://')):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid callback URL format. Must start with http:// or https://"
                )
            if len(request.callback_url) > 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Callback URL is too long"
                )
        
        # Validate file extension from URL
        key_path = Path(request.model_oss_key)
        model_ext = key_path.suffix.lower()
        if model_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Create conversion request
        conversion_request = RKNNConversionRequest(
            target_platform=request.target_platform,
            hybrid_quant=request.hybrid_quant,
            quantized_algorithm=request.quantized_algorithm,
            optimization_level=request.optimization_level,
            rknn_batchsize=request.rknn_batchsize,
            with_acc_analysis=request.with_acc_analysis,
            step=request.step
        )
        
        # Validate parameters
        validation_result = task_manager.adapter.validate_conversion_params(
            source_format="onnx",
            target_format="rknn",
            **conversion_request.model_dump()
        )
        
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters: {', '.join(validation_result['errors'])}"
            )
        
        # Create task (use provided task_id if any)
        task_id = task_manager.create_task(
            task_id=request.task_id,
            callback_url=request.callback_url,
            callback_token=request.callback_token,
            callback_payload=request.callback_payload,
        )
        
        # Create output directory for this task
        task_output_dir = OUTPUT_DIR / task_id
        task_output_dir.mkdir(exist_ok=True)
        
        # Prepare paths
        model_path = TEMP_DIR / f"{task_id}_model.onnx"

        if is_sync:
            # Run conversion synchronously
            await task_manager.run_conversion_from_url(
                task_id=task_id,
                model_oss_key=request.model_oss_key,
                model_path=str(model_path),
                output_dir=str(task_output_dir),
                output_oss_key=request.output_oss_key,
                **conversion_request.model_dump(),
            )

            task = task_manager.get_task(task_id)
            status = task.status if task else ConversionStatus.FAILED
            message = (
                "Conversion completed successfully from URL"
                if status == ConversionStatus.COMPLETED
                else "Conversion finished with errors from URL"
            )

            return ConversionResponse(
                task_id=task_id,
                status=status,
                message=message,
                created_at=(task.created_at if task else datetime.utcnow().isoformat()),
            )
        else:
            # Schedule conversion using FastAPI BackgroundTasks
            background_tasks.add_task(
                task_manager.run_conversion_from_url,
                task_id=task_id,
                model_oss_key=request.model_oss_key,
                model_path=str(model_path),
                output_dir=str(task_output_dir),
                output_oss_key=request.output_oss_key,
                **conversion_request.model_dump(),
            )

            return ConversionResponse(
                task_id=task_id,
                status=ConversionStatus.PENDING,
                message="Conversion task created successfully from URL",
                created_at=datetime.utcnow().isoformat(),
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversion task: {str(e)}")


@router.get("/tasks/{task_id}/download")
async def download_result(task_id: str):
    """Download converted model"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != ConversionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")
    
    if not task.output_path or not Path(task.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=task.output_path,
        filename=Path(task.output_path).name,
        media_type="application/octet-stream"
    )
