"""API routers for model conversion endpoints"""

import os
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
    
    # Create task
    task_id = task_manager.create_task()
    
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
        
        background_tasks.add_task(
            task_manager.run_conversion,
            task_id=task_id,
            source_format="onnx",
            target_format="rknn",
            model_path=str(model_path),
            output_dir=str(task_output_dir),
            **conversion_params
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
    request: RKNNConversionURLRequest
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
        
        # Create task
        task_id = task_manager.create_task(
            callback_url=request.callback_url,
            callback_token=request.callback_token,
            callback_payload=request.callback_payload,
        )
        
        # Create output directory for this task
        task_output_dir = OUTPUT_DIR / task_id
        task_output_dir.mkdir(exist_ok=True)
        
        # Execute conversion inline (no background task)
        model_path = TEMP_DIR / f"{task_id}_model.onnx"

        await task_manager.run_conversion_from_url(
            task_id=task_id,
            model_oss_key=request.model_oss_key,
            model_path=str(model_path),
            output_dir=str(task_output_dir),
            output_oss_key=request.output_oss_key,
            **conversion_request.model_dump(),
        )

        # Build response based on final task status
        final_task = task_manager.get_task(task_id)
        final_status = final_task.status if final_task else ConversionStatus.FAILED
        message = (
            "Conversion completed successfully from URL"
            if final_status == ConversionStatus.COMPLETED
            else "Conversion failed from URL"
        )

        return ConversionResponse(
            task_id=task_id,
            status=final_status,
            message=message,
            created_at=final_task.created_at if final_task else datetime.utcnow().isoformat(),
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


@router.post("/test/download", response_model=DownloadTestResponse)
async def test_download_source(request: DownloadTestRequest):
    """Test if a third-party URL or an OSS key can be downloaded locally.

    Downloads to a temporary file under TEMP_DIR and cleans it up after the check.
    Enforces MAX_FILE_SIZE unless a smaller `max_size` is provided in the request.
    """
    # Validate parameters: exactly one of url or model_oss_key
    if bool(request.url) == bool(request.model_oss_key):
        raise HTTPException(status_code=400, detail="Provide exactly one of 'url' or 'model_oss_key'")

    max_size = request.max_size or MAX_FILE_SIZE
    temp_id = uuid.uuid4().hex

    if request.url:
        # Validate URL format
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")

        temp_path = TEMP_DIR / f"download_test_{temp_id}"
        total_size = 0
        try:
            timeout = aiohttp.ClientTimeout(total=60, sock_connect=15, sock_read=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(request.url) as response:
                    if response.status != 200:
                        return DownloadTestResponse(
                            ok=False, source="url", size=None,
                            message=f"HTTP {response.status} while downloading"
                        )

                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            if int(content_length) > max_size:
                                return DownloadTestResponse(
                                    ok=False, source="url", size=None,
                                    message="Content length exceeds maximum allowed size"
                                )
                        except Exception:
                            # Proceed with streaming
                            pass

                    chunk_size = 1024 * 512  # 512KB
                    with open(temp_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            total_size += len(chunk)
                            if total_size > max_size:
                                return DownloadTestResponse(
                                    ok=False, source="url", size=total_size,
                                    message="Downloaded size exceeds maximum allowed size"
                                )
                            f.write(chunk)

            return DownloadTestResponse(
                ok=True, source="url", size=total_size, message=f"Downloaded to {temp_path.name} (cleaned up)"
            )
        except Exception as e:
            return DownloadTestResponse(
                ok=False, source="url", size=total_size or None, message=str(e)
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    else:
        # OSS key path
        temp_path = TEMP_DIR / f"download_test_{temp_id}"
        try:
            local_path = await download_file_from_cloud(request.model_oss_key, str(temp_path), max_size=max_size)
            size = Path(local_path).stat().st_size if Path(local_path).exists() else None
            return DownloadTestResponse(
                ok=True, source="oss", size=size, message=f"Downloaded to {Path(local_path).name} (cleaned up)"
            )
        except Exception as e:
            return DownloadTestResponse(
                ok=False, source="oss", size=None, message=str(e)
            )
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
