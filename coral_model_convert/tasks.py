"""Task management for async conversion operations"""

import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from .models import ConversionStatus, ConversionResult
from .adapter import ModelConverterAdapter
from .exceptions import ModelConversionError
from .utils.cloud import (
    upload_file_to_cloud,
    is_cloud_enabled,
    generate_signed_url,
    download_file_from_cloud,
)
from .config import MAX_FILE_SIZE


@dataclass
class CallbackConfig:
    """Callback configuration for task events"""

    url: str
    token: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    notify_on: Tuple[ConversionStatus, ...] = (
        ConversionStatus.PROCESSING,
        ConversionStatus.COMPLETED,
        ConversionStatus.FAILED,
    )


class TaskManager:
    """Manages async conversion tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, ConversionResult] = {}
        self.adapter = ModelConverterAdapter()
        self.callbacks: Dict[str, CallbackConfig] = {}
    
    def create_task(
        self,
        task_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        callback_token: Optional[str] = None,
        callback_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new conversion task"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = ConversionResult(
            task_id=task_id,
            status=ConversionStatus.PENDING,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.tasks[task_id] = task
        if callback_url:
            self.callbacks[task_id] = CallbackConfig(
                url=callback_url,
                token=callback_token,
                payload=callback_payload,
            )
        return task_id
    
    def get_task(self, task_id: str) -> Optional[ConversionResult]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: ConversionStatus, **kwargs):
        """Update task status and metadata"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            if status in [ConversionStatus.COMPLETED, ConversionStatus.FAILED]:
                task.completed_at = datetime.utcnow().isoformat()
            
            callback_cfg = self.callbacks.get(task_id)
            if callback_cfg and status in callback_cfg.notify_on:
                asyncio.create_task(self._dispatch_callback(task_id, callback_cfg))
                if status in [ConversionStatus.COMPLETED, ConversionStatus.FAILED]:
                    # 清理已完成任务的回调配置
                    self.callbacks.pop(task_id, None)

    async def _dispatch_callback(self, task_id: str, callback_cfg: CallbackConfig):
        """Send callback notification with task status"""
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Callback skipped, task {task_id} not found")
            return

        payload = {
            "task_id": task_id,
            "status": task.status.value,
            "metadata": task.metadata or {},
            "error_message": task.error_message,
            "callback_payload": callback_cfg.payload,
            "timestamp": datetime.utcnow().isoformat(),
        }

        headers = {"Content-Type": "application/json"}
        if callback_cfg.token:
            headers["Authorization"] = f"Bearer {callback_cfg.token}"

        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    callback_cfg.url, json=payload, headers=headers
                ) as response:
                    if response.status >= 400:
                        text = await response.text()
                        logger.error(
                            f"Callback to {callback_cfg.url} failed: {response.status} - {text}"
                        )
                    else:
                        logger.info(
                            f"Callback to {callback_cfg.url} succeeded for task {task_id}"
                        )
        except Exception as exc:
            logger.exception(
                f"Error while sending callback to {callback_cfg.url}: {exc}"
            )
    
    async def run_conversion(
        self,
        task_id: str,
        source_format: str,
        target_format: str,
        model_path: str,
        output_dir: str,
        **kwargs
    ):
        """Run conversion in background"""
        try:
            self.update_task_status(task_id, ConversionStatus.PROCESSING)
            logger.info(f"Starting conversion task {task_id}")
            
            # Create a wrapper function to handle keyword arguments
            def convert_wrapper():
                return self.adapter.convert_model(
                    source_format=source_format,
                    target_format=target_format,
                    model_path=model_path,
                    output_dir=output_dir,
                    **kwargs
                )
            
            # Run conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, convert_wrapper)
            
            self.update_task_status(
                task_id,
                ConversionStatus.COMPLETED,
                output_path=result.get("output_path"),
                metadata=result
            )
            
            logger.info(f"Conversion task {task_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Conversion task {task_id} failed")
            self.update_task_status(
                task_id,
                ConversionStatus.FAILED,
                error_message=str(e)
            )

    async def run_conversion_from_url(
        self,
        task_id: str,
        model_oss_key: str,
        model_path: str,
        output_dir: str,
        output_oss_key: Optional[str] = None,
        **kwargs
    ):
        """Download model via OSS key and run conversion in background"""
        try:
            self.update_task_status(task_id, ConversionStatus.PROCESSING)
            logger.info(f"Starting URL-based conversion task {task_id}")
            
            # Download model directly from OSS to target path
            logger.info(f"Downloading model from OSS: {model_oss_key} -> {model_path}")
            await download_file_from_cloud(model_oss_key, model_path, max_size=MAX_FILE_SIZE)
            logger.info(f"Model downloaded to: {model_path}")
            
            # Create a wrapper function for conversion
            def convert_wrapper():
                return self.adapter.convert_model(
                    source_format="onnx",
                    target_format="rknn",
                    model_path=model_path,
                    output_dir=output_dir,
                    **kwargs
                )
            
            # Run conversion in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, convert_wrapper)
            
            # Upload to cloud if oss_key is provided and cloud is enabled
            cloud_key = None
            if output_oss_key and is_cloud_enabled():
                try:
                    output_path = result.get("output_path")
                    if output_path and Path(output_path).exists():
                        logger.info(f"Uploading converted model to cloud: {output_oss_key}")
                        cloud_key = await upload_file_to_cloud(output_path, output_oss_key)
                        logger.info(f"Model uploaded successfully to: {cloud_key}")
                except Exception as e:
                    logger.error(f"Failed to upload to cloud: {str(e)}")
                    # Don't fail the task if cloud upload fails
            elif output_oss_key and not is_cloud_enabled():
                logger.warning("OSS key provided but cloud storage is not configured")
            
            # Update result metadata to include cloud key and signed URL
            if cloud_key:
                result["cloud_key"] = cloud_key
                try:
                    result["cloud_url"] = await generate_signed_url(cloud_key)
                except Exception as e:
                    logger.warning(f"Failed to generate signed URL for {cloud_key}: {e}")
            
            self.update_task_status(
                task_id,
                ConversionStatus.COMPLETED,
                output_path=result.get("output_path"),
                metadata=result
            )
            
            logger.info(f"URL-based conversion task {task_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"URL-based conversion task {task_id} failed")
            self.update_task_status(
                task_id,
                ConversionStatus.FAILED,
                error_message=str(e)
            )
            try:
                path_obj = Path(model_path)
                if path_obj.exists():
                    path_obj.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary model file {model_path}: {cleanup_error}")


# Global task manager instance
task_manager = TaskManager()
