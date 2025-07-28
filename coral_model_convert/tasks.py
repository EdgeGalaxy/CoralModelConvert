"""Task management for async conversion operations"""

import uuid
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from loguru import logger

from .models import ConversionStatus, ConversionResult
from .adapter import ModelConverterAdapter
from .exceptions import ModelConversionError


class TaskManager:
    """Manages async conversion tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, ConversionResult] = {}
        self.adapter = ModelConverterAdapter()
    
    def create_task(self, task_id: Optional[str] = None) -> str:
        """Create a new conversion task"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = ConversionResult(
            task_id=task_id,
            status=ConversionStatus.PENDING,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.tasks[task_id] = task
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
        model_url: str,
        model_path: str,
        output_dir: str,
        **kwargs
    ):
        """Download model from URL and run conversion in background"""
        try:
            self.update_task_status(task_id, ConversionStatus.PROCESSING)
            logger.info(f"Starting URL-based conversion task {task_id}")
            
            # Download model from URL
            logger.info(f"Downloading model from URL: {model_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download model: HTTP {response.status}")
                    
                    content = await response.read()
                    
                    # Save downloaded content to file
                    with open(model_path, "wb") as f:
                        f.write(content)
            
            logger.info(f"Model downloaded and saved to: {model_path}")
            
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


# Global task manager instance
task_manager = TaskManager()