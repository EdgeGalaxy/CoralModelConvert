"""Global exception handlers for the FastAPI application"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from .exceptions import (
    ModelConversionError,
    ModelLoadError,
    ModelBuildError,
    ModelExportError,
    UnsupportedModelFormatError,
    InvalidParameterError
)


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(ModelConversionError)
    async def model_conversion_error_handler(request: Request, exc: ModelConversionError):
        logger.error(f"Model conversion error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Model Conversion Error",
                "detail": str(exc),
                "type": "model_conversion_error"
            }
        )
    
    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(request: Request, exc: ModelLoadError):
        logger.error(f"Model load error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Model Load Error",
                "detail": str(exc),
                "type": "model_load_error"
            }
        )
    
    @app.exception_handler(ModelBuildError)
    async def model_build_error_handler(request: Request, exc: ModelBuildError):
        logger.error(f"Model build error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Build Error",
                "detail": str(exc),
                "type": "model_build_error"
            }
        )
    
    @app.exception_handler(ModelExportError)
    async def model_export_error_handler(request: Request, exc: ModelExportError):
        logger.error(f"Model export error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Export Error",
                "detail": str(exc),
                "type": "model_export_error"
            }
        )
    
    @app.exception_handler(UnsupportedModelFormatError)
    async def unsupported_format_error_handler(request: Request, exc: UnsupportedModelFormatError):
        logger.error(f"Unsupported format error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported Model Format",
                "detail": str(exc),
                "type": "unsupported_format_error"
            }
        )
    
    @app.exception_handler(InvalidParameterError)
    async def invalid_parameter_error_handler(request: Request, exc: InvalidParameterError):
        logger.error(f"Invalid parameter error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid Parameters",
                "detail": str(exc),
                "type": "invalid_parameter_error"
            }
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_error_handler(request: Request, exc: FileNotFoundError):
        logger.error(f"File not found error: {str(exc)}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "File Not Found",
                "detail": str(exc),
                "type": "file_not_found_error"
            }
        )
    
    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        logger.error(f"Permission error: {str(exc)}")
        return JSONResponse(
            status_code=403,
            content={
                "error": "Permission Denied",
                "detail": "Insufficient permissions to access the requested resource",
                "type": "permission_error"
            }
        )
    
    @app.exception_handler(OSError)
    async def os_error_handler(request: Request, exc: OSError):
        logger.error(f"OS error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "System Error",
                "detail": "A system-level error occurred",
                "type": "os_error"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred",
                "type": "internal_error"
            }
        )