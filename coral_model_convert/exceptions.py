"""Custom exceptions for model conversion services"""


class ModelConversionError(Exception):
    """Base exception for model conversion errors"""
    pass


class ModelLoadError(ModelConversionError):
    """Raised when model loading fails"""
    pass


class ModelBuildError(ModelConversionError):
    """Raised when model building fails"""
    pass


class ModelExportError(ModelConversionError):
    """Raised when model exporting fails"""
    pass


class UnsupportedModelFormatError(ModelConversionError):
    """Raised when model format is not supported"""
    pass


class InvalidParameterError(ModelConversionError):
    """Raised when invalid parameters are provided"""
    pass