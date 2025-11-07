"""Cloud storage utilities for file upload operations"""

from typing import Union, Optional
import os
import oss2
from asyncer import asyncify
from loguru import logger
from pathlib import Path


class CloudConfig:
    """Cloud storage configuration"""
    
    def __init__(self):
        self.oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.oss_endpoint = os.getenv("OSS_ENDPOINT")
        self.oss_bucket_name = os.getenv("OSS_BUCKET_NAME")
        
        # Validate required settings
        required_vars = {
            "OSS_ACCESS_KEY_ID": self.oss_access_key_id,
            "OSS_ACCESS_KEY_SECRET": self.oss_access_key_secret,
            "OSS_ENDPOINT": self.oss_endpoint,
            "OSS_BUCKET_NAME": self.oss_bucket_name,
        }
        missing_vars = [name for name, value in required_vars.items() if not value]
        if missing_vars:
            missing_str = ", ".join(missing_vars)
            message = (
                "OSS 配置缺失，无法启动 CoralModelConvert。"
                f" 请设置环境变量: {missing_str}"
            )
            logger.error(message)
            raise RuntimeError(message)

        self.enabled = True
    
    def get_bucket(self):
        """Get OSS bucket instance"""
        if not self.enabled:
            raise ValueError("OSS configuration is incomplete")
        
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, self.oss_bucket_name)
        return bucket


# Global config instance
cloud_config = CloudConfig()


async def upload_file_to_cloud(file_path: str, key: str) -> str:
    """Upload a file to cloud storage
    
    Args:
        file_path (str): Path to the local file to upload
        key (str): OSS key where the file will be stored
        
    Returns:
        str: The OSS key of the uploaded file
        
    Raises:
        ValueError: If cloud configuration is incomplete
        Exception: If upload fails
    """
    if not cloud_config.enabled:
        raise ValueError("Cloud storage is not configured")
    
    try:
        bucket = cloud_config.get_bucket()
        
        # Upload file
        await asyncify(bucket.put_object_from_file)(key, file_path)
        
        logger.info(f"File uploaded successfully to cloud: {key}")
        return key
        
    except Exception as e:
        logger.error(f"Failed to upload file to cloud: {str(e)}")
        raise


async def upload_data_to_cloud(data: Union[str, bytes], key: str) -> str:
    """Upload data directly to cloud storage
    
    Args:
        data (Union[str, bytes]): Data to upload
        key (str): OSS key where the data will be stored
        
    Returns:
        str: The OSS key of the uploaded data
        
    Raises:
        ValueError: If cloud configuration is incomplete
        Exception: If upload fails
    """
    if not cloud_config.enabled:
        raise ValueError("Cloud storage is not configured")
    
    try:
        bucket = cloud_config.get_bucket()
        
        # Upload data
        await asyncify(bucket.put_object)(key=key, data=data)
        
        logger.info(f"Data uploaded successfully to cloud: {key}")
        return key
        
    except Exception as e:
        logger.error(f"Failed to upload data to cloud: {str(e)}")
        raise


def is_cloud_enabled() -> bool:
    """Check if cloud storage is properly configured"""
    return cloud_config.enabled


async def generate_signed_url(key: str, expires: int = 3600) -> str:
    """Generate a signed URL for the given OSS object key.

    Args:
        key (str): OSS object key that needs to be accessed.
        expires (int): Expiration time in seconds for the signed URL.

    Returns:
        str: Signed URL for temporary access.

    Raises:
        ValueError: If cloud configuration is incomplete.
    """
    if not cloud_config.enabled:
        raise ValueError("Cloud storage is not configured")

    bucket = cloud_config.get_bucket()
    return await asyncify(bucket.sign_url)("GET", key, expires=expires)


async def download_file_from_cloud(key: str, file_path: str, max_size: Optional[int] = None) -> str:
    """Download a file from cloud storage to a local path.

    Args:
        key (str): OSS object key to download.
        file_path (str): Local filesystem path to save the file.
        max_size (Optional[int]): Optional maximum allowed size in bytes.

    Returns:
        str: Local path where the file was saved.

    Raises:
        ValueError: If cloud storage is not configured or size exceeds max_size.
        Exception: If download fails for other reasons.
    """
    if not cloud_config.enabled:
        raise ValueError("Cloud storage is not configured")

    try:
        bucket = cloud_config.get_bucket()

        # Pre-check size via HEAD if requested
        if max_size is not None:
            try:
                head = await asyncify(bucket.head_object)(key)
                # Try headers, fall back to attribute
                content_length = None
                if hasattr(head, "headers") and head.headers is not None:
                    cl = head.headers.get("content-length") or head.headers.get("Content-Length")
                    if cl is not None:
                        try:
                            content_length = int(cl)
                        except Exception:
                            content_length = None
                if content_length is None and hasattr(head, "content_length"):
                    content_length = head.content_length

                if content_length is not None and max_size is not None and content_length > max_size:
                    raise ValueError("Object is larger than allowed maximum size")
            except Exception as e:
                # Not fatal for HEAD failures; log and continue with best-effort download
                logger.warning(f"Failed to get object size for {key}: {e} Continuing to download and will verify size after.")

        # Ensure local directory exists
        dest = Path(file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download to file
        await asyncify(bucket.get_object_to_file)(key, str(dest))

        # Post-check size if requested
        if max_size is not None and dest.exists():
            size = dest.stat().st_size
            if size > max_size:
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise ValueError("Downloaded file exceeds maximum allowed size")

        logger.info(f"File downloaded successfully from cloud: {key} -> {file_path}")
        return str(dest)

    except Exception as e:
        logger.error(f"Failed to download file from cloud: {key} -> {file_path}: {e}")
        raise
