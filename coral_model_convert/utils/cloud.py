"""Cloud storage utilities for file upload operations"""

from typing import Union, Optional
import os
import oss2
from asyncer import asyncify
from loguru import logger


class CloudConfig:
    """Cloud storage configuration"""
    
    def __init__(self):
        self.oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.oss_endpoint = os.getenv("OSS_ENDPOINT")
        self.oss_bucket_name = os.getenv("OSS_BUCKET_NAME")
        
        # Validate required settings
        if not all([self.oss_access_key_id, self.oss_access_key_secret, self.oss_endpoint, self.oss_bucket_name]):
            logger.warning("OSS configuration incomplete. Cloud upload will be disabled.")
            self.enabled = False
        else:
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