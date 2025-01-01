from abc import ABC, abstractmethod
import os
import shutil
import boto3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import json

logger = logging.getLogger(__name__)

class StorageService(ABC):
    """Abstract base class defining storage interface"""
    
    @abstractmethod
    def store_file(self, file_data: bytes, filename: str) -> str:
        """Store a file and return its path"""
        pass
        
    @abstractmethod
    def get_file(self, file_path: str) -> Optional[bytes]:
        """Retrieve a file's contents"""
        pass
        
    @abstractmethod
    def cleanup_old_files(self):
        """Clean up old files"""
        pass

class LocalStorageService(StorageService):
    """Local storage implementation for development"""
    
    def __init__(self):
        """Initialize local storage"""
        self.storage_dir = Path("local_storage")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create a manifest file to track stored files
        self.manifest_file = self.storage_dir / "manifest.json"
        self.manifest = self._load_manifest()
        
    def _load_manifest(self) -> dict:
        """Load or create storage manifest"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {'files': {}}
        return {'files': {}}
        
    def _save_manifest(self):
        """Save storage manifest"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
            
    def store_file(self, file_data: bytes, filename: str) -> str:
        """
        Store file locally
        
        Args:
            file_data: File contents
            filename: Original filename
            
        Returns:
            str: Local file path
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            file_path = self.storage_dir / safe_filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_data)
                
            # Update manifest
            self.manifest['files'][safe_filename] = {
                'original_name': filename,
                'timestamp': timestamp,
                'size': len(file_data)
            }
            self._save_manifest()
            
            logger.info(f"Stored file locally: {safe_filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error storing file locally: {e}")
            raise
            
    def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file contents
        
        Args:
            file_path: Path to file
            
        Returns:
            Optional[bytes]: File contents if found
        """
        try:
            path = Path(file_path)
            if path.exists():
                with open(path, 'rb') as f:
                    return f.read()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving file: {e}")
            return None
            
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Remove old files
        
        Args:
            max_age_hours: Maximum age of files to keep
        """
        try:
            current_time = datetime.now()
            files_to_remove = []
            
            for filename, info in self.manifest['files'].items():
                file_time = datetime.strptime(info['timestamp'], '%Y%m%d_%H%M%S')
                age = current_time - file_time
                
                if age.total_seconds() > max_age_hours * 3600:
                    file_path = self.storage_dir / filename
                    if file_path.exists():
                        file_path.unlink()
                    files_to_remove.append(filename)
                    
            # Update manifest
            for filename in files_to_remove:
                del self.manifest['files'][filename]
            self._save_manifest()
            
            logger.info(f"Cleaned up {len(files_to_remove)} old files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class CloudStorageService(StorageService):
    """Cloud storage implementation for production"""
    
    def __init__(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            self.bucket_name = os.getenv('S3_BUCKET_NAME')
            if not self.bucket_name:
                raise ValueError("S3_BUCKET_NAME not set")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
            
    def store_file(self, file_data: bytes, filename: str) -> str:
        """Store file in S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_path = f"uploads/{timestamp}_{filename}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_path,
                Body=file_data
            )
            
            return s3_path
            
        except Exception as e:
            logger.error(f"Error storing file in S3: {e}")
            raise
            
    def get_file(self, s3_path: str) -> Optional[bytes]:
        """Retrieve file from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Error retrieving file from S3: {e}")
            return None
            
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old files from S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='uploads/'
            )
            
            if 'Contents' not in response:
                return
                
            current_time = datetime.now()
            
            for obj in response['Contents']:
                age = current_time - obj['LastModified'].replace(tzinfo=None)
                if age.total_seconds() > max_age_hours * 3600:
                    self.s3_client.delete_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
        except Exception as e:
            logger.error(f"Error cleaning up S3 files: {e}")

def get_storage_service() -> StorageService:
    """
    Factory function to get appropriate storage service
    
    Returns:
        StorageService: Local or Cloud storage service based on environment
    """
    if os.getenv('USE_CLOUD_STORAGE', '').lower() == 'true':
        return CloudStorageService()
    return LocalStorageService()
