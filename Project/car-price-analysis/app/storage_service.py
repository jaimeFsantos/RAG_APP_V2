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
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old files"""
        pass
        
    @abstractmethod
    def upload_file(self, file_data: Union[bytes, str], filename: str) -> str:
        """Upload a file and return its path"""
        pass


# storage_service.py
class LocalStorageService(StorageService):
    """Local storage implementation for development"""
    
    def __init__(self):
        """Initialize local storage"""
        self.storage_dir = Path("local_storage")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create specific subdirectories
        self.analytics_dir = self.storage_dir / "analytics"
        self.uploads_dir = self.storage_dir / "uploads"
        self.temp_dir = self.storage_dir / "temp"
        
        # Create all subdirectories
        for directory in [self.analytics_dir, self.uploads_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        # Create manifest file to track stored files
        self.manifest_file = self.storage_dir / "manifest.json"
        self.manifest = self._load_manifest()
        
    def store_file(self, file_data: bytes, filename: str) -> str:
        """Store file locally with subdirectory support"""
        try:
            # Handle subdirectory paths
            if filename.startswith('analytics/'):
                base_dir = self.analytics_dir
                filename = filename.replace('analytics/', '')
            elif filename.startswith('temp/'):
                base_dir = self.temp_dir
                filename = filename.replace('temp/', '')
            else:
                base_dir = self.uploads_dir
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            file_path = base_dir / safe_filename
            
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
            logger.error(f"Error storing file locally: {str(e)}")
            raise
    
    def _create_manifest(self):
        """Create initial manifest file"""
        manifest = {
            'created_at': datetime.now().isoformat(),
            'files': {}
        }
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def _load_manifest(self) -> dict:
        """Load or create storage manifest"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid manifest file, creating new one")
                return {'files': {}}
        return {'files': {}}
    
    def _save_manifest(self):
        """Save current manifest to file"""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest: {e}")
    
    def upload_file(self, file_data: Union[bytes, str], filename: str) -> str:
        """Upload file data (string or binary)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            file_path = self.temp_dir / safe_filename
            
            if isinstance(file_data, str):
                with open(file_path, 'w') as f:
                    f.write(file_data)
            else:
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                    
            logger.info(f"Uploaded file to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """Retrieve file contents"""
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
        """Remove old temporary files"""
        try:
            current_time = datetime.now()
            
            for directory in [self.temp_dir, self.viz_dir]:
                for file_path in directory.glob('*'):
                    if file_path.is_file():
                        file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_age.total_seconds() > max_age_hours * 3600:
                            file_path.unlink()
                            logger.info(f"Cleaned up old file: {file_path}")
                            
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")
    def upload_file(self, file_data: Union[bytes, str], filename: str) -> str:
        """Upload file to local storage with verification"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}"
            file_path = self.storage_dir / safe_filename
            
            # Log the attempt
            logger.info(f"Attempting to store file: {safe_filename}")
            
            if isinstance(file_data, str):
                with open(file_path, 'w') as f:
                    f.write(file_data)
            else:
                with open(file_path, 'wb') as f:
                    f.write(file_data)
            
            # Verify file was created
            if not file_path.exists():
                raise FileNotFoundError(f"Failed to create file: {file_path}")
                
            logger.info(f"Successfully stored file at: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def cleanup_old_files(self, max_age_hours: int = 1, max_folder_size_mb: int = 350):
        """Clean up old files from local storage if the folder exceeds max size"""
        try:
            current_time = datetime.now()
            total_size = 0

            # Calculate the total size of the folder
            for file_path in self.storage_dir.glob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            # Convert total size to MB
            total_size_mb = total_size / (1024 * 1024)

            if total_size_mb > max_folder_size_mb:
                logger.info(f"Folder size ({total_size_mb:.2f} MB) exceeds limit ({max_folder_size_mb} MB). Cleaning up.")

                for file_path in self.storage_dir.glob('*'):
                    if file_path.name == 'manifest.json':
                        continue

                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)

                    # Delete files older than max_age_hours
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path.name}")
            else:
                logger.info(f"Folder size ({total_size_mb:.2f} MB) is within the limit.")

        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

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
