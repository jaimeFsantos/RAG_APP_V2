import os
import hashlib
import boto3
from botocore.exceptions import ClientError
import streamlit as st
from datetime import datetime, timedelta
import json
import logging
from typing import Optional, Dict, Any
import secrets
import hmac

class SecurityManager:
    def __init__(self):
        """Initialize security manager with configuration"""
        self.session_duration = timedelta(hours=12)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for security events"""
        self.logger = logging.getLogger('security_manager')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('security.log')
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def initialize_session_state(self):
        """Initialize security-related session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.now()

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash using constant-time comparison"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return hmac.compare_digest(password_hash, stored_hash)
        except Exception as e:
            self.logger.error(f"Password verification error: {str(e)}")
            return False

    def check_session_timeout(self) -> bool:
        """Check if current session has timed out"""
        if datetime.now() - st.session_state.last_activity > self.session_duration:
            st.session_state.authenticated = False
            return True
        st.session_state.last_activity = datetime.now()
        return False

    def update_failed_login(self):
        """Update failed login attempts and implement exponential backoff"""
        st.session_state.login_attempts += 1
        wait_time = min(300, 2 ** (st.session_state.login_attempts - 1))  # Max 5 minutes
        st.session_state.next_attempt = datetime.now() + timedelta(seconds=wait_time)

    def can_attempt_login(self) -> bool:
        """Check if login can be attempted based on previous failures"""
        if st.session_state.login_attempts == 0:
            return True
        return datetime.now() >= st.session_state.next_attempt

class StorageManager:
    def __init__(self):
        """Initialize storage manager with AWS credentials and bucket configuration"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.local_cache_dir = '.cache'
        os.makedirs(self.local_cache_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for storage operations"""
        self.logger = logging.getLogger('storage_manager')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('storage.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for secure file access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            self.logger.error(f"Error generating presigned URL: {str(e)}")
            return None

    async def upload_file(self, file_obj, folder: str) -> Optional[str]:
        """Upload file to S3 with proper error handling and logging"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(file_obj.read()).hexdigest()
            file_obj.seek(0)  # Reset file pointer
            
            # Create a unique file path
            file_path = f"{folder}/{timestamp}_{file_hash}_{file_obj.name}"
            
            # Upload to S3
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, file_path)
            
            self.logger.info(f"Successfully uploaded file: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            return None

    async def download_file(self, file_path: str) -> Optional[str]:
        """Download file from S3 to local cache with proper error handling"""
        try:
            local_path = os.path.join(self.local_cache_dir, os.path.basename(file_path))
            
            # Download file from S3
            self.s3_client.download_file(self.bucket_name, file_path, local_path)
            
            self.logger.info(f"Successfully downloaded file: {file_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            return None

    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old files from local cache"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(self.local_cache_dir):
                file_path = os.path.join(self.local_cache_dir, filename)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if current_time - file_modified > timedelta(hours=max_age_hours):
                    os.remove(file_path)
                    self.logger.info(f"Removed cached file: {filename}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {str(e)}")

class FileValidator:
    def __init__(self):
        """Initialize file validator with security configurations"""
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_extensions = {'.csv', '.pdf', '.xlsx'}
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for file validation"""
        self.logger = logging.getLogger('file_validator')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('file_validation.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def validate_file(self, file_obj) -> tuple[bool, str]:
        """Validate uploaded file for security and compatibility"""
        try:
            # Check file size
            file_obj.seek(0, 2)  # Seek to end
            size = file_obj.tell()
            file_obj.seek(0)  # Reset position
            
            if size > self.max_file_size:
                return False, "File size exceeds maximum limit"

            # Check file extension
            file_ext = os.path.splitext(file_obj.name)[1].lower()
            if file_ext not in self.allowed_extensions:
                return False, "File type not allowed"

            # Basic content validation
            content_sample = file_obj.read(1024)  # Read first 1KB
            file_obj.seek(0)  # Reset position

            # Check for potential malicious content
            if b'<script' in content_sample or b'<?php' in content_sample:
                return False, "File contains potentially malicious content"

            self.logger.info(f"File validated successfully: {file_obj.name}")
            return True, "File validated successfully"
            
        except Exception as e:
            self.logger.error(f"Error validating file: {str(e)}")
            return False, f"Error validating file: {str(e)}"