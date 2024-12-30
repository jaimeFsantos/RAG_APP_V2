"""
Enhanced Security Module

Implements comprehensive security features for the application including
audit logging, encryption, and secure file storage.

Environment:
    AWS EC2 Free Tier
    AWS S3 for secure storage
    AWS KMS for key management
    AWS DynamoDB for audit logs

Features:
    - Audit logging with CloudWatch integration
    - File encryption using AWS KMS
    - Secure file storage in S3
    - Session management
    - Access control
"""



import os
import boto3
import json
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
from botocore.exceptions import ClientError
import streamlit as st
from abc import ABC, abstractmethod
import uuid
import pytz
from functools import wraps
from dataclasses import dataclass
from enum import Enum

class AuditEventType(Enum):
    """Audit event types for tracking system activities"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    FILE_UPLOAD = "file_upload"
    FILE_ACCESS = "file_access"
    PREDICTION_MADE = "prediction_made"
    CHAT_INTERACTION = "chat_interaction"
    MODEL_TRAINING = "model_training"
    SYSTEM_ERROR = "system_error"
    USER_LOGOUT = "user_logout"
    SESSION_TIMEOUT = "session_timeout"

@dataclass
class AuditEvent:
    """Structure for audit events"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    ip_address: str
    details: Dict[str, Any]
    status: str
    session_id: str

# In enhanced_security.py

class SecurityConfig:
    """Security configuration settings"""
    def __init__(self):
        self.SESSION_DURATION = timedelta(hours=12)
        self.MAX_LOGIN_ATTEMPTS = 5
        self.LOCKOUT_DURATION = timedelta(minutes=30)
        self.PASSWORD_MIN_LENGTH = 12
        self.REQUIRE_MFA = False
        self.FILE_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB
        self.ALLOWED_EXTENSIONS = {'.csv', '.pdf', '.xlsx'}
        self.AUDIT_RETENTION_DAYS = 90
        
        # AWS Configuration - Now instance variables
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        self.kms_key_id = os.getenv('KMS_KEY_ID')
        
        # Generate encryption key if not exists
        if not os.getenv('FERNET_KEY'):
            key = Fernet.generate_key()
            os.environ['FERNET_KEY'] = key.decode()

class EncryptionService:
    """Handles data encryption and decryption"""
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.fernet = Fernet(os.getenv('FERNET_KEY').encode())
        self.kms_client = boto3.client('kms', region_name=self.config.aws_region)

class AuditLogger:
    """Handles audit logging with AWS integration"""
    def __init__(self):
        self.config = SecurityConfig()
        self.dynamodb = boto3.resource('dynamodb', region_name=self.config.AWS_REGION)
        self.table = self.dynamodb.Table('car_app_audit_logs')
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging handlers"""
        self.logger = logging.getLogger('audit_logger')
        self.logger.setLevel(logging.INFO)
        
        # Add CloudWatch handler if available
        if os.getenv('ENABLE_CLOUDWATCH', 'false').lower() == 'true':
            cloudwatch_handler = self._setup_cloudwatch_handler()
            self.logger.addHandler(cloudwatch_handler)
    
class AuditLogger:
    """Handles audit logging with AWS integration"""
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.dynamodb = boto3.resource('dynamodb', region_name=self.config.aws_region)
        self.table = self.dynamodb.Table('car_app_audit_logs')
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging handlers"""
        self.logger = logging.getLogger('audit_logger')
        self.logger.setLevel(logging.INFO)
        
        # Add CloudWatch handler if available
        if os.getenv('ENABLE_CLOUDWATCH', 'false').lower() == 'true':
            cloudwatch_handler = self._setup_cloudwatch_handler()
            self.logger.addHandler(cloudwatch_handler)
    
    def _setup_cloudwatch_handler(self):
        """Set up CloudWatch logging handler"""
        try:
            cloudwatch_client = boto3.client(
                'logs',
                region_name=self.config.aws_region
            )
            return cloudwatch_client
        except Exception as e:
            self.logger.error(f"Failed to setup CloudWatch handler: {e}")
            return None
    
    def log_event(self, event: AuditEvent):
        """Log an audit event to DynamoDB and CloudWatch"""
        try:
            # Prepare event data
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'details': event.details,
                'status': event.status,
                'session_id': event.session_id
            }
            
            # Log to DynamoDB
            self.table.put_item(Item=event_data)
            
            # Log to CloudWatch
            self.logger.info(json.dumps(event_data))
            
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
            raise

    def get_user_activity(self, user_id: str, start_date: datetime = None) -> List[Dict]:
        """Retrieve user activity logs"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=self.config.AUDIT_RETENTION_DAYS)
            
            response = self.table.query(
                KeyConditionExpression='user_id = :uid AND timestamp >= :start',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':start': start_date.isoformat()
                }
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            self.logger.error(f"Error retrieving user activity: {str(e)}")
            return []

    def get_user_activity(self, user_id: str, start_date: datetime = None) -> List[Dict]:
        """Retrieve user activity logs"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=self.config.AUDIT_RETENTION_DAYS)
            
            response = self.table.query(
                KeyConditionExpression='user_id = :uid AND timestamp >= :start',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':start': start_date.isoformat()
                }
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            self.logger.error(f"Error retrieving user activity: {str(e)}")
            return []

class SecureStorageService:
    """Handles secure file storage with AWS S3"""
    def __init__(self):
        self.config = SecurityConfig()
        self.s3_client = boto3.client('s3', region_name=self.config.AWS_REGION)
        self.encryption_service = EncryptionService()
        self.audit_logger = AuditLogger()
    
    def upload_file(self, file_obj, user_id: str, folder: str) -> Optional[str]:
        """Upload file with encryption to S3"""
        try:
            # Generate unique file path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(file_obj.read()).hexdigest()
            file_obj.seek(0)
            
            file_path = f"{folder}/{timestamp}_{file_hash}_{file_obj.name}"
            
            # Encrypt file content
            encrypted_data = self.encryption_service.encrypt_data(file_obj.read())
            
            # Upload to S3 with server-side encryption
            self.s3_client.put_object(
                Bucket=self.config.S3_BUCKET,
                Key=file_path,
                Body=encrypted_data,
                ServerSideEncryption='aws:kms',
                SSEKMSKeyId=self.config.KMS_KEY_ID
            )
            
            # Log upload event
            self.audit_logger.log_event(AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.FILE_UPLOAD,
                timestamp=datetime.now(pytz.UTC),
                user_id=user_id,
                ip_address=st.session_state.get('client_ip', 'unknown'),
                details={'file_name': file_obj.name, 'file_path': file_path},
                status='success',
                session_id=st.session_state.get('session_id', 'unknown')
            ))
            
            return file_path
            
        except Exception as e:
            self.audit_logger.log_event(AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.FILE_UPLOAD,
                timestamp=datetime.now(pytz.UTC),
                user_id=user_id,
                ip_address=st.session_state.get('client_ip', 'unknown'),
                details={'file_name': file_obj.name, 'error': str(e)},
                status='error',
                session_id=st.session_state.get('session_id', 'unknown')
            ))
            raise

class EnhancedSecurityManager:
    """Enhanced security manager with complete audit trail"""
    def __init__(self):
        self.config = SecurityConfig()
        self.audit_logger = AuditLogger()
        self.storage_service = SecureStorageService()
        self.encryption_service = EncryptionService()
        self._setup_session_tracking()
    
    def _setup_session_tracking(self):
        """Initialize session tracking"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'client_ip' not in st.session_state:
            st.session_state.client_ip = self._get_client_ip()
    
    def _get_client_ip(self) -> str:
        """Get client IP address from Streamlit request"""
        try:
            return st.get_client_ip()
        except:
            return "unknown"
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password with constant-time comparison"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            is_valid = hmac.compare_digest(password_hash, stored_hash)
            
            # Log authentication attempt
            self.audit_logger.log_event(AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.LOGIN_ATTEMPT,
                timestamp=datetime.now(pytz.UTC),
                user_id=st.session_state.get('username', 'unknown'),
                ip_address=st.session_state.client_ip,
                details={'success': is_valid},
                status='success' if is_valid else 'failure',
                session_id=st.session_state.session_id
            ))
            
            return is_valid
            
        except Exception as e:
            self.audit_logger.log_event(AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.SYSTEM_ERROR,
                timestamp=datetime.now(pytz.UTC),
                user_id=st.session_state.get('username', 'unknown'),
                ip_address=st.session_state.client_ip,
                details={'error': str(e)},
                status='error',
                session_id=st.session_state.session_id
            ))
            return False

def audit_trail(event_type: AuditEventType):
    """Decorator for adding audit trail to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now(pytz.UTC)
            try:
                result = func(*args, **kwargs)
                
                # Log successful execution
                if hasattr(args[0], 'audit_logger'):
                    args[0].audit_logger.log_event(AuditEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=event_type,
                        timestamp=start_time,
                        user_id=st.session_state.get('username', 'unknown'),
                        ip_address=st.session_state.get('client_ip', 'unknown'),
                        details={
                            'function': func.__name__,
                            'duration': (datetime.now(pytz.UTC) - start_time).total_seconds()
                        },
                        status='success',
                        session_id=st.session_state.get('session_id', 'unknown')
                    ))
                return result
                
            except Exception as e:
                # Log error
                if hasattr(args[0], 'audit_logger'):
                    args[0].audit_logger.log_event(AuditEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=AuditEventType.SYSTEM_ERROR,
                        timestamp=start_time,
                        user_id=st.session_state.get('username', 'unknown'),
                        ip_address=st.session_state.get('client_ip', 'unknown'),
                        details={
                            'function': func.__name__,
                            'error': str(e),
                            'duration': (datetime.now(pytz.UTC) - start_time).total_seconds()
                        },
                        status='error',
                        session_id=st.session_state.get('session_id', 'unknown')
                    ))
                raise
        return wrapper
    return decorator