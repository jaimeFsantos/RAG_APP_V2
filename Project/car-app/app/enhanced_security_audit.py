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

import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import hashlib
import logging
import json
from enum import Enum
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityMode(Enum):
    LOCAL = "local"
    EC2 = "ec2"

class SecurityConfig:
    """Security configuration with local development support"""
    def __init__(self, mode: SecurityMode = SecurityMode.LOCAL):
        self.mode = mode
        self.SESSION_DURATION = timedelta(hours=12)
        self.MAX_LOGIN_ATTEMPTS = 5
        self.LOCKOUT_DURATION = timedelta(minutes=30)
        self.FILE_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB
        self.AUDIT_RETENTION_DAYS = 30
        
        # Set up environment-specific configuration
        if self.mode == SecurityMode.LOCAL:
            self._setup_local_config()
            self.s3_client = None
            self.kms_client = None
            self.dynamodb = None
        else:
            self._setup_aws_config()
    
    def _setup_local_config(self):
        """Configure local development settings"""
        self.storage_path = os.path.join(os.getcwd(), 'local_security')
        os.makedirs(self.storage_path, exist_ok=True)
        
        config_file = os.path.join(self.storage_path, 'security_config.json')
        if not os.path.exists(config_file):
            self._create_local_config(config_file)
        
        self._load_local_config(config_file)
    
    def _create_local_config(self, config_file: str):
        """Create initial local security configuration"""
        config = {
            'admin_username': 'admin',
            'admin_password_hash': hashlib.sha256('admin'.encode()).hexdigest(),
            'encryption_key': base64.b64encode(os.urandom(32)).decode(),
            'created_at': datetime.now().isoformat(),
            'admin_password': 'admin'
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_local_config(self, config_file: str):
        """Load local security configuration"""
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        self.admin_username = config['admin_username']
        self.admin_password_hash = config['admin_password_hash']
        self.encryption_key = config['encryption_key']
    
    def _setup_aws_config(self):
        """Configure AWS integration"""
        try:
            self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
            self.s3_client = boto3.client('s3', region_name=self.aws_region)
            self.kms_client = boto3.client('kms', region_name=self.aws_region)
            self.dynamodb = boto3.resource('dynamodb', region_name=self.aws_region)
        except Exception as e:
            logger.error(f"Error initializing AWS services: {e}")
            raise

class LocalAuditLogger:
    """Local implementation of audit logging for development"""
    def __init__(self, storage_path: str):
        self.storage_path = os.path.join(storage_path, 'audit_logs')
        os.makedirs(self.storage_path, exist_ok=True)
        self.current_log_file = os.path.join(
            self.storage_path, 
            f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
    
    def log_event(self, event: AuditEvent):
        """Log audit event to local file"""
        try:
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
            
            with open(self.current_log_file, 'a') as f:
                json.dump(event_data, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error logging local audit event: {e}")
    
    def get_user_activity(self, user_id: str, start_date: datetime = None) -> List[Dict]:
        """Retrieve user activity from local logs"""
        activities = []
        log_files = sorted(os.listdir(self.storage_path))
        
        for log_file in log_files:
            if not log_file.endswith('.jsonl'):
                continue
                
            with open(os.path.join(self.storage_path, log_file), 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event['user_id'] == user_id:
                            if start_date is None or datetime.fromisoformat(event['timestamp']) >= start_date:
                                activities.append(event)
                    except json.JSONDecodeError:
                        continue
        
        return activities

class LocalStorageService:
    """Local storage service for development"""
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.storage_path = config.storage_path
        
    def store_file(self, file_data: bytes, filename: str) -> str:
        """Store file locally"""
        file_path = os.path.join(self.storage_path, filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        return file_path
        
    def get_file(self, filename: str) -> Optional[bytes]:
        """Retrieve file from local storage"""
        file_path = os.path.join(self.storage_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        return None

class EncryptionService:
    """Handles data encryption and decryption"""
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        fernet_key = os.getenv('FERNET_KEY')
        if not fernet_key:
            # Generate a key if not provided
            fernet_key = base64.b64encode(os.urandom(32)).decode()
            os.environ['FERNET_KEY'] = fernet_key
        self.fernet = Fernet(fernet_key.encode())

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
    def __init__(self, mode: SecurityMode = SecurityMode.LOCAL):
        self.config = SecurityConfig(mode=mode)
        
        # Initialize services based on mode
        if mode == SecurityMode.LOCAL:
            self.audit_logger = LocalAuditLogger(self.config.storage_path)
            self.storage_service = LocalStorageService(self.config)
        else:
            self.audit_logger = AuditLogger()
            self.storage_service = SecureStorageService()
            
        self.encryption_service = EncryptionService(self.config)
        self._setup_session_tracking()
        self.max_attempts = 5
        self.lockout_duration = timedelta(minutes=15)  # Adjusted for EC2 free tier resources
        
    def _setup_session_tracking(self):
        """Setup session tracking with more lenient timeout"""
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
            st.session_state.last_activity = datetime.now()
            st.session_state.authenticated = False

    def check_session_timeout(self) -> bool:
        """Check if session has timed out with proper initialization check"""
        if not hasattr(st.session_state, 'last_activity'):
            self._setup_session_tracking()
            return False
            
        now = datetime.now()
        # More lenient 12-hour timeout for development
        timeout_duration = timedelta(hours=12)
        
        if now - st.session_state.last_activity > timeout_duration:
            return True
            
        # Update last activity timestamp
        st.session_state.last_activity = now
        return False

    def initialize_security_components(self):
        """Initialize security components with proper session handling"""
        try:
            if 'security_manager' not in st.session_state:
                st.session_state.security_manager = self.security_manager
                
            # Setup session tracking
            self._setup_session_tracking()
            
            # Initialize security session state
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = {}
            
            logger.info("Security components initialized successfully")
                    
        except Exception as e:
            logger.error(f"Security initialization error: {str(e)}")
            st.error("Error initializing security components. Running in limited mode.")

    def can_attempt_login(self) -> bool:
        """Check if login attempts are allowed"""
        try:
            # Reset attempts if lockout period has passed
            if hasattr(st.session_state, 'next_attempt'):
                if datetime.now(pytz.UTC) >= st.session_state.next_attempt:
                    st.session_state.login_attempts = 0
                    return True
                    
            return st.session_state.login_attempts < self.max_attempts
            
        except Exception as e:
            logger.error(f"Error checking login attempts: {e}")
            return False

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            # For EC2 deployment, use environment variables
            if self.config.mode == SecurityMode.EC2:
                valid_password = os.getenv("APP_PASSWORD", "admin")  # Default for testing
                return password == valid_password
            
            # For local mode, verify against stored hash
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == stored_hash
            
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False

    def update_failed_login(self):
        """Update failed login attempts"""
        try:
            st.session_state.login_attempts += 1
            
            # Set lockout if max attempts reached
            if st.session_state.login_attempts >= self.max_attempts:
                st.session_state.next_attempt = datetime.now(pytz.UTC) + self.lockout_duration
                logger.warning(f"Account locked for {self.lockout_duration}")
                
        except Exception as e:
            logger.error(f"Error updating failed login: {e}")

    def check_session_timeout(self) -> bool:
        """Check if session has timed out"""
        try:
            if 'last_activity' not in st.session_state:
                return True
                
            timeout = timedelta(hours=8)  # Adjusted for EC2 free tier
            current_time = datetime.now(pytz.UTC)
            
            if current_time - st.session_state.last_activity > timeout:
                return True
                
            # Update last activity
            st.session_state.last_activity = current_time
            return False
            
        except Exception as e:
            logger.error(f"Error checking session timeout: {e}")
            return True
        
    def _setup_session_tracking(self):
        """Initialize session tracking variables"""
        try:
            if 'last_activity' not in st.session_state:
                st.session_state.last_activity = datetime.now(pytz.UTC)
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            if 'login_attempts' not in st.session_state:
                st.session_state.login_attempts = 0
            if 'next_attempt' not in st.session_state:
                st.session_state.next_attempt = datetime.now(pytz.UTC)
                
            # Add IP tracking if available
            if 'client_ip' not in st.session_state:
                st.session_state.client_ip = self._get_client_ip()
                
            logger.info("Session tracking initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up session tracking: {e}")
            raise

    def _get_client_ip(self) -> str:
        """Get client IP address from request headers"""
        try:
            # For EC2 deployment behind load balancer
            if 'X-Forwarded-For' in st.request.headers:
                return st.request.headers['X-Forwarded-For'].split(',')[0]
            # Direct connection
            return st.request.remote_ip
        except:
            return "unknown"

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