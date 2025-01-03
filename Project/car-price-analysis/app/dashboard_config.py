"""
Dashboard Configuration

Handles configuration settings for the visualization dashboard,
supporting both local development and cloud deployment.
"""

import os
from typing import Dict, Any
import json

class DashboardConfig:
    """Configuration management for dashboard"""
    
    def __init__(self):
        self.is_cloud = os.getenv('USE_CLOUD_STORAGE', '').lower() == 'true'
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Chart configurations
        self.chart_config = {
            'default_height': 400,
            'default_width': None,  # Use container width
            'color_scheme': 'blues',
            'template': 'plotly_white'
        }
        
        # Feature configurations
        self.feature_config = {
            'numeric_features': ['year', 'condition', 'odometer', 'sellingprice'],
            'categorical_features': ['make', 'model', 'trim', 'body', 'transmission', 
                                   'state', 'color', 'interior', 'seller']
        }
        
    @property
    def is_ec2(self) -> bool:
        """Check if running on EC2"""
        return os.getenv('AWS_EXECUTION_ENV', '').startswith('AWS_ECS')
        
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return {
            'is_cloud': self.is_cloud,
            'bucket': self.s3_bucket,
            'region': self.region
        }
        
    def get_chart_config(self) -> Dict[str, Any]:
        """Get chart configuration"""
        return self.chart_config
        
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration"""
        return self.feature_config
