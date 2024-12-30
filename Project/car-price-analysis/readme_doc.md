# Car Analysis Suite Documentation

## Project Overview
Cloud-based car analysis platform combining ML price prediction, AI chat assistance, and market analytics, optimized for AWS EC2 free tier deployment.

## System Architecture

### Core Components & Optimizations

#### 1. AI Chat Assistant (`AI_Chat_Analyst_Script.py`)
```python
class QASystem:
    def __init__(self, chunk_size=500, chunk_overlap=25):
        self.chunk_size = chunk_size  # Optimized for EC2 memory
        self.setup_memory_management()
    
    def setup_memory_management(self):
        self.batch_size = 1000
        self.cache_size = 50 * 1024 * 1024  # 50MB cache limit
```

- Memory-optimized document processing
- Efficient caching with size limits
- Batch processing for large files
- Automatic resource cleanup

#### 2. Price Prediction (`Pricing_Func.py`)
```python
class CarPricePredictor:
    def __init__(self, models=None, fast_mode=True):
        self.max_workers = 2  # Limited for EC2
        self.setup_resource_constraints()
    
    def tune_model(self, model_type, X, y):
        # Early stopping and resource limits
        return self._optimize_training(model_type, X, y)
```

- Resource-aware model training
- Dynamic batch sizing
- Optimized feature engineering
- Cached predictions

#### 3. Main Application (`Main_App.py`)
```python
class CombinedCarApp:
    def __init__(self):
        self.setup_monitoring()
        self.initialize_cloud_storage()
    
    def monitor_resources(self):
        # Resource monitoring and management
        self._check_memory_usage()
        self._optimize_storage()
```

- Cloud-based storage integration
- Resource monitoring
- Dynamic memory management
- Efficient state handling

## Performance Metrics

### Resource Usage
| Component | Memory (MB) | CPU (%) | Response Time (s) |
|-----------|------------|---------|-------------------|
| Base App   | 200       | 5-10    | -                |
| Chat System| 250-300   | 30-40   | 1-3              |
| Prediction | 300-500   | 60-70   | 2-5              |
| Analysis   | 200-300   | 20-30   | 1-2              |

### Optimizations
1. **Memory Management**
   - Batch processing: 1000 records
   - Cache limit: 50MB
   - Document chunking: 500 bytes
   - Auto cleanup threshold: 80% memory usage

2. **Processing Optimization**
   - Worker threads: 2 max
   - Query timeout: 30s
   - Cache TTL: 1 hour
   - Batch size: 1000 records

## Cloud Infrastructure

### AWS EC2 Configuration
```bash
Instance Type: t2.micro
vCPU: 1
Memory: 1GB
Storage: 8GB
Region: us-east-1
```

### Storage Strategy
1. **S3 Integration**
```python
class SecureStorageService:
    def __init__(self):
        self.setup_s3_client()
        self.configure_lifecycle_rules()
```

2. **Data Management**
- Compression for storage
- Automated cleanup
- Lifecycle policies
- Backup strategy

## Security Implementation

### Enhanced Security (`enhanced_security.py`)
```python
class EnhancedSecurityManager:
    def __init__(self):
        self.configure_cloud_security()
        self.setup_monitoring()
```

Key Features:
- Cloud-based authentication
- Resource encryption
- Audit logging
- Access control

## API Documentation

### 1. Chat System API
```python
def create_chain(
    sources: List[Dict[str, Union[str, List[str]]]], 
    batch_size: int = 1000
) -> Chain:
    """Create optimized QA chain for EC2 deployment"""
```

### 2. Price Prediction API
```python
def predict_price(
    input_data: Dict[str, Any],
    fast_mode: bool = True
) -> Dict[str, float]:
    """Generate resource-optimized price predictions"""
```

## Deployment Guide

### 1. Environment Setup
```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Application Settings
export APP_MEMORY_LIMIT=800
export CACHE_SIZE=50
export WORKER_COUNT=2
```

### 2. Application Launch
```bash
# Install dependencies
pip install -r requirements.txt

# Start application
streamlit run Main_App.py --server.maxUploadSize=50
```

## Monitoring & Maintenance

### 1. Resource Monitoring
```python
class ResourceMonitor:
    def __init__(self):
        self.setup_cloudwatch()
        self.configure_alerts()
```

- Memory usage tracking
- CPU utilization
- Storage monitoring
- Performance metrics

### 2. Maintenance Tasks
- Daily cache cleanup
- Weekly log rotation
- Monthly performance review
- Quarterly security audit

## Best Practices

### 1. Development Guidelines
- Use resource-aware coding
- Implement proper error handling
- Follow AWS best practices
- Maintain security standards

### 2. Deployment Rules
- Regular backups
- Version control
- Documentation updates
- Security patches

## Future Roadmap

### 1. Technical Improvements
- Enhanced caching
- Better compression
- Distributed processing
- Advanced monitoring

### 2. Feature Additions
- Real-time analysis
- Advanced visualizations
- API expansion
- Integration options