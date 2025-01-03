"""
AI Chat Analysis System

Implements the core QA system with RAG capabilities and
market analysis integration.

Environment:
    AWS EC2 Free Tier

Components:
    - Document processing and chunking
    - Vector store management
    - RAG implementation
    - Market analysis integration
    - Visualization generation
"""

# Core Python imports
import os
import io
import gc
import json
import hashlib
import pickle
import warnings
from datetime import datetime, timedelta
from typing import List, Union, Dict, Any, Optional
import time 


# Data manipulation and processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from joblib import Parallel, delayed
from dataclasses import dataclass

# Visualization
import plotly.graph_objects as go

# Machine Learning Interpretability
import shap

# Document processing
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# LangChain components
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Concurrency
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

# AWS SDK
import boto3
from botocore.exceptions import ClientError

# System monitoring
import psutil

# Logging
import logging

# Suppress warnings
warnings.filterwarnings('ignore')


# Configure logging with a more efficient format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add after the MarketAnalyzer class
class VisualizationGenerator:
    """Handles generation of visualizations for QA responses"""
    
    def create_visualization(self, viz_type: str, data: dict) -> go.Figure:
        """
        Create visualization based on type and data
        
        Args:
            viz_type: Type of visualization to create
            data: Data for visualization
            
        Returns:
            plotly.graph_objects.Figure
        """
        try:
            if viz_type == 'price_trends':
                return self._create_price_trend_viz(data)
            elif viz_type == 'feature_importance':
                return self._create_feature_importance_viz(data)
            elif viz_type == 'market_analysis':
                return self._create_market_analysis_viz(data)
            return None
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
            
    def _create_price_trend_viz(self, data: dict) -> go.Figure:
        """Create price trend visualization"""
        fig = go.Figure()
        
        # Add price trend line
        fig.add_trace(go.Scatter(
            x=list(data.get('dates', [])),
            y=list(data.get('prices', [])),
            mode='lines+markers',
            name='Price Trend'
        ))
        
        fig.update_layout(
            title='Price Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            template='plotly_white'
        )
        
        return fig
        
    def _create_feature_importance_viz(self, data: dict) -> go.Figure:
        """Create feature importance visualization"""
        fig = go.Figure()
        
        # Sort features by importance
        features = list(data.keys())
        importances = list(data.values())
        
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400,
            template='plotly_white'
        )
        
        return fig
        
    def _create_market_analysis_viz(self, data: dict) -> go.Figure:
        """Create market analysis visualization"""
        fig = go.Figure()
        
        # Add market segments
        if 'segments' in data:
            fig.add_trace(go.Bar(
                x=list(data['segments'].keys()),
                y=list(data['segments'].values()),
                name='Market Distribution'
            ))
            
        fig.update_layout(
            title='Market Analysis',
            xaxis_title='Segment',
            yaxis_title='Count',
            height=400,
            template='plotly_white'
        )
        
        return fig

class PreCalculationPipeline:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        self.shap_cache = SHAPCache(cache_dir=cache_dir)
        self.model_cache = {}
        self.feature_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        
    def _cache_key(self, df):
        """Generate cache key based on dataframe characteristics"""
        return hashlib.md5(
            f"{df.shape}_{list(df.columns)}_{df.index[0]}_{df.index[-1]}".encode()
        ).hexdigest()
        
    def load_cached_features(self, cache_key):
        """Load cached feature engineering results"""
        cache_file = os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def save_cached_features(self, cache_key, features_data):
        """Save feature engineering results to cache"""
        cache_file = os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(features_data, f)
            
    def preprocess_data(self, df, predictor):
        """Optimized data preprocessing with caching"""
        cache_key = self._cache_key(df)
        cached_features = self.load_cached_features(cache_key)
        
        if cached_features is not None:
            return cached_features
            
        # Process data in parallel chunks
        chunk_size = 10000
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        with ThreadPoolExecutor() as executor:
            processed_chunks = list(executor.map(predictor.prepare_data, chunks))
        
        processed_data = pd.concat(processed_chunks)
        features = predictor.engineer_features(processed_data)
        
        # Cache the results
        features_data = {
            'processed_data': processed_data,
            'features': features
        }
        self.save_cached_features(cache_key, features_data)
        
        return features_data
    
class SHAPCache:
    def __init__(self, cache_dir: str = ".cache", max_cache_size_mb: int = 500, cache_ttl_days: int = 30):
        """
        Initialize SHAP cache with size limits and TTL.
        
        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB
            cache_ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self.cache: Dict[str, str] = {}  # Maps hash to filename
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_metadata()
        self._cleanup_expired()

    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    # Convert string dates back to datetime
                    for entry in self.metadata.values():
                        entry['last_access'] = datetime.fromisoformat(entry['last_access'])
        except Exception as e:
            logging.error(f"Error loading cache metadata: {e}")
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata_copy = {}
            for key, value in self.metadata.items():
                metadata_copy[key] = value.copy()
                # Convert datetime to string for JSON serialization
                metadata_copy[key]['last_access'] = value['last_access'].isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_copy, f)
        except Exception as e:
            logging.error(f"Error saving cache metadata: {e}")

    def _cleanup_expired(self):
        """Remove expired cache entries and ensure cache size is within limits."""
        current_time = datetime.now()
        total_size = 0
        entries_to_remove = []

        # Identify expired entries and calculate total size
        for cache_hash, meta in self.metadata.items():
            if current_time - meta['last_access'] > self.cache_ttl:
                entries_to_remove.append(cache_hash)
            else:
                total_size += meta['size']

        # Remove expired entries
        for cache_hash in entries_to_remove:
            self._remove_cache_entry(cache_hash)

        # If still over size limit, remove oldest entries
        if total_size > self.max_cache_size:
            sorted_entries = sorted(
                self.metadata.items(),
                key=lambda x: x[1]['last_access']
            )
            
            for cache_hash, _ in sorted_entries:
                if total_size <= self.max_cache_size:
                    break
                total_size -= self.metadata[cache_hash]['size']
                self._remove_cache_entry(cache_hash)

    def _remove_cache_entry(self, cache_hash: str):
        """Remove a cache entry and its associated files."""
        try:
            filepath = os.path.join(self.cache_dir, f"{cache_hash}.pkl")
            if os.path.exists(filepath):
                os.remove(filepath)
            self.metadata.pop(cache_hash, None)
        except Exception as e:
            logging.error(f"Error removing cache entry {cache_hash}: {e}")

    def _generate_cache_key(self, model, input_data) -> str:
        """Generate a unique cache key based on model type and input data."""
        try:
            # Get model parameters as string
            if hasattr(model, 'get_params'):
                model_params = str(model.get_params())
            else:
                model_params = str(model.__class__.__name__)

            # Hash model parameters and input data
            key_components = [
                model_params,
                str(input_data.shape),
                hashlib.md5(input_data.values.tobytes()).hexdigest()
            ]
            
            return hashlib.sha256(''.join(key_components).encode()).hexdigest()
        except Exception as e:
            logging.error(f"Error generating cache key: {e}")
            return None

    def get(self, model, input_data) -> Optional[np.ndarray]:
        """Retrieve SHAP values from cache if available."""
        cache_key = self._generate_cache_key(model, input_data)
        if not cache_key or cache_key not in self.metadata:
            return None

        try:
            filepath = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    shap_values = pickle.load(f)
                
                # Update last access time
                self.metadata[cache_key]['last_access'] = datetime.now()
                self._save_metadata()
                
                return shap_values
        except Exception as e:
            logging.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, model, input_data, shap_values: np.ndarray):
        """Store SHAP values in cache."""
        cache_key = self._generate_cache_key(model, input_data)
        if not cache_key:
            return

        try:
            filepath = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Save SHAP values
            with open(filepath, 'wb') as f:
                pickle.dump(shap_values, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'size': os.path.getsize(filepath),
                'last_access': datetime.now()
            }
            
            self._save_metadata()
            self._cleanup_expired()
        except Exception as e:
            logging.error(f"Error storing in cache: {e}")
            
def compute_shap_values(model, input_data, cache: SHAPCache) -> np.ndarray:
    """
    Compute SHAP values with caching support.
    
    Args:
        model: Trained model
        input_data: Input data for SHAP analysis
        cache: SHAPCache instance
    
    Returns:
        numpy.ndarray: SHAP values
    """
    # Try to get from cache first
    shap_values = cache.get(model, input_data)
    if shap_values is not None:
        return shap_values

    try:
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # Store in cache
        cache.set(model, input_data, shap_values)
        
        return shap_values
    except Exception as e:
        logging.error(f"Error computing SHAP values: {e}")
        raise



#####################################################################################################################################    
class DocumentLoader:
    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.ocr_failures = []  # Track OCR failures
        
    @staticmethod
    def process_image(image_bytes, dpi=300):
        """Optimized image processing for OCR with enhanced error handling"""
        if not image_bytes:
            return ""
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale for better OCR
            image = image.convert('L')
            
            # Apply image enhancement techniques
            # Increase contrast for better text recognition
            image = image.point(lambda x: 0 if x < 128 else 255, '1')
            
            # Multiple OCR attempts with different configurations
            configs = [
                '--dpi 300 --oem 3 --psm 6',  # Default
                '--dpi 300 --oem 3 --psm 1',  # Automatic page segmentation
                '--dpi 300 --oem 1 --psm 6'   # Legacy engine
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if text.strip():  # If we got meaningful text, return it
                        return text
                except Exception:
                    continue
            
            return ""  # Return empty string if all attempts fail
            
        except Exception as e:
            # Don't log the error, just return empty string
            return ""
    
    def load_pdf_page(self, args):
        """Process a single PDF page with enhanced error handling"""
        page, file_path = args
        try:
            # First attempt: direct text extraction
            text = page.get_text()
            
            # If no text found, try OCR
            if not text.strip():
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    text = self.process_image(pix.tobytes())
                except Exception:
                    # If OCR fails, return empty string without logging
                    text = ""
            
            return text.strip(), page.number
            
        except Exception:
            # If page processing fails entirely, return empty result
            return "", page.number
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF with enhanced memory management and error handling"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        documents = []
        doc = None
        logger.info(f"Opening PDF file: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"Successfully opened PDF with {total_pages} pages")
            
            # Process pages in smaller batches to manage memory
            batch_size = 5  # Reduced batch size for EC2 free tier
            for i in range(0, total_pages, batch_size):
                batch_pages = list(range(i, min(i + batch_size, total_pages)))
                logger.info(f"Processing batch of pages {i+1} to {min(i + batch_size, total_pages)}")
                
                # Process each page in the batch
                for page_num in batch_pages:
                    try:
                        page = doc[page_num]
                        text = page.get_text()
                        
                        if not text.strip():
                            logger.info(f"No text found on page {page_num + 1}, attempting OCR")
                            try:
                                # Reduced resolution for memory efficiency
                                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                                img_data = pix.tobytes()
                                pix = None  # Clear pixmap from memory
                                
                                if img_data:
                                    text = self.process_image(img_data)
                            except Exception as e:
                                logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                                continue
                        
                        if text.strip():
                            documents.append(
                                Document(
                                    page_content=text.strip(),
                                    metadata={
                                        "source": file_path,
                                        "page": page_num,
                                        "extraction_method": "text" if text else "ocr"
                                    }
                                )
                            )
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                    
                    # Clear memory after each page
                    gc.collect()
                
                logger.info(f"Completed batch. Documents extracted: {len(documents)}")
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
        finally:
            if doc:
                doc.close()
                doc = None
            gc.collect()
        
        return documents

    def load_csv(
        self, 
        file_path: str, 
        text_columns: Union[List[str], None] = None,
        batch_size: int = 1000,
        max_rows: int = None
    ) -> List[Document]:
        """
        Load a CSV file efficiently by batching and smart text combination.
        
        Args:
            file_path: Path to the CSV file
            text_columns: Specific columns to include
            batch_size: Number of rows to process at once
            max_rows: Maximum number of rows to process (None for all)
        """
        try:
            # Read CSV in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=batch_size):
                if text_columns:
                    chunk = chunk[text_columns]
                
                # Convert categorical columns to category type for memory efficiency
                for col in chunk.select_dtypes(include=['object', 'int64', 'float']).columns:
                    chunk[col] = chunk[col]
                chunks.append(chunk)
                
                if max_rows and len(chunks) * batch_size >= max_rows:
                    break
            
            # Combine chunks
            data = pd.concat(chunks)
            if max_rows:
                data = data.head(max_rows)
            
            # Group similar records together to reduce redundancy
            documents = []
            
            # If we have specific columns that define groups (like make/model for cars)
            if 'make' in data.columns and 'model' in data.columns:
                # Group by make and model
                grouped = data.groupby(['make', 'model'])
                
                for (make, model), group in grouped:
                    # Create a summary for each make/model combination
                    summary = f"Make: {make}\nModel: {model}\n"
                    
                    # Add aggregate information
                    summary += f"Number of vehicles: {len(group)}\n"
                    
                    # Add unique values for important fields
                    for col in group.columns:
                        if col not in ['make', 'model']:
                            unique_values = group[col].unique()
                            if len(unique_values) <= 10:  # Only include if not too many unique values
                                values_str = ', '.join(str(v) for v in unique_values if str(v) != 'nan')
                                summary += f"{col}: {values_str}\n"
                    
                    documents.append(Document(
                        page_content=summary,
                        metadata={
                            "source": file_path,
                            "make": make,
                            "model": model,
                            "row_count": len(group)
                        }
                    ))
            else:
                # For other types of CSVs, batch rows together
                for i in range(0, len(data), 50):  # Process 50 rows at a time
                    batch = data.iloc[i:i+50]
                    
                    # Create a summary of the batch
                    summary = f"Batch {i//50 + 1} Summary:\n"
                    
                    # Add column summaries
                    for col in batch.columns:
                        unique_values = batch[col].unique()
                        if len(unique_values) <= 10:
                            values_str = ', '.join(str(v) for v in unique_values if str(v) != 'nan')
                            summary += f"{col}: {values_str}\n"
                        else:
                            summary += f"{col}: {len(unique_values)} unique values\n"
                    
                    documents.append(Document(
                        page_content=summary,
                        metadata={
                            "source": file_path,
                            "batch_number": i//50 + 1,
                            "row_count": len(batch)
                        }
                    ))
            
            logger.info(f"Created {len(documents)} document chunks from CSV")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
            
class CarPricePredictor:
    def __init__(self, models=None, fast_mode=True, max_samples=None):
        self.shap_cache = SHAPCache()
        self.scaler = StandardScaler()
        self.best_models = {}
        self.fast_mode = fast_mode
        self.max_samples = max_samples
        self.feature_columns = None
        self.is_trained = False
        self.metrics = {}
        self.unique_values = {}
        
        # Use EFS for model persistence
        self.model_path = os.getenv('MODEL_PATH', '/mnt/efs/models')
        os.makedirs(self.model_path, exist_ok=True)
        
        self._initialize_models(models)
        self._initialize_param_grids()
    def _initialize_models(self, models):
        self.available_models = {
            'ridge': {'speed': 1, 'name': 'Ridge Regression'},
            'lasso': {'speed': 2, 'name': 'Lasso Regression'},
            'gbm': {'speed': 3, 'name': 'Gradient Boosting'},
            'rf': {'speed': 4, 'name': 'Random Forest'},
            'xgb': {'speed': 5, 'name': 'XGBoost'}
        }
        self.selected_models = models if models else list(self.available_models.keys())
        
    def _initialize_param_grids(self):
        # Define more targeted parameter grids
        self.param_grids = {
            'regular': {
                'rf': {
                    'n_estimators': [100, 200],  # Reduced options
                    'max_depth': [10, 20, None],  # More focused depth options
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                },
                'gbm': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],  # More focused learning rates
                    'max_depth': [4, 6],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'xgb': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [4, 6],
                    'min_child_weight': [3],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
            },
            'fast': {
                'rf': {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'gbm': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2]
                },
                'xgb': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [5],
                    'min_child_weight': [3],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8]
                }
            }
        }

    def tune_model(self, model_type, X, y):
        """Optimized model tuning with early stopping and incremental search"""
        param_grid = self.get_param_grid(model_type)
        if not param_grid:
            return None

        # Initialize base model with early stopping
        if model_type == 'rf':
            base_model = RandomForestRegressor(
                random_state=42,
                n_jobs=2 if not self.fast_mode else 1,
                warm_start=True  # Enable warm start for incremental training
            )
        elif model_type == 'gbm':
            base_model = GradientBoostingRegressor(
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=5,  # Early stopping
                tol=1e-4
            )
        else:
            return None

        # Implement randomized pre-search to identify promising regions
        n_pre_iter = 5 if self.fast_mode else 10
        pre_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_pre_iter,
            cv=3 if self.fast_mode else 5,
            scoring='neg_mean_squared_error',
            n_jobs=2,
            random_state=42
        )
        pre_search.fit(X, y)

        # Refine param_grid based on pre-search results
        best_params = pre_search.best_params_
        refined_param_grid = {}
        for param, value in best_params.items():
            if param in param_grid:
                orig_values = param_grid[param]
                if isinstance(value, (int, float)):
                    # Create a focused range around the best value
                    if len(orig_values) > 1:
                        step = (max(orig_values) - min(orig_values)) / (len(orig_values) - 1)
                        refined_values = [value - step, value, value + step]
                        refined_values = [v for v in refined_values if min(orig_values) <= v <= max(orig_values)]
                        refined_param_grid[param] = refined_values
                    else:
                        refined_param_grid[param] = [value]
                else:
                    refined_param_grid[param] = [value]

        # Final grid search with refined parameters
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=refined_param_grid,
            cv=3 if self.fast_mode else 5,
            scoring='neg_mean_squared_error',
            n_jobs=2,
            verbose=0
        )

        # Implement early stopping callback if supported
        if hasattr(base_model, 'n_iter_no_change'):
            base_model.n_iter_no_change = 5
            base_model.tol = 1e-4

        try:
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error during GridSearchCV for {model_type}: {str(e)}")
            return None

    def fit(self, X, y):
        """Optimized model fitting with parallel processing and memory management"""
        self.feature_columns = X.columns.tolist()

        def train_model(model_type):
            try:
                if model_type in ['rf', 'gbm', 'xgb']:
                    # Use smaller data sample for initial tuning
                    sample_size = min(10000, len(X))
                    X_sample = X.sample(n=sample_size, random_state=42)
                    y_sample = y[X_sample.index]
                    
                    model = self.tune_model(model_type, X_sample, y_sample)
                    
                    if model:
                        # Fit the tuned model on full dataset
                        model.fit(X, y)
                    return model_type, model
                elif model_type == 'lasso':
                    return model_type, LassoCV(cv=3 if self.fast_mode else 5, random_state=42).fit(X, y)
                elif model_type == 'ridge':
                    return model_type, RidgeCV(cv=3 if self.fast_mode else 5).fit(X, y)
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                return model_type, None

        # Train models in parallel with memory management
        with ThreadPoolExecutor(max_workers=min(len(self.selected_models), 4)) as executor:
            results = list(executor.map(lambda m: train_model(m), self.selected_models))

        self.best_models = {name: model for name, model in results if model is not None}

        # Create ensemble if multiple models are available
        if len(self.best_models) > 1:
            self.ensemble = VotingRegressor([
                (name, model) for name, model in self.best_models.items()
            ])
            self.ensemble.fit(X, y)

        self.is_trained = True
        
        # Clean up memory
        gc.collect()

    def update_unique_values(self, df):
        def safe_sort(values):
            cleaned_values = [str(x) for x in values if pd.notna(x)]
            return sorted(cleaned_values)
        
        self.unique_values = {
            'state': safe_sort(df['state'].unique()),
            'body': safe_sort(df['body'].unique()),
            'transmission': safe_sort(df['transmission'].unique()),
            'color': safe_sort(df['color'].unique()),
            'interior': safe_sort(df['interior'].unique())
        }

    def remove_outliers(self, df, threshold=1.5):
        initial_rows = len(df)
        
        Q1 = df['sellingprice'].quantile(0.25)
        Q3 = df['sellingprice'].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (
            (df['sellingprice'] < (Q1 - threshold * IQR)) | 
            (df['sellingprice'] > (Q3 + threshold * IQR))
        )
        
        df_cleaned = df[~outlier_condition]
    
        return df_cleaned

    def prepare_data(self, df):
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)
        
        string_columns = ['state', 'body', 'transmission', 'color', 'interior', 'trim', 'sell']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self.update_unique_values(df)
        
        drop_cols = ['datetime', 'Day of Sale', 'Weekend', 'vin', 'make', 'model','saledate']
        df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        
        fill_values = {
            'transmission': 'unknown',
            'interior': 'unknown',
            'color': 'unknown',
            'body': 'unknown',
            'interior': 'unknown',
            'trim': 'unknown',
            'state': 'unknown',
            'condition': df['condition'].median(),
            'odometer': df['odometer'].median()
        }
        
        for col, fill_value in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        
        df = df[df['sellingprice'] > 0]
        df = pd.DataFrame(df)
        df.dropna(inplace=True)
        
        df= self.remove_outliers(df)
        
        return df

    def engineer_features(self, df):
        self.original_features = df.copy()
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['sellingprice', 'mmr']]
        
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
        
        key_numeric = ['odometer', 'year', 'condition']
        for col in key_numeric:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        categorical_cols = ['body', 'transmission', 'state', 'color', 'interior', 'trim', 'seller']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        df['age_miles'] = df['vehicle_age'] * df['odometer']
        df['age_squared'] = df['vehicle_age'] ** 2
        
        return df

    def remove_multicollinearity(self, X, threshold=0.95):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            return X.drop(columns=to_drop)
        return X
    
    def _more_tags(self):
        return {"requires_positive_y": False}
    
    def get_param_grid(self, model_type):
        param_grid = self.param_grids['fast' if self.fast_mode else 'regular'].get(model_type)
        if not param_grid:
            raise ValueError(f"No parameter grid found for model type: {model_type}")
        return param_grid


    def tune_model(self, model_type, X, y):
#        st.write(f"Starting tune_model for: {model_type}")  # Debugging model type
        param_grid = self.get_param_grid(model_type) #Call function
#        st.write(f"Parameter grid for {model_type}: {param_grid}")  # Debugging param_grid

        if not param_grid:
#            st.error(f"No parameter grid defined for model type: {model_type}")
            return None

        if model_type == 'rf':
            base_model = RandomForestRegressor(random_state=42, n_jobs=2 if not self.fast_mode else 1)
        elif model_type == 'gbm':
            base_model = GradientBoostingRegressor(random_state=42)
        #elif model_type == 'xgb':
        #    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1 if not self.fast_mode else 1)
        else:
#            st.error(f"Unknown model type: {model_type}")
            return None

#        st.write(f"Base model for {model_type}: {base_model}")  # Debugging base_model

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            n_jobs=2,
            cv=3 if self.fast_mode else 5,
            scoring='neg_mean_squared_error',
            verbose=1
        )

        try:
            grid_search.fit(X, y)
#            st.write(f"Best estimator for {model_type}: {grid_search.best_estimator_}")  # Debugging best estimator
            return grid_search.best_estimator_
        except Exception as e:
#            st.error(f"Error during GridSearchCV for {model_type}: {e}")
            return None


    def fit(self, X, y):
        self.feature_columns = X.columns.tolist()

        def train_model(model_type):
            try:
                if model_type in ['rf', 'gbm', 'xgb']:
                    return model_type, self.tune_model(model_type, X, y)
                elif model_type == 'lasso':
                    return model_type, LassoCV(cv=3 if self.fast_mode else 5, random_state=42).fit(X, y)
                elif model_type == 'ridge':
                    return model_type, RidgeCV(cv=3 if self.fast_mode else 5).fit(X, y)
            except Exception as e:
#                st.error(f"Error training {model_type}: {e}")
                return model_type, None

        results = Parallel(n_jobs=1)(delayed(train_model)(model) for model in self.selected_models)
        self.best_models = {name: model for name, model in results if model is not None}

        if len(self.best_models) > 1:
            self.ensemble = VotingRegressor([
                (name, model) for name, model in self.best_models.items()
            ])
            self.ensemble.fit(X, y)

        self.is_trained = True


    def evaluate(self, X, y):
        metrics = {}
        predictions = {}
        
        for name, model in self.best_models.items():
            pred = model.predict(X)
            predictions[name] = pred
            metrics[name] = {
                'r2': r2_score(y, pred),
                'rmse': np.sqrt(mean_squared_error(y, pred)),
                'mape': mean_absolute_percentage_error(y, pred)
            }
        
        if len(self.best_models) > 1:
            ensemble_pred = self.ensemble.predict(X)
            predictions['ensemble'] = ensemble_pred
            metrics['ensemble'] = {
                'r2': r2_score(y, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
                'mape': mean_absolute_percentage_error(y, ensemble_pred)
            }
        
        self.metrics = metrics
        self.predictions = predictions
        return metrics, predictions

    def prepare_prediction_data(self, input_data):
        """Prepare input data for prediction"""
        df = pd.DataFrame([input_data])
        
        df['vehicle_age'] = 2024 - df['year']
        df['age_miles'] = df['vehicle_age'] * df['odometer']
        df['age_squared'] = df['vehicle_age'] ** 2
        
        numeric_cols = ['odometer', 'year', 'condition']
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log(df[col])
            df[f'{col}_squared'] = df[col] ** 2
        
        categorical_cols = ['body', 'transmission', 'state', 'color', 'interior']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        missing_cols = set(self.feature_columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
            
        df_encoded = df_encoded[self.feature_columns]
        
        return df_encoded

    def create_what_if_prediction(self, input_data):
        if not self.is_trained or self.feature_columns is None:
            raise ValueError("Model must be trained before making predictions.")
        
        df_encoded = self.prepare_prediction_data(input_data)
        
        X_scaled = self.scaler.transform(df_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        predictions = []
        model_predictions = {}
        for model_name, model in self.best_models.items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            model_predictions[model_name] = pred
        
        if len(self.best_models) > 1:
            ensemble_pred = self.ensemble.predict(X_scaled)[0]
            predictions.append(ensemble_pred)
            model_predictions['ensemble'] = ensemble_pred
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        confidence_interval = (
            mean_pred - (1.96 * std_pred),
            mean_pred + (1.96 * std_pred)
        )
        
        mape = np.mean([metrics['mape'] for metrics in self.metrics.values()])
        prediction_interval = (
            mean_pred * (1 - mape),
            mean_pred * (1 + mape)
        )
        
        return {
            'predicted_price': mean_pred,
            'confidence_interval': confidence_interval,
            'prediction_interval': prediction_interval,
            'std_dev': std_pred,
            'model_predictions': model_predictions,
            'mape': mape
        }

    def analyze_shap_values(self, X_test):
        """
        Generate SHAP analysis summary for the model with caching.
        """
        try:
            if 'rf' not in self.best_models:
                return "SHAP analysis unavailable - Random Forest model not trained"
        
            model = self.best_models['rf']
            shap_values = compute_shap_values(model, X_test, self.shap_cache)
        
            
            # Calculate mean absolute SHAP values for feature importance
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            # Create a summary string of top features
            top_features = feature_importance.head(5)
            summary = "Top 5 most important features:\n"
            for _, row in top_features.iterrows():
                summary += f"- {row['feature']}: {row['importance']:.4f}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            return "SHAP analysis failed"

    
class DocumentTracer:
    def __init__(self):
        self.trace_history = {}
        
    def trace_documents(self, question: str, retrieved_docs: List[Document]) -> Dict:
        """Track which documents were used to answer each question"""
        doc_sources = []
        for doc in retrieved_docs:
            source = {
                'source': doc.metadata.get('source', 'unknown'),
                'type': doc.metadata.get('type', 'unknown'),
                'content_preview': doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            }
            doc_sources.append(source)
            
        self.trace_history[question] = doc_sources
        return doc_sources

    def get_trace(self, question: str) -> List[Dict]:
        """Retrieve the document trace for a specific question"""
        return self.trace_history.get(question, [])
        
    def print_trace(self, question: str):
        """Print the document trace in a readable format"""
        traces = self.get_trace(question)
        print(f"\nDocuments used for question: {question}")
        print("-" * 50)
        for idx, trace in enumerate(traces, 1):
            print(f"\nDocument {idx}:")
            print(f"Source: {trace['source']}")
            print(f"Type: {trace['type']}")
            print(f"Content Preview: {trace['content_preview']}")
            print("-" * 30)


class MarketAnalyzer:
    def __init__(self, model_years=range(1992, 2025)):
        self.model_years = model_years
        self.segment_analysis = {}
        self.feature_preferences = {}
        self.market_trends = {}
        self.data = None

    def analyze_feature_impact(self, df):
        """
        Analyze the impact of various features on price and marketability.
        """
        feature_cols = ['color', 'interior', 'transmission', 'body']
        feature_analysis = {}
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
                
            feature_stats = []
            
            for make_model, group in df.groupby(['make', 'model']):
                if len(group) < 50:  # Skip groups with insufficient data
                    continue
                    
                feature_impact = {}
                baseline_price = group['sellingprice'].median()
                
                for value in group[feature].unique():
                    subset = group[group[feature] == value]
                    if len(subset) < 10:
                        continue
                        
                    median_price = subset['sellingprice'].median()
                    price_premium = (median_price - baseline_price) / baseline_price
                    
                    feature_impact[value] = {
                        'price_premium': price_premium,
                        'sample_size': len(subset),
                        'median_price': median_price
                    }
                
                if feature_impact:
                    feature_stats.append({
                        'make_model': '_'.join(make_model),
                        'impacts': feature_impact
                    })
            
            feature_analysis[feature] = feature_stats
        
        return feature_analysis

    def generate_market_insights(self, df, segment=None):
        """
        Generate market insights with integrated feature analysis.
        """
        if segment:
            df = df[df['make'].str.cat(df['model'], sep='_') == segment]
        
        insights = {
            'feature_impact': self.analyze_feature_impact(df),
            'market_summary': {
                'median_price': df['sellingprice'].median(),
                'price_range': (df['sellingprice'].quantile(0.25), df['sellingprice'].quantile(0.75)),
                'popular_colors': df['color'].value_counts().head(5).to_dict(),
                'popular_interiors': df['interior'].value_counts().head(5).to_dict(),
                'transmission_split': df['transmission'].value_counts(normalize=True).to_dict()
            }
        }
        
        return insights

    def optimize_feature_combinations(self, df, target_make_model=None):
        """
        Find optimal feature combinations, maintained for QA system compatibility.
        """
        if target_make_model:
            df = df[df['make'].str.cat(df['model'], sep='_') == target_make_model]
        
        combinations = []
        
        features = ['color', 'interior', 'transmission']
        for color in df['color'].unique():
            for interior in df['interior'].unique():
                for transmission in df['transmission'].unique():
                    subset = df[
                        (df['color'] == color) &
                        (df['interior'] == interior) &
                        (df['transmission'] == transmission)
                    ]
                    
                    if len(subset) >= 10:
                        combinations.append({
                            'color': color,
                            'interior': interior,
                            'transmission': transmission,
                            'median_price': subset['sellingprice'].median(),
                            'sample_size': len(subset),
                            'price_percentile': subset['sellingprice'].median() / df['sellingprice'].median()
                        })
        
        return pd.DataFrame(combinations) if combinations else pd.DataFrame()
    
class BalancedRetriever:
    def __init__(self, base_retriever, min_docs_per_type=2):
        self.base_retriever = base_retriever
        self.min_docs_per_type = min_docs_per_type
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Initialize empty lists for different document types
        model_docs = []
        market_docs = []
        final_docs = []
        
        # Keep retrieving documents until we have enough of each type
        docs_needed = True
        k = 20  # Start with 20 documents
        max_attempts = 5  # Limit the number of attempts to prevent infinite loops
        attempt = 0
        
        while docs_needed and attempt < max_attempts:
            # Get a batch of documents
            current_docs = self.base_retriever.get_relevant_documents(query, k=k)
            
            # Categorize documents
            for doc in current_docs:
                if doc.metadata.get('source') == 'model_analysis' and len(model_docs) < self.min_docs_per_type:
                    if doc not in model_docs:
                        model_docs.append(doc)
                elif doc.metadata.get('type', '').startswith('market_') and len(market_docs) < self.min_docs_per_type:
                    if doc not in market_docs:
                        market_docs.append(doc)
            
            # Check if we have enough documents
            if len(model_docs) >= self.min_docs_per_type and len(market_docs) >= self.min_docs_per_type:
                docs_needed = False
            else:
                # Increase k for next attempt
                k *= 2
                attempt += 1
        
        # Add minimum required documents from each type
        final_docs.extend(model_docs[:self.min_docs_per_type])
        final_docs.extend(market_docs[:self.min_docs_per_type])
        
        # Add remaining relevant documents up to a reasonable total (e.g., 10)
        remaining_slots = 10 - len(final_docs)
        if remaining_slots > 0:
            unused_docs = [doc for doc in current_docs if doc not in final_docs]
            final_docs.extend(unused_docs[:remaining_slots])
        
        # Log the document distribution for debugging
        print(f"\nDocument distribution in retrieval:")
        print(f"Model analysis documents: {len([d for d in final_docs if d.metadata.get('source') == 'model_analysis'])}")
        print(f"Market analysis documents: {len([d for d in final_docs if d.metadata.get('type', '').startswith('market_')])}")
        print(f"Total documents: {len(final_docs)}")
        
        return final_docs

class QASystem(MarketAnalyzer):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 25,  memory_limit_mb: int = 512):
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        super().__init__()  # Initialize MarketAnalyzer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = DocumentLoader()
        self.vector_db = None
        self.predictor = CarPricePredictor(models=['rf', 'gbm'], fast_mode=True)
        self.predictor_analysis = None
        self.predictor_context = None
        self.feature_analysis = None
        self.market_insights = None
        self.data_df = None
        self.document_tracer = DocumentTracer()
        self.chain = None  # Initialize chain as None
    def _check_memory(self):
        """Monitor memory usage and trigger cleanup if needed"""
        if psutil.Process().memory_info().rss > self.memory_limit:
            gc.collect()
            return False
        return True
        
    def _determine_visualization_type(self, query: str) -> str:
        """
        Determine the type of visualization needed based on the query.
        
        Args:
            query (str): The user's query string
            
        Returns:
            str: Visualization type ('price_trends', 'feature_importance', 'market_analysis', or None)
        """
        keywords = {
            'price_trends': ['trend', 'price', 'cost', 'value', 'historical'],
            'feature_importance': ['feature', 'factor', 'impact', 'influence'],
            'market_analysis': ['market', 'segment', 'compare', 'analysis']
        }
        
        query = query.lower()
        for viz_type, words in keywords.items():
            if any(word in query for word in words):
                return viz_type
        return None

    def generate_visualization(self, query: str, viz_type: str) -> Optional[Dict]:
        """
        Generate visualization data based on query type and context.
        
        Args:
            query (str): The user's query
            viz_type (str): Type of visualization to generate
            
        Returns:
            Optional[Dict]: Visualization data if available, None otherwise
        """
        try:
            viz_generator = VisualizationGenerator()
            
            if viz_type == 'price_trends':
                if self.data_df is not None:
                    data = self.data_df.copy()
                    if 'saledate' in data.columns:
                        data['date'] = pd.to_datetime(data['saledate'])
                        trends = data.groupby(data['date'].dt.strftime('%Y-%m'))[['sellingprice']].mean()
                        viz_data = trends.to_dict()['sellingprice']
                        return {
                            'type': 'price_trends',
                            'plot': viz_generator.create_price_trends_viz(viz_data)
                        }
                        
            elif viz_type == 'feature_importance':
                if hasattr(self, 'predictor') and hasattr(self.predictor, 'best_models'):
                    if 'rf' in self.predictor.best_models:
                        model = self.predictor.best_models['rf']
                        importance = dict(zip(
                            self.predictor.feature_columns,
                            model.feature_importances_
                        ))
                        return {
                            'type': 'feature_importance',
                            'plot': viz_generator.create_feature_importance_viz(importance)
                        }
                        
            elif viz_type == 'market_analysis':
                if hasattr(self, 'market_insights'):
                    market_data = self.market_insights.get('market_summary', {})
                    return {
                        'type': 'market_analysis',
                        'plot': viz_generator.create_market_analysis_viz(market_data)
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None
    def process_predictor_outputs(self, prediction_result: Dict[str, Any]):
        """
        Process and store predictor outputs for context enhancement.
        
        Args:
            prediction_result: Dictionary containing prediction results
        """
        try:
            if prediction_result:
                context = []
                
                # Format prediction details
                context.append(f"""
                Price Prediction Analysis:
                - Predicted Price: ${prediction_result['predicted_price']:,.2f}
                - Confidence Range: ${prediction_result['prediction_interval'][0]:,.2f} to ${prediction_result['prediction_interval'][1]:,.2f}
                - Model Confidence: {(1 - prediction_result['mape']) * 100:.1f}%
                """)
                
                # Create document from context
                doc = Document(
                    page_content="\n".join(context),
                    metadata={"source": "predictor_output", "type": "prediction_analysis"}
                )
                
                # Store predictor context
                self.predictor_context = "\n".join(context)
                
                # Update vector store with new document if it exists
                if self.vector_db:
                    self.vector_db.add_documents([doc])
                    logger.info("Added predictor outputs to vector store")
                    
        except Exception as e:
            logger.error(f"Error processing predictor outputs: {e}")

    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Initialize embedding model
            self.embedding_model = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            
            # Initialize LLM
            self.llm = ChatOllama(model="mistral")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
        
    def process_sources(self, sources: List[Dict[str, Union[str, List[str]]]]) -> List[Document]:
        gc.collect()
        all_documents = []
        self.csv_file_path = None
        
        for source in sources:
            file_path = source["path"]
            file_type = source["type"].lower()
            
            try:
                if file_type == "csv":
                    self.csv_file_path = file_path
                    logger.info(f"Processing CSV file: {file_path}")
                    
                    # Read the CSV file
                    self.data_df = pd.read_csv(file_path)
                    logger.info(f"Successfully read CSV with shape: {self.data_df.shape}")
                    
                    # Create CSV documents
                    text_columns = source.get("columns", None)
                    csv_documents = self.loader.load_csv(file_path, text_columns)
                    all_documents.extend(csv_documents)
                    
                    # Generate market insights
                    self.market_insights = self.generate_market_insights(self.data_df)
                    
                    try:
                        market_docs = self._create_market_analysis_documents()
                        if market_docs:
                            logger.info(f"Created {len(market_docs)} market analysis documents")
                            all_documents.extend(market_docs)
                        else:
                            logger.warning("No market analysis documents were created")
                    except Exception as e:
                        logger.error(f"Error creating market analysis documents: {str(e)}")
                    
                    # Process data for price prediction
                    processed_data = self.predictor.prepare_data(self.data_df)
                    processed_data = processed_data.sample(frac=0.01, random_state=42)
                    features = self.predictor.engineer_features(processed_data)
                    X = features.drop('sellingprice', axis=1)
                    y = features['sellingprice']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    self.predictor.fit(X_train, y_train)
                    metrics, _ = self.predictor.evaluate(X_test, y_test)
                    
                    # Generate predictor context
                    performance_summary = self._generate_performance_summary(metrics)
                    shap_summary = self.analyze_shap_values(X_test)
                    self.predictor_context = f"{performance_summary}\n\nFeature Importance Analysis:\n{shap_summary}"
                    
                    predictor_doc = Document(
                        page_content=self.predictor_context,
                        metadata={"source": "model_analysis", "type": "predictor_context"}
                    )
                    if predictor_doc:
                        logger.info("Predictor context document created")
                    all_documents.append(predictor_doc)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                
        return all_documents
    
    def _create_market_analysis_documents(self) -> List[Document]:
        """Create documents from market analysis insights with improved error handling"""
        market_docs = []
        
        try:
            # Feature impact analysis document
            if 'feature_impact' in self.market_insights:
                feature_summary = "Feature Impact Analysis:\n"
                for feature, analysis in self.market_insights['feature_impact'].items():
                    if not analysis:  # Skip empty analyses
                        continue
                        
                    feature_summary += f"\n{feature.upper()} Impact:\n"
                    for segment in analysis[:5]:  # Top 5 segments
                        if not isinstance(segment, dict):  # Type check
                            continue
                            
                        feature_summary += f"\nMake/Model: {segment.get('make_model', 'Unknown')}\n"
                        impacts = segment.get('impacts', {})
                        for value, impact in impacts.items():
                            if not isinstance(impact, dict):  # Type check
                                continue
                                
                            feature_summary += (
                                f"- {value}:\n"
                                f"  Price Premium: {impact.get('price_premium', 0):.2%}\n"
                                f"  Selling Speed Advantage: {impact.get('selling_speed_advantage', 0):.1f} days\n"
                            )
                
                if feature_summary != "Feature Impact Analysis:\n":  # Only add if we have content
                    market_docs.append(Document(
                        page_content=feature_summary,
                        metadata={"source": "market_analysis", "type": "feature_impact"}
                    ))
            
            # Market summary document
            if 'market_summary' in self.market_insights:
                summary = self.market_insights['market_summary']
                if isinstance(summary, dict):  # Type check
                    summary_text = "Market Overview:\n"
                    
                    # Add basic statistics with safe gets
                    summary_text += f"Median Price: ${summary.get('median_price', 0):,.2f}\n"
                    
                    price_range = summary.get('price_range', (0, 0))
                    if isinstance(price_range, tuple) and len(price_range) == 2:
                        summary_text += f"Price Range: ${price_range[0]:,.2f} - ${price_range[1]:,.2f}\n"
                    
                    # Add popular features with safe gets
                    popular_colors = summary.get('popular_colors', {})
                    if popular_colors:
                        summary_text += f"\nPopular Colors:\n"
                        for color, count in popular_colors.items():
                            summary_text += f"- {color}: {count} vehicles\n"
                    
                    market_docs.append(Document(
                        page_content=summary_text,
                        metadata={"source": "market_analysis", "type": "market_summary"}
                    ))
        
        except Exception as e:
            logger.error(f"Error in _create_market_analysis_documents: {str(e)}")
        
        return market_docs
    def _generate_performance_summary(self, metrics: Dict) -> str:
        """Generate a formatted performance summary string"""
        summary = "Model Performance Summary:\n"
        for model_name, model_metrics in metrics.items():
            summary += f"\n{model_name}:\n"
            for metric_name, value in model_metrics.items():
                summary += f"- {metric_name}: {value:.4f}\n"
        return summary

    def analyze_shap_values(self, X_test):
        """Generate SHAP analysis summary for the model"""
        try:
            if 'rf' not in self.predictor.best_models:
                return "SHAP analysis unavailable - Random Forest model not trained"
            
            # Create explainer for Random Forest model
            explainer = shap.TreeExplainer(self.predictor.best_models['rf'])
            
            # Calculate SHAP values for a sample of test data
            # Use a smaller sample size for performance
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values for feature importance
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            # Create a summary string of top features
            top_features = feature_importance.head(5)
            summary = "Top 5 most important features:\n"
            for _, row in top_features.iterrows():
                summary += f"- {row['feature']}: {row['importance']:.4f}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            return "SHAP analysis failed"
        
    def ask(self, query: str) -> str:
        """
        Wrapper method for chain invocation with better error handling
        
        Args:
            query (str): The question to be answered
            
        Returns:
            str: Response from the QA system
        """
        try:
            # First validate chain exists
            if not hasattr(self, 'chain') or self.chain is None:
                # Try to initialize chain if not done
                if hasattr(self, 'vector_db') and self.vector_db is not None:
                    template = """Question: {question}
                    Context: {context}
                    Please provide a comprehensive response."""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    llm = ChatOllama(model="mistral")
                    retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
                    
                    def get_context(question):
                        docs = retriever.get_relevant_documents(question)
                        return "\n".join(doc.page_content for doc in docs)
                    
                    self.chain = (
                        {"context": get_context, "question": RunnablePassthrough()}
                        | prompt 
                        | llm 
                        | StrOutputParser()
                    )
                else:
                    logger.error("Vector store not initialized")
                    return "System not ready. Please ensure documents are loaded first."
                    
            if self.chain is None:
                logger.error("Chain initialization failed")
                return "System not ready. Please initialize first."
                    
            response = self.chain.invoke(query)
            return response
                
        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            return f"Error processing query: {str(e)}"

    def create_chain(self, sources: List[Dict[str, Union[str, List[str]]]]):
        """Create QA chain with robust error handling and storage management"""
        try:
            if not sources:
                logger.error("No sources provided")
                return None 
                
            # Process documents first before initializing components
            documents = self.process_sources(sources)
            if not documents:
                logger.error("No documents could be processed from sources")
                return None
                
            # Initialize components with retries and proper configuration
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Initialize without timeout parameter
                    embedding_model = OllamaEmbeddings(
                        model="nomic-embed-text"
                    )
                    # For ChatOllama, use temperature instead of timeout
                    llm = ChatOllama(
                        model="mistral",
                        temperature=0.7
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to initialize components after {max_retries} attempts: {e}")
                        return None
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)

            # Create vector store with memory management
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                splits = text_splitter.split_documents(documents)
                
                # Clear memory before vector store creation
                gc.collect()
                
                # Add error handling for FAISS initialization
                try:
                    self.vector_db = FAISS.from_documents(
                        documents=splits,
                        embedding=embedding_model
                    )
                except ModuleNotFoundError as e:
                    if "faiss.swigfaiss_avx512" in str(e):
                        logger.info("Falling back to AVX2 FAISS implementation")
                        # Let it continue as FAISS will automatically fall back
                    else:
                        raise
                
                # Ensure vector store was created
                if not self.vector_db:
                    raise ValueError("Vector store creation failed")
                    
            except Exception as e:
                logger.error(f"Error creating vector store: {e}")
                return None

            # Create chain with simplified components
            try:
                template = """Question: {question}
                Context: {context}
                Please provide a comprehensive response."""
                
                prompt = ChatPromptTemplate.from_template(template)
                retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
                
                # Create context getter with error handling
                def get_context(question):
                    try:
                        docs = retriever.get_relevant_documents(question)
                        context = "\n".join(doc.page_content for doc in docs)
                        return context
                    except Exception as e:
                        logger.error(f"Error getting context: {e}")
                        return ""
                
                # Build chain with minimal configuration
                chain = (
                    {"context": get_context, "question": RunnablePassthrough()}
                    | prompt 
                    | llm 
                    | StrOutputParser()
                )
                
                # Simple chain test
                try:
                    test_response = chain.invoke("test")
                    if test_response:
                        logger.info("Chain created and tested successfully")
                        return chain
                    else:
                        raise ValueError("Chain test failed - empty response")
                except Exception as e:
                    logger.error(f"Chain test failed: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating chain components: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in create_chain: {e}")
            return None

    def _get_s3_content(self, s3_path):
        """Get content from S3 with error handling"""
        try:
            bucket = s3_path.split('/')[2]
            key = '/'.join(s3_path.split('/')[3:])
            
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Error reading from S3: {e}")
            return None

@dataclass
class PredictionContext:
    """Data class for storing prediction context"""
    prediction_data: Dict[str, Any]
    features: Dict[str, float]
    timestamp: str
    query: str
    visualization_data: Optional[Dict] = None

class EnhancedQASystem(QASystem):
    """
    Enhanced QA System with integrated prediction context and visualization capabilities.
    
    Attributes:
        prediction_store (Dict): Store for prediction contexts
        visualization_cache (Dict): Cache for visualization data
        s3_client: AWS S3 client for cloud storage
        bucket_name (str): S3 bucket name
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 25):
        super().__init__(chunk_size, chunk_overlap)
        self.prediction_store = {}
        self.visualization_cache = {}
        self.setup_cloud_storage()
        self.setup_visualization_handlers()
        self.viz_generator = VisualizationGenerator()
        

    # Add the new generate_response method here, right after ask()
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate comprehensive response with combined visualizations.
        
        Args:
            query: User's question
            
        Returns:
            Dict containing text response and relevant visualizations
        """
        try:
            # Get text response first
            text_response = self.ask(query)
            
            # Only proceed with visualizations if we have valid response and data
            if text_response and not text_response.startswith("Error") and not text_response.startswith("System not ready"):
                figures = []  # List to hold multiple visualizations
                
                if hasattr(self, 'data_df') and self.data_df is not None:
                    # Check for different visualization types needed
                    viz_types = []
                    
                    # Price trends
                    if any(word in query.lower() for word in ['trend', 'price', 'cost', 'value', 'historical']):
                        if self._get_price_trends_data():
                            viz_types.append('price_trends')
                            
                    # Feature importance 
                    if any(word in query.lower() for word in ['feature', 'factor', 'impact', 'influence']):
                        if self._get_feature_importance_data():
                            viz_types.append('feature_importance')
                            
                    # Market analysis
                    if any(word in query.lower() for word in ['market', 'segment', 'compare', 'analysis']):
                        if self._get_market_analysis_data():
                            viz_types.append('market_analysis')
                    
                    # Generate all relevant visualizations
                    for viz_type in viz_types:
                        viz_data = None
                        
                        if viz_type == 'price_trends':
                            viz_data = self._get_price_trends_data()
                        elif viz_type == 'feature_importance':
                            viz_data = self._get_feature_importance_data()
                        elif viz_type == 'market_analysis':
                            viz_data = self._get_market_analysis_data()
                            
                        if viz_data and hasattr(self, 'viz_generator'):
                            fig = self.viz_generator.create_visualization(viz_type, viz_data)
                            if fig:
                                # Update layout for consistent look
                                fig.update_layout(
                                    height=400,
                                    margin=dict(l=40, r=40, t=40, b=40),
                                    template='plotly_white',
                                    showlegend=True
                                )
                                figures.append(fig)
                    
                    # Combine figures if we have multiple
                    if len(figures) > 1:
                        combined_fig = go.Figure()
                        row_height = 400  # Height per visualization
                        
                        for i, fig in enumerate(figures):
                            for trace in fig.data:
                                # Adjust y positions for each visualization
                                if hasattr(trace, 'y'):
                                    trace.y = [val + (i * row_height) for val in trace.y]
                                combined_fig.add_trace(trace)
                        
                        # Update layout for combined figure
                        combined_fig.update_layout(
                            height=row_height * len(figures),
                            title="Combined Analysis",
                            showlegend=True,
                            template='plotly_white'
                        )
                        
                        return {
                            'text': text_response,
                            'visualization': combined_fig
                        }
                    elif len(figures) == 1:
                        return {
                            'text': text_response,
                            'visualization': figures[0]
                        }
                
                return {
                    'text': text_response,
                    'visualization': None
                }
            else:
                return {
                    'text': text_response,
                    'visualization': None
                }
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'text': "Error processing request. Please try again.",
                'visualization': None
            }
    def setup_cloud_storage(self):
        """Initialize AWS S3 connection with error handling"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            self.bucket_name = os.getenv('AWS_BUCKET_NAME')
            logger.info("Successfully initialized cloud storage")
        except Exception as e:
            logger.error(f"Failed to initialize cloud storage: {e}")
            self.s3_client = None
            
    def setup_visualization_handlers(self):
        """Set up visualization type handlers"""
        self.viz_handlers = {
            'price_trend': self._create_price_trend_viz,
            'feature_importance': self._create_feature_importance_viz,
            'market_comparison': self._create_market_comparison_viz
        }

    def store_prediction_context(self, 
                        prediction_result: Dict[str, Any], 
                        query: str,
                        features: Dict[str, float]) -> str:
        """
        Store prediction context with cloud backup.
        
        Args:
            prediction_result: Dictionary containing prediction outputs
            query: Original user query
            features: Feature importance values
            
        Returns:
            str: Context ID for reference
        """
        try:
            # Create context object
            context = PredictionContext(
                prediction_data=prediction_result,
                features=features,
                timestamp=datetime.now().isoformat(),
                query=query
            )
            
            # Generate unique ID
            context_id = hashlib.md5(
                f"{query}_{context.timestamp}".encode()
            ).hexdigest()
            
            # Store locally
            self.prediction_store[context_id] = context
            
            # Store in cloud if available
            if self.s3_client:
                self._store_in_cloud(context_id, context)
                
            return context_id
            
        except Exception as e:
            logger.error(f"Error storing prediction context: {e}")
            return None
            
    def _store_in_cloud(self, context_id: str, context: PredictionContext):
        """Store context in S3 with error handling"""
        try:
            context_json = json.dumps({
                'prediction_data': context.prediction_data,
                'features': context.features,
                'timestamp': context.timestamp,
                'query': context.query
            })
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f'prediction_contexts/{context_id}.json',
                Body=context_json
            )
            logger.info(f"Successfully stored context {context_id} in cloud")
            
        except ClientError as e:
            logger.error(f"Failed to store context in cloud: {e}")

    def get_visualization_data(self, 
                            query: str, 
                            context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get visualization data based on query and context.
        
        Args:
            query: User query
            context_id: Optional context ID for specific prediction
            
        Returns:
            Dict containing visualization data
        """
        try:
            # Determine visualization type needed
            viz_type = self._determine_visualization_type(query.lower())
            
            if not viz_type:
                return None
                
            # Get context data
            context = None
            if context_id and context_id in self.prediction_store:
                context = self.prediction_store[context_id]
            else:
                # Get most recent context
                recent_contexts = sorted(
                    self.prediction_store.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                if recent_contexts:
                    context = recent_contexts[0][1]
            
            if not context:
                return None
                
            # Create visualization data
            viz_handler = self.viz_handlers.get(viz_type)
            if viz_handler:
                return viz_handler(context)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            return None

    def _determine_visualization_type(self, query: str) -> Optional[str]:
        """Determine appropriate visualization type based on query"""
        if any(word in query for word in ['trend', 'price', 'value', 'cost']):
            return 'price_trend'
        elif any(word in query for word in ['feature', 'factor', 'important']):
            return 'feature_importance'
        elif any(word in query for word in ['compare', 'market', 'similar']):
            return 'market_comparison'
        return None

    def _create_price_trend_viz(self, context: PredictionContext) -> Dict[str, Any]:
        """Create price trend visualization data"""
        try:
            pred_data = context.prediction_data
            
            return {
                'type': 'price_trend',
                'data': {
                    'predicted_price': pred_data['predicted_price'],
                    'confidence_interval': pred_data['confidence_interval'],
                    'historical_prices': self._get_historical_prices()
                }
            }
        except Exception as e:
            logger.error(f"Error creating price trend visualization: {e}")
            return None

    def _create_feature_importance_viz(self, context: PredictionContext) -> Dict[str, Any]:
        """Create feature importance visualization data"""
        try:
            # Sort features by importance
            sorted_features = dict(sorted(
                context.features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10])
            
            return {
                'type': 'feature_importance',
                'data': {
                    'features': list(sorted_features.keys()),
                    'importance': list(sorted_features.values())
                }
            }
        except Exception as e:
            logger.error(f"Error creating feature importance visualization: {e}")
            return None

    def _create_market_comparison_viz(self, context: PredictionContext) -> Dict[str, Any]:
        """Create market comparison visualization data"""
        try:
            pred_data = context.prediction_data
            
            return {
                'type': 'market_comparison',
                'data': {
                    'predicted_price': pred_data['predicted_price'],
                    'market_average': self._get_market_average(),
                    'similar_vehicles': self._get_similar_vehicles(
                        pred_data.get('vehicle_details', {})
                    )
                }
            }
        except Exception as e:
            logger.error(f"Error creating market comparison visualization: {e}")
            return None

    def _get_historical_prices(self) -> Dict[str, float]:
        """Get historical price data from market analysis"""
        try:
            if hasattr(self, 'market_insights'):
                return self.market_insights.get('historical_prices', {})
            return {}
        except Exception:
            return {}

    def _get_market_average(self) -> float:
        """Get market average price"""
        try:
            if hasattr(self, 'market_insights'):
                return self.market_insights.get('market_summary', {}).get('median_price', 0)
            return 0
        except Exception:
            return 0

    def _get_similar_vehicles(self, vehicle_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get similar vehicle listings"""
        try:
            if hasattr(self, 'market_insights'):
                similar = self.market_insights.get('similar_vehicles', [])
                return similar[:5]  # Return top 5 similar vehicles
            return []
        except Exception:
            return []
        
    def _get_price_trends_data(self) -> Dict:
        """Get enhanced price trends data"""
        try:
            if hasattr(self, 'data_df'):
                df = self.data_df.copy()
                if 'saledate' in df.columns:
                    df['date'] = pd.to_datetime(df['saledate'])
                    # Monthly trends
                    monthly = df.groupby(df['date'].dt.strftime('%Y-%m'))[['sellingprice']].agg({
                        'sellingprice': ['mean', 'median', 'count']
                    }).reset_index()
                    
                    return {
                        'dates': monthly['date'].tolist(),
                        'mean_price': monthly[('sellingprice', 'mean')].tolist(),
                        'median_price': monthly[('sellingprice', 'median')].tolist(),
                        'volume': monthly[('sellingprice', 'count')].tolist()
                    }
            return None
        except Exception:
            return None

    def _get_feature_importance_data(self) -> Dict:
        """Get enhanced feature importance data"""
        try:
            if hasattr(self, 'predictor') and hasattr(self.predictor, 'best_models'):
                if 'rf' in self.predictor.best_models:
                    model = self.predictor.best_models['rf']
                    importance = dict(zip(
                        self.predictor.feature_columns,
                        model.feature_importances_
                    ))
                    
                    # Sort and get top features
                    sorted_features = dict(sorted(
                        importance.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:10])
                    
                    return {
                        'features': list(sorted_features.keys()),
                        'importance': list(sorted_features.values()),
                        'color_scale': [
                            'rgb(8,48,107)',
                            'rgb(8,81,156)',
                            'rgb(33,113,181)',
                            'rgb(66,146,198)',
                            'rgb(107,174,214)',
                            'rgb(158,202,225)',
                            'rgb(198,219,239)',
                            'rgb(222,235,247)',
                            'rgb(247,251,255)'
                        ][:len(sorted_features)]
                    }
            return None
        except Exception:
            return None

    def update_chain_with_prediction(self, prediction_id: str):
        """
        Update QA chain with new prediction context.
        
        Args:
            prediction_id: ID of prediction context to add
        """
        try:
            if prediction_id not in self.prediction_store:
                logger.warning(f"Prediction context {prediction_id} not found")
                return
                
            context = self.prediction_store[prediction_id]
            
            # Create document from prediction context
            doc = Document(
                page_content=json.dumps({
                    'prediction': context.prediction_data,
                    'features': context.features,
                    'query': context.query
                }, indent=2),
                metadata={
                    'source': 'prediction_context',
                    'id': prediction_id,
                    'timestamp': context.timestamp
                }
            )
            
            # Add to vector store
            if self.vector_db:
                self.vector_db.add_documents([doc])
                logger.info(f"Successfully added prediction context {prediction_id} to QA chain")
            
        except Exception as e:
            logger.error(f"Error updating chain with prediction: {e}")

        
def main():
    """Main execution with updated system"""
    try:
        sources = [
            {"path": "Sources/mmv.pdf", "type": "pdf"},
            {"path": "Sources/autoconsumer.pdf", "type": "pdf"},
            {"path": "Sources/car_prices.csv", "type": "csv"},
            {"columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                        'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                        'seller', 'mmr', 'sellingprice', 'saledate']}
        ]
        
        # Initialize enhanced QA system instead of OptimizedQASystem
        qa_system = EnhancedQASystem(chunk_size=1000, chunk_overlap=50)
        chain = qa_system.create_chain(sources)
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            try:
                # Get response and visualization if available
                response = chain.invoke(question)
                print(f"\nAnswer: {response}")
                
                # Get visualization data if applicable
                viz_data = qa_system.get_visualization_data(question)
                if viz_data:
                    print("\nVisualization data available!")
                    print(json.dumps(viz_data, indent=2))
                    
                print("-" * 50)
                qa_system.document_tracer.print_trace(question)
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()