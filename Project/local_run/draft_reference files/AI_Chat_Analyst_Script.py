from PIL import Image
import pytesseract
import io
import fitz
import numpy as np
import pandas as pd
from langchain.docstore.document import Document
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from typing import List, Union, Dict
import os
from concurrent.futures import ThreadPoolExecutor
import gc

from multiprocessing import Pool, cpu_count
from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import shap
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict, List, Any
from joblib import Parallel, delayed

import hashlib
import pickle
import os
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json
import shap
import warnings
warnings.filterwarnings('ignore')


# Configure logging with a more efficient format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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



#############################################################################################################
class DocumentLoader:
    def __init__(self, batch_size: int = 1000):
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
        """
        Load PDF with enhanced logging for debugging.
        """
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
            
            # Process pages in smaller batches
            batch_size = 10
            for i in range(0, total_pages, batch_size):
                batch_pages = list(range(i, min(i + batch_size, total_pages)))
                logger.info(f"Processing batch of pages {i+1} to {min(i + batch_size, total_pages)}")
                
                # Process each page in the batch
                for page_num in batch_pages:
                    try:
                        page = doc[page_num]
                        logger.info(f"Processing page {page_num + 1}/{total_pages}")
                        
                        # Text extraction
                        text = page.get_text()
                        
                        if not text.strip():
                            logger.info(f"No text found on page {page_num + 1}, attempting OCR")
                            try:
                                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                                img_data = pix.tobytes()
                                pix = None
                                
                                if img_data:
                                    text = self.process_image(img_data)
                                    if text:
                                        logger.info(f"Successfully extracted text via OCR for page {page_num + 1}")
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
                            logger.info(f"Successfully extracted content from page {page_num + 1}")
                        else:
                            logger.warning(f"No content extracted from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                    
                    finally:
                        gc.collect()
                
                logger.info(f"Completed processing batch. Documents extracted so far: {len(documents)}")
                gc.collect()
        
        except fitz.FileDataError as e:
            logger.error(f"Invalid or corrupted PDF file {file_path}: {str(e)}")
            return []
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
        
        finally:
            if doc:
                try:
                    doc.close()
                    logger.info(f"Successfully closed PDF file: {file_path}")
                except Exception:
                    logger.warning(f"Error closing PDF file: {file_path}")
                doc = None
            gc.collect()
        
        logger.info(f"Completed processing {file_path}. Total documents extracted: {len(documents)}")
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
                n_jobs=-1 if not self.fast_mode else 1,
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
            n_jobs=-1,
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
            n_jobs=-1,
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
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1 if not self.fast_mode else 1)
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
            n_jobs=-1,
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

        results = Parallel(n_jobs=-1)(delayed(train_model)(model) for model in self.selected_models)
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

    
import pandas as pd
import numpy as np
from typing import Dict, List

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
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
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
        self.data_df = None  # Store the DataFrame for market analysis
        self.document_tracer = DocumentTracer()  # Initialize document tracer
    def process_sources(self, sources: List[Dict[str, Union[str, List[str]]]]) -> List[Document]:
        all_documents = []
        self.csv_file_path = None
        
        for source in sources:
            file_path = source["path"]
            file_type = source["type"].lower()
            
            try:
                if file_type == "pdf":
                    documents = self.loader.load_pdf(file_path)
                    all_documents.extend(documents)
                elif file_type == "csv":
                    self.csv_file_path = file_path
                    text_columns = source.get("columns", None)
                    csv_documents = self.loader.load_csv(file_path, text_columns)
                    all_documents.extend(csv_documents)
                    
                    # Load and process data for analysis
                    self.data_df = pd.read_csv(file_path)
                    
                    self.market_insights = self.generate_market_insights(self.data_df)
                    # Create market analysis documents with error handling
                    try:
                        market_docs = self._create_market_analysis_documents()
                        if market_docs:  # Only extend if documents were created
                            logger.info(f"Created {len(market_docs)} market analysis documents")
                            all_documents.extend(market_docs)
                        else:
                            logger.warning("No market analysis documents were created")
                    except Exception as e:
                        logger.error(f"Error creating market analysis documents: {str(e)}")
                    
                    # Process data for price prediction
                    processed_data = self.predictor.prepare_data(self.data_df)
                    processed_data = processed_data.sample(frac=0.01,random_state=42)
                    features = self.predictor.engineer_features(processed_data)
                    X = features.drop('sellingprice', axis=1)
                    y = features['sellingprice']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    self.predictor.fit(X_train, y_train)
                    metrics, _ = self.predictor.evaluate(X_test, y_test)
                    
                    # Generate and add predictor context
                    performance_summary = self._generate_performance_summary(metrics)
                    shap_summary = self.analyze_shap_values(X_test)
                    self.predictor_context = f"{performance_summary}\n\nFeature Importance Analysis:\n{shap_summary}"
                    
                    predictor_doc = Document(
                        page_content=self.predictor_context,
                        metadata={"source": "model_analysis", "type": "predictor_context"}
                    )
                    all_documents.append(predictor_doc)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                
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
    
    def create_chain(self, sources: List[Dict[str, Union[str, List[str]]]]):
        """Create enhanced QA chain with integrated market analysis context"""
        try:
            # Process documents and generate market analysis
            documents = self.process_sources(sources)
            
            if not documents:
                raise ValueError("No documents were successfully loaded")

            # Create market analysis documents
            market_docs = []
            model_docs = []
            
            if self.market_insights:
                # Feature Impact Analysis
                feature_analysis_doc = Document(
                    page_content="Feature Analysis Overview:\n" + str(self.market_insights.get('feature_impact', {})),
                    metadata={"source": "market_analysis", "type": "feature_analysis"}
                )
                market_docs.append(feature_analysis_doc)
                
                # Market Summary
                if 'market_summary' in self.market_insights:
                    summary = self.market_insights['market_summary']
                    summary_context = (
                        "Overall Market Summary:\n"
                        f"- Median Market Price: ${summary.get('median_price', 0):,.2f}\n"
                        "Market Trends Analysis:\n"
                        "- Price distribution and popular configurations\n"
                        "- Regional market variations\n"
                        "- Seasonal trends and patterns"
                    )
                    
                    market_docs.append(Document(
                        page_content=summary_context,
                        metadata={"source": "market_analysis", "type": "market_summary"}
                    ))

                # Additional market analysis document
                market_trend_doc = Document(
                    page_content="Market Trend Analysis:\n" + 
                                "- Price trends across segments\n" +
                                "- Popular feature combinations\n" +
                                "- Regional demand patterns",
                    metadata={"source": "market_analysis", "type": "market_trends"}
                )
                market_docs.append(market_trend_doc)

            # Create model analysis documents
            if self.predictor_context:
                predictor_doc = Document(
                    page_content=self.predictor_context,
                    metadata={"source": "model_analysis", "type": "predictor_context"}
                )
                model_docs.append(predictor_doc)

                # Additional model analysis documents
                model_docs.extend([
                    Document(
                        page_content="Model Performance Metrics:\n" +
                                    "- Accuracy across different price ranges\n" +
                                    "- Feature importance rankings\n" +
                                    "- Prediction confidence levels",
                        metadata={"source": "model_analysis", "type": "performance_metrics"}
                    ),
                    Document(
                        page_content="Price Prediction Factors:\n" +
                                    "- Key value drivers\n" +
                                    "- Impact of vehicle characteristics\n" +
                                    "- Market condition effects",
                        metadata={"source": "model_analysis", "type": "prediction_factors"}
                    )
                ])

            # Verify minimum document requirements
            if len(market_docs) < 2 or len(model_docs) < 2:
                raise ValueError(f"Insufficient documents: Market docs: {len(market_docs)}, Model docs: {len(model_docs)}. Minimum 2 each required.")

            # Combine all documents
            all_documents = documents + market_docs + model_docs
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(all_documents)
            
            # Create vector store
            embedding_model = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            
            self.vector_db = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model
            )
            
            # Initialize LLM
            llm = ChatOllama(model="mistral")
            
            # Create balanced retriever
            base_retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
            balanced_retriever = BalancedRetriever(base_retriever, min_docs_per_type=2)
            
            # Enhanced template
            template = """Analyze the following question using both document context and market analysis:

    Question: {question}

    Context from documents and market analysis:
    {context}

    Please provide a comprehensive response that:
    1. Addresses the specific question
    2. Incorporates relevant market trends and patterns
    3. Provides specific data points and statistics when available
    4. Offers actionable insights based on market analysis
    5. Considers both general trends and segment-specific patterns

    Develop the response with a focus on clarity, depth, and relevance to the question and can consider similar queries too it.
    Use specific numbers, percentages, and trends from the analysis when relevant to support your response."""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            def trace_and_retrieve(question: str):
                # Use balanced retriever to ensure minimum documents of each type
                retrieved_docs = balanced_retriever.get_relevant_documents(question)
                
                # Trace the documents for debugging
                self.document_tracer.trace_documents(question, retrieved_docs)
                
                # Verify retrieved documents meet minimum requirements
                market_count = len([d for d in retrieved_docs if d.metadata.get('source') == 'market_analysis'])
                model_count = len([d for d in retrieved_docs if d.metadata.get('source') == 'model_analysis'])
                
                if market_count < 2 or model_count < 2:
                    logger.warning(f"Retrieved documents below minimum requirement: Market docs: {market_count}, Model docs: {model_count}")
                    
                    # Add additional documents if needed
                    if market_count < 2:
                        retrieved_docs.extend(market_docs[:2-market_count])
                    if model_count < 2:
                        retrieved_docs.extend(model_docs[:2-model_count])
                
                return retrieved_docs
            
            chain = (
                {"context": trace_and_retrieve, "question": RunnablePassthrough()} 
                | prompt 
                | llm 
                | StrOutputParser()
            )
            
            return chain

        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
        
        

class OptimizedQASystem(QASystem):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.pre_calc_pipeline = PreCalculationPipeline()
        self.processed_data = None
        self.feature_cache = {}
        
    def _initialize_ml_components(self, df):
        """Initialize ML components with pre-calculation"""
        try:
            # Pre-process data using pipeline
            features_data = self.pre_calc_pipeline.preprocess_data(df, self.predictor)
            self.processed_data = features_data['processed_data']
            features = features_data['features']
            
            # Prepare train-test split
            X = features.drop('sellingprice', axis=1)
            y = features['sellingprice']
            
            # Use stratified sampling for better representation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=pd.qcut(y, q=5, labels=False, duplicates='drop')
            )
            
            # Train models in parallel
            def train_model_parallel(model_name):
                try:
                    if model_name in ['rf', 'gbm']:
                        model = self.predictor.tune_model(model_name, X_train, y_train)
                    elif model_name == 'lasso':
                        model = LassoCV(cv=3, random_state=42).fit(X_train, y_train)
                    elif model_name == 'ridge':
                        model = RidgeCV(cv=3).fit(X_train, y_train)
                    return model_name, model
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    return model_name, None
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                model_results = list(executor.map(
                    lambda m: train_model_parallel(m),
                    self.predictor.selected_models
                ))
            
            self.predictor.best_models = {
                name: model for name, model in model_results if model is not None
            }
            
            # Pre-calculate SHAP values for feature importance
            if 'rf' in self.predictor.best_models:
                self._precalculate_shap_values(X_test)
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error in ML initialization: {str(e)}")
            raise
            
    def _precalculate_shap_values(self, X_test):
        """Pre-calculate and cache SHAP values"""
        try:
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            shap_values = compute_shap_values(
                self.predictor.best_models['rf'],
                X_sample,
                self.pre_calc_pipeline.shap_cache
            )
            
            # Cache feature importance
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            self.feature_cache['importance'] = feature_importance
            
        except Exception as e:
            logger.error(f"Error pre-calculating SHAP values: {str(e)}")
            
    def process_sources(self, sources: List[Dict[str, Union[str, List[str]]]]) -> List[Document]:
        """Enhanced source processing with pre-calculation pipeline"""
        all_documents = []
        
        for source in sources:
            if source["type"].lower() == "csv":
                # Load and pre-process CSV data
                self.data_df = pd.read_csv(source["path"])
                X_test, y_test = self._initialize_ml_components(self.data_df)
                
                # Generate and evaluate predictions
                metrics, _ = self.predictor.evaluate(X_test, y_test)
                
                # Create enhanced context documents
                self._create_enhanced_context_documents(metrics, all_documents)
            else:
                # Process other document types normally
                documents = super().process_sources([source])
                all_documents.extend(documents)
        
        return all_documents
        
    def _create_enhanced_context_documents(self, metrics, all_documents):
        """Create enhanced context documents with pre-calculated insights"""
        # Add model performance document
        performance_summary = self._generate_performance_summary(metrics)
        all_documents.append(Document(
            page_content=performance_summary,
            metadata={"source": "model_analysis", "type": "model_performance"}
        ))
        
        # Add feature importance document if available
        if 'importance' in self.feature_cache:
            feature_importance = self.feature_cache['importance']
            importance_summary = "Feature Importance Analysis:\n"
            for _, row in feature_importance.head(10).iterrows():
                importance_summary += f"- {row['feature']}: {row['importance']:.4f}\n"
                
            all_documents.append(Document(
                page_content=importance_summary,
                metadata={"source": "model_analysis", "type": "feature_importance"}
            ))
            
        # Add market insights document
        if hasattr(self, 'market_insights'):
            market_summary = self._generate_market_summary()
            all_documents.append(Document(
                page_content=market_summary,
                metadata={"source": "market_analysis", "type": "market_insights"}
            ))


        
def main():
    """Optimized main execution"""
    try:
        sources = [
            {
                "path": "CARPRICEPREDICTIONUSINGMACHINELEARNINGTECHNIQUES.pdf",
                "type": "pdf"
            },
                        {
                "path": "autoconsumer.pdf",
                "type": "pdf"
            },
            {
                "path": "car_prices.csv",
                "type": "csv",
                "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 'vin', 'state','condition', 'odometer', 'color', 'interior', 'seller', 'mmr','sellingprice', 'saledate']
            }
        ]
        
        # Initialize QA system
        qa_system = OptimizedQASystem(chunk_size=1000, chunk_overlap=50)
        chain = qa_system.create_chain(sources)
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            try:
                response = chain.invoke(question)
                print(f"\nAnswer: {response}")
                print("-" * 50)
                response = chain.invoke(question)
                qa_system.document_tracer.print_trace(question)
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()