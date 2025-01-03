"""
Car Price Prediction Module

Implements machine learning models for car price prediction with
integrated market analysis capabilities.

Environment:
    AWS EC2 Free Tier

Features:
    - Multiple model training (RF, GBM, XGBoost)
    - Feature importance analysis using SHAP
    - Market trend analysis
    - Price prediction with confidence intervals
    - Memory-optimized processing for free tier constraints
"""

# Standard Library Imports
import os
import json
import time
import hashlib
import logging
import datetime
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any

# Third-Party Libraries
import streamlit as st
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import joblib

# Machine Learning and Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap

# Application-Specific Imports
from AI_Chat_Analyst_Script import QASystem, Document

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarPriceAnalyst:
    """
    Enhanced price analyst with integrated QA capabilities.
    
    Attributes:
        qa_system (QASystem): Reference to QA system
        predictor (CarPricePredictor): Price prediction model
        context_store (Dict): Store for prediction contexts
    """
    
    def __init__(self):
        self.qa_system = QASystem(chunk_size=1000, chunk_overlap=50)
        self.predictor = None
        self.context_store = {}
        
    def analyze_and_store(self, prediction_result: Dict[str, Any], query: str):
        """
        Analyze prediction results and store context for QA system.
        
        Args:
            prediction_result: Dictionary containing prediction outputs
            query: Original user query
        """
        try:
            # Store prediction context
            self.qa_system.update_predictor_context({
                'prediction': prediction_result,
                'query': query,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update local context store
            context_id = hashlib.md5(str(prediction_result).encode()).hexdigest()
            self.context_store[context_id] = {
                'data': prediction_result,
                'query': query,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_and_store: {e}")
    
    def setup_sources(self):
        """Setup sources for the QA system with proper initialization"""
        try:
            # Initialize QA system with proper parameters
            self.qa_system = QASystem(chunk_size=1000, chunk_overlap=50)
            
            sources = [
                {"path": "pricing_data.csv", "type": "csv"},
                {"path": "pricing_manual.pdf", "type": "pdf"},
                {"path": "pricing_outputs.json", "type": "json"}
            ]
            
            # Create chain with error handling
            try:
                self.chain = self.qa_system.create_chain(sources)
                logging.info("QA System successfully initialized for Pricing Function")
            except Exception as e:
                logging.error(f"Error creating QA chain: {e}")
                self.chain = None
                
        except Exception as e:
            logging.error(f"Error setting up QA system: {e}")
            raise
        
    def process_pricing_data(self, data: pd.DataFrame):
            """Process data through the pricing pipeline"""
            try:
                # Prepare data
                processed_data = self.predictor.prepare_data(data)
                
                # Engineer features
                features = self.predictor.engineer_features(processed_data)
                
                # Split features and target
                X = features.drop(['sellingprice'], axis=1)
                y = features['sellingprice']
                
                # Train test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Fit models and scaler
                self.predictor.fit(X_train, y_train)
                
                # Evaluate models
                metrics, predictions = self.predictor.evaluate(X_test, y_test)
                
                return {
                    'metrics': metrics,
                    'processed_data': processed_data,
                    'features': features,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
            except Exception as e:
                logging.error(f"Error in process_pricing_data: {e}")
                raise

    def get_response(self, query: str, total_data: pd.DataFrame, filtered_data: pd.DataFrame, 
                    prediction_result: Dict[str, Any] = None, shap_values: Dict[str, float] = None) -> str:
        """Generate focused analysis using QA system instead of OpenAI"""
        try:
            # Prepare context from prediction results and SHAP values
            context = []
            
            if prediction_result:
                context.append(f"""
                Prediction Details:
                - Predicted Price: ${prediction_result['predicted_price']:,.2f}
                - Price Range: ${prediction_result['prediction_interval'][0]:,.2f} to ${prediction_result['prediction_interval'][1]:,.2f}
                - Confidence: {(1 - prediction_result['mape']) * 100:.1f}%
                """)

            if shap_values:
                # Add top 5 most influential features
                sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                feature_impacts = []
                for feature, impact in sorted_features:
                    direction = "increases" if impact > 0 else "decreases"
                    feature_name = feature.replace('_', ' ').title()
                    feature_impacts.append(f"- {feature_name} {direction} price by ${abs(impact):,.2f}")
                context.append("\nKey Price Factors:\n" + "\n".join(feature_impacts))

            # Combine query with context
            enhanced_query = f"""
            Context:
            {' '.join(context)}
            
            Question: {query}
            
            Please provide a clear, direct analysis focused on answering the question.
            """
            
            # Use QA system for response
            response = self.qa_system.ask(enhanced_query)
            
            # Format the response for Streamlit
            self.format_claude_response(response)
            return ""
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error analyzing data: {str(e)}"

    def format_claude_response(self, response):
        """Format the response with Streamlit markdown"""
        try:
            # Convert response to string and clean up
            text = str(response).strip()
            
            # Add styling
            st.markdown("""
                <style>
                p {
                    margin-bottom: 1em;
                    line-height: 1.6;
                }
                strong {
                    color: #1f77b4;
                    font-weight: 600;
                }
                ul {
                    margin: 1em 0;
                    padding-left: 2em;
                }
                li {
                    margin-bottom: 0.5em;
                    line-height: 1.6;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Display the formatted response
            st.markdown(text, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error formatting response: {str(e)}")
            return str(response)
        
    
class PricingFuncAIChat:
    def __init__(self):
        # Initialize QA System with Pricing Function Sources
        self.qa_system = QASystem(chunk_size=1000, chunk_overlap=50)
        self.chain = None
        self.setup_sources()

    def setup_sources(self):
        """Setup sources for the QA system"""
        sources = [
            {"path": "pricing_data.csv", "type": "csv"},
            {"path": "pricing_manual.pdf", "type": "pdf"},
            {"path": "pricing_outputs.json", "type": "json"}
        ]
        try:
            self.chain = self.qa_system.create_chain(sources)
            logging.info("QA System successfully initialized for Pricing Function")
        except Exception as e:
            logging.error(f"Error setting up sources for QA System: {e}")

    def ask(self, query: str):
        """Ask a question to the QA system"""
        try:
            return self.qa_system.ask(query)
        except Exception as e:
            logging.error(f"Error in QA system query: {e}")
            return "An error occurred while processing your query."
        
    def create_documents_from_outputs(self, outputs: dict, output_path: str = "pricing_outputs.json"):
        """
        Save outputs as documents for RAG integration.
        
        Args:
            outputs: Dictionary containing outputs to save as documents.
            output_path: Path to save the JSON file.
        """
        try:
            # Convert outputs to Document objects
            documents = [
                Document(
                    page_content=json.dumps({key: value}, indent=2),
                    metadata={"source": "pricing_func_output", "key": key}
                )
                for key, value in outputs.items()
            ]

            # Save documents to a file for later use in QA System
            with open(output_path, "w") as f:
                json.dump([doc.dict() for doc in documents], f)

            logging.info(f"Pricing outputs successfully saved as documents at {output_path}")
        except Exception as e:
            logging.error(f"Error creating documents from outputs: {e}")


class CarPricePredictor:
    def __init__(self, models=None, fast_mode=False, max_samples=None, cache_dir='model_cache'):
        # Initialize cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.best_models = {}
        self.fast_mode = fast_mode
        self.max_samples = max_samples
        self.feature_columns = None
        self.is_trained = False
        self.metrics = {}
        self.unique_values = {}
        self.required_columns = {
            'categorical': ['make', 'model', 'trim', 'body', 'transmission', 
                        'state', 'color', 'interior', 'seller'],
            'numeric': ['condition', 'odometer', 'sellingprice']
        }
        # Configure parallel processing based on system resources
        self.n_jobs = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        if self.fast_mode:
            self.n_jobs = min(2, self.n_jobs)
        
        self.available_models = {
            'ridge': {'speed': 1, 'name': 'Ridge Regression'},
            'lasso': {'speed': 2, 'name': 'Lasso Regression'},
            'gbm': {'speed': 3, 'name': 'Gradient Boosting'},
            'rf': {'speed': 4, 'name': 'Random Forest'},
            'xgb': {'speed': 5, 'name': 'XGBoost'}
        }
        
        self.selected_models = models if models else list(self.available_models.keys())
        
        self.param_grids = {
            'regular': {
                'rf': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gbm': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'xgb': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
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
        
    def get_prediction_context(self, prediction_result, input_data):
        """
        Format prediction results for QA system context
        
        Args:
            prediction_result: Dictionary with prediction outputs
            input_data: Original input features
            
        Returns:
            Dict with formatted context
        """
        context = {
            'prediction': {
                'price': prediction_result['predicted_price'],
                'confidence_interval': prediction_result['confidence_interval'],
                'error_margin': prediction_result['mape']
            },
            'input_features': input_data,
            'model_performance': {
                name: {
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2']
                }
                for name, metrics in self.metrics.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add SHAP values if available
        if hasattr(self, 'feature_importance'):
            context['feature_importance'] = self.feature_importance
            
        return context

        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the input DataFrame structure and required columns."""
        try:
            all_required = (self.required_columns['categorical'] + 
                    self.required_columns['numeric'])
            missing_columns = [col for col in all_required if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False

    def update_unique_values(self, df: pd.DataFrame):
        """Update unique values for categorical columns."""
        try:
            def safe_sort(values):
                cleaned_values = [str(x) for x in values if pd.notna(x)]
                return sorted(cleaned_values)
            
            for col in self.required_columns['categorical']:
                if col in df.columns:
                    self.unique_values[col] = safe_sort(df[col].unique())
                    
        except Exception as e:
            logger.error(f"Error updating unique values: {str(e)}")
            raise
        
    def _clean_numeric_columns(self, df: pd.DataFrame):
        """Clean numeric columns with robust error handling."""
        numeric_cols = {
            'year': (int, 1900, 2024),
            'odometer': (float, 0, 500000),
            'sellingprice': (float, 100, 1000000),
            'condition': (float, 0, 50)
        }

        for col, (dtype, min_val, max_val) in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).replace(r'[^\d.-]', '', regex=True), 
                                errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                df.loc[df[col] < min_val, col] = min_val
                df.loc[df[col] > max_val, col] = max_val
                df[col] = df[col].astype(dtype)

    def _clean_categorical_columns(self, df: pd.DataFrame):
        """Clean categorical columns with robust error handling."""
        categorical_cols = {
            'make': str, 'model': str, 'trim': str,
            'state': str, 'body': str, 'transmission': str,
            'color': str, 'interior': str, 'seller': str
        }

        for col, dtype in categorical_cols.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(r'[^\w\s-]', '', regex=True)
                df[col] = df[col].replace('NAN', 'UNKNOWN')
                df[col] = df[col].fillna('UNKNOWN')


    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with enhanced error handling."""
        try:
            df = df.copy()
            df.drop('vin', axis=1, errors='ignore', inplace=True)
            df.drop('saledate', axis=1, errors='ignore', inplace=True)
            
            # Initial validation
            if not self.validate_data(df):
                raise ValueError("Data validation failed")

            # Clean and process columns
            self._clean_numeric_columns(df)
            self._clean_categorical_columns(df)
            
            # Remove outliers
            df = self.remove_outliers(df)
            
            # Update unique values for UI
            self.update_unique_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def _prepare_data_cached(self, data_hash):
        """Cached version of data preparation"""
        # Convert hash back to dataframe (implementation needed)
        df = self._hash_to_df(data_hash)
        return self.prepare_data(df)
    
    def _generate_cache_key(self, data_df, model_type):
        """Generate a unique cache key based on data and model parameters"""
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data_df).values).hexdigest()
        params = {
            'model_type': model_type,
            'fast_mode': self.fast_mode,
            'max_samples': self.max_samples
        }
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return f"{data_hash}_{param_hash}"

    def _get_cached_model(self, cache_key, model_type):
        """Retrieve model from cache if available"""
        cache_path = self.cache_dir / f"{model_type}_{cache_key}.joblib"
        if cache_path.exists():
            try:
                return joblib.load(cache_path)
            except:
                return None
        return None

    def _save_model_to_cache(self, model, cache_key, model_type):
        """Save trained model to cache"""
        cache_path = self.cache_dir / f"{model_type}_{cache_key}.joblib"
        joblib.dump(model, cache_path)

    def _get_base_model(self, model_type):
        """Get base model with configured parameters"""
        if model_type == 'rf':
            return RandomForestRegressor(
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=0
            )
        elif model_type == 'gbm':
            return GradientBoostingRegressor(
                random_state=42,
                verbose=0
            )
        elif model_type == 'xgb':
            return xgb.XGBRegressor(
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=0,
                tree_method='hist'
            )
        elif model_type == 'lasso':
            return LassoCV(
                cv=3 if self.fast_mode else 5,
                random_state=42,
                max_iter=2000,
                n_jobs=self.n_jobs
            )
        elif model_type == 'ridge':
            return RidgeCV(
                cv=3 if self.fast_mode else 5,
                n_jobs=self.n_jobs
            )

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        """Remove price outliers using IQR method."""
        try:
            Q1 = df['sellingprice'].quantile(0.25)
            Q3 = df['sellingprice'].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_condition = (
                (df['sellingprice'] >= (Q1 - threshold * IQR)) & 
                (df['sellingprice'] <= (Q3 + threshold * IQR))
            )
            
            return df[outlier_condition]
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with proper type handling."""
        try:
            df = df.copy()
            
            # Store original features
            self.original_features = df.copy()
            
            # Create numeric feature transformations
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_cols = [col for col in numeric_cols if col not in ['sellingprice', 'mmr']]
            
            # Log transform for positive numeric columns
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[f'{col}_log'] = np.log(df[col])
            
            # Square transform for key numeric features
            key_numeric = ['odometer', 'year', 'condition']
            for col in key_numeric:
                if col in df.columns:
                    df[f'{col}_squared'] = df[col] ** 2
            
            # One-hot encode categorical columns
            categorical_cols = [col for col in self.required_columns['categorical'] 
                            if col in df.columns]
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            
            # Create interaction features
            df['vehicle_age'] = 2024 - df['year']
            df['age_miles'] = df['vehicle_age'] * df['odometer']
            df['age_squared'] = df['vehicle_age'] ** 2
            
            return df
            
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            raise

    def remove_multicollinearity(self, X, threshold=0.95):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            return X.drop(columns=to_drop)
        return X

    def fit(self, X, y):
        """Modified fit method to avoid pickle errors"""
        self.feature_columns = X.columns.tolist()
        
        # Fit the scaler and transform the data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Train models sequentially to avoid pickle errors
        self.best_models = {}
        
        for model_type in self.selected_models:
            try:
                if model_type in ['rf', 'gbm', 'xgb']:
                    model = self.tune_model(model_type, X_scaled, y)
                elif model_type == 'lasso':
                    model = LassoCV(cv=3 if self.fast_mode else 5, random_state=42).fit(X_scaled, y)
                elif model_type == 'ridge':
                    model = RidgeCV(cv=3 if self.fast_mode else 5).fit(X_scaled, y)
                    
                if model is not None:
                    self.best_models[model_type] = model
                    
            except Exception as e:
                logging.error(f"Error training {model_type}: {e}")
                continue
        
        # Train ensemble if multiple models are available
        if len(self.best_models) > 1:
            self.ensemble = VotingRegressor([
                (name, model) for name, model in self.best_models.items()
            ])
            self.ensemble.fit(X_scaled, y)
        
        self.is_trained = True

    def tune_model(self, model_type, X, y):
        """Modified tune_model method with simplified parallel processing"""
        try:
            param_grid = self.param_grids['fast' if self.fast_mode else 'regular'].get(model_type)
            if not param_grid:
                return None

            # Initialize base model
            if model_type == 'rf':
                base_model = RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1 if not self.fast_mode else 1
                )
            elif model_type == 'gbm':
                base_model = GradientBoostingRegressor(
                    random_state=42
                )
            else:
                return None

            # Use GridSearchCV with built-in parallelization
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3 if self.fast_mode else 5,
                scoring='neg_mean_squared_error',
                n_jobs=-1 if not self.fast_mode else 1,
                verbose=0
            )

            grid_search.fit(X, y)
            return grid_search.best_estimator_

        except Exception as e:
            logging.error(f"Error in tune_model for {model_type}: {e}")
            return None

    def cleanup_cache(self, max_age_days=30):
        """Clean up old cache files"""
        current_time = time.time()
        for cache_file in self.cache_dir.glob("*.joblib"):
            file_age_days = (current_time - cache_file.stat().st_mtime) / (24 * 3600)
            if file_age_days > max_age_days:
                cache_file.unlink()

    def __del__(self):
        """Cleanup when instance is deleted"""
        try:
            self.cleanup_cache()
        except:
            pass

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
        
        # Error occurs here - scaler not fitted
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

def analyze_shap_values(predictor, input_data):
    """Generate SHAP analysis for the prediction"""
    try:
        if 'rf' not in predictor.best_models:
            return None
            
        model = predictor.best_models['rf']
        explainer = shap.TreeExplainer(model)
        
        # Prepare the input data
        df_encoded = predictor.prepare_prediction_data(input_data)
        X_scaled = predictor.scaler.transform(df_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=predictor.feature_columns)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_scaled)
        
        # Create dictionary of feature importance
        feature_importance = dict(zip(predictor.feature_columns, shap_values[0]))
        return feature_importance
        
    except Exception as e:
        logging.error(f"Error in SHAP analysis: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Car Price Predictor",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Car Price Predictor")
    
    # File upload in sidebar
    st.sidebar.header("Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Car Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            total_data = pd.read_csv(uploaded_file)
            
            string_cols = ['make', 'model', 'trim', 'state', 'body', 'transmission', 'color', 'interior']
            for col in string_cols:
                if col in total_data.columns:
                    total_data[col] = total_data[col].astype(str)
            
            # Initialize predictor
            predictor = CarPricePredictor(
                models=['gbm', 'rf', 'xgb'],
                fast_mode=True
            )
            
            predictor.update_unique_values(total_data)
            
            # Model settings
            st.sidebar.subheader("Model Settings")
            fast_mode = st.sidebar.checkbox("Fast Mode (Quick Training)", value=True)
            predictor.fast_mode = fast_mode
            
            # SECTION 1: PRICE PREDICTOR
            st.header("Select Vehicle")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                make = st.selectbox("Make", options=sorted(total_data['make'].unique()))
            
            filtered_models = total_data[total_data['make'] == make]['model'].unique()
            with col2:
                model = st.selectbox("Model", options=sorted(filtered_models))
            
            filtered_trims = total_data[
                (total_data['make'] == make) & 
                (total_data['model'] == model)
            ]['trim'].unique()
            with col3:
                trim = st.selectbox("Trim", options=sorted(filtered_trims))
            
            # Filter data for selected vehicle
            filtered_data = total_data[
                (total_data['make'] == make) &
                (total_data['model'] == model) &
                (total_data['trim'] == trim)
            ]
            
            st.info(f"Number of samples for this vehicle: {len(filtered_data)}")
            
            # Model training section
            if len(filtered_data) > 5:  # Minimum samples needed for training
                if st.button("Train Models", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            # Prepare and engineer features
                            df = predictor.prepare_data(filtered_data)
                            df_engineered = predictor.engineer_features(df)
                            
                            # Split features and target
                            X = df_engineered.drop(['sellingprice', 'mmr'] if 'mmr' in df_engineered.columns else ['sellingprice'], axis=1)
                            y = df_engineered['sellingprice']
                            
                            X = predictor.remove_multicollinearity(X)
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            predictor.fit(X_train, y_train)
                            metrics, predictions = predictor.evaluate(X_test, y_test)
                            
                            st.session_state['predictor'] = predictor
                            st.session_state['metrics'] = metrics
                            st.session_state['filtered_data'] = filtered_data
                            st.session_state['total_data'] = total_data
                            
                            st.success("Models trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
            else:
                st.warning("Not enough samples to train models. Please select a different vehicle with more data.")
            
            # Display model performance metrics
            if 'metrics' in st.session_state:
                st.header("Model Performance")
                
                avg_metrics = {
                    'RMSE': np.mean([m['rmse'] for m in st.session_state['metrics'].values()]),
                    'R²': np.mean([m['r2'] for m in st.session_state['metrics'].values()]),
                    'Error %': np.mean([m['mape'] for m in st.session_state['metrics'].values()]) * 100
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Error", f"{avg_metrics['Error %']:.1f}%")
                with col2:
                    st.metric("RMSE", f"${avg_metrics['RMSE']:,.0f}")
                with col3:
                    st.metric("R² Score", f"{avg_metrics['R²']:.3f}")
            
            # SECTION 2: PRICE ESTIMATOR
            st.header("Price Estimator")
            
            if 'predictor' in st.session_state:
                predictor = st.session_state['predictor']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
                    condition = st.number_input("Condition (1-50)", min_value=1.0, max_value=50.0, value=25.0, step=1.0)
                    odometer = st.number_input("Mileage", min_value=0, value=50000, step=1000)
                    state = st.selectbox("State", options=predictor.unique_values['state'])
                
                with col2:
                    body = st.selectbox("Body Style", options=predictor.unique_values['body'])
                    transmission = st.selectbox("Transmission", options=predictor.unique_values['transmission'])
                    color = st.selectbox("Color", options=predictor.unique_values['color'])
                    interior = st.selectbox("Interior", options=predictor.unique_values['interior'])
                
                if st.button("Get Price Estimate", type="primary"):
                    try:
                        input_data = {
                            'state': state,
                            'body': body,
                            'transmission': transmission,
                            'color': color,
                            'interior': interior,
                            'year': year,
                            'condition': condition,
                            'odometer': odometer
                        }
                        
                        prediction_result = predictor.create_what_if_prediction(input_data)
                        st.session_state['last_prediction'] = input_data
                        st.session_state['last_prediction_result'] = prediction_result
                        
                        mean_price = prediction_result['predicted_price']
                        mape = prediction_result['mape']
                        
                        low_estimate = mean_price * (1 - mape)
                        high_estimate = mean_price * (1 + mape)
                        
                        st.subheader("Price Estimates")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Low Estimate", f"${low_estimate:,.0f}")
                        with col2:
                            st.metric("Best Estimate", f"${mean_price:,.0f}")
                        with col3:
                            st.metric("High Estimate", f"${high_estimate:,.0f}")
                        
                        st.info(f"Estimated error margin: ±{mape*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        
            # AI Chat Assistant Section
            if 'predictor' in st.session_state:
                st.header("💡 AI Insights")

                # Initialize analyst with QASystem if not already done
                if 'analyst' not in st.session_state:
                    try:
                        st.session_state.analyst = CarPriceAnalyst()  # Uses QASystem internally
                    except Exception as e:
                        st.error(f"Error initializing AI Chat Assistant: {str(e)}")
                
                # Create the chat interface
                query = st.text_input(
                    "Ask me anything about the market, predictions, or pricing factors:",
                    placeholder="E.g., What factors most affect this car's value? How does it compare to similar models?"
                )
                
                if query:
                    with st.spinner("Analyzing data and generating insights..."):
                        try:
                            # Get SHAP values if there's a recent prediction
                            shap_values = None
                            if 'last_prediction' in st.session_state:
                                shap_values = analyze_shap_values(
                                    predictor,
                                    st.session_state['last_prediction']
                                )
                            
                            # Get response using CarPriceAnalyst with QASystem
                            response = st.session_state.analyst.get_response(
                                query,
                                total_data,
                                filtered_data,
                                st.session_state.get('last_prediction_result'),
                                shap_values
                            )
                            
                            if response is not None:
                                st.markdown("### Analysis")
                                st.session_state.analyst.format_claude_response(response)
                            
                            # Show feature impact visualization if available
                            if shap_values:
                                st.subheader("Feature Impact Visualization")
                                
                                feature_importance = pd.DataFrame({
                                    'Feature': list(shap_values.keys()),
                                    'Impact': list(shap_values.values())
                                })
                                feature_importance = feature_importance.sort_values('Impact', key=abs, ascending=True)
                                
                                fig = go.Figure(go.Bar(
                                    x=feature_importance['Impact'],
                                    y=feature_importance['Feature'],
                                    orientation='h'
                                ))
                                
                                fig.update_layout(
                                    title="Feature Impact on Price Prediction",
                                    xaxis_title="Impact on Price ($)",
                                    yaxis_title="Feature",
                                    height=max(400, len(feature_importance) * 20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                            logging.error(f"Chat error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file containing car sales data to begin.")

if __name__ == "_main_":
    main()