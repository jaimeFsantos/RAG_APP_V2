import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from joblib import Parallel, delayed
import warnings
from functools import lru_cache
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    fast_mode: bool
    max_samples: int = None
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    n_jobs: int = -1  # Added for parallel processing control

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.unique_values = {}
        self._feature_cache = {}
        
    @lru_cache(maxsize=32)
    def get_unique_values(self, df_hash: str) -> Dict[str, List[str]]:
        """Cache unique values for categorical columns"""
        df = pd.read_json(df_hash)  # Convert hash back to dataframe
        return {
            'state': sorted(df['state'].unique()),
            'body': sorted(df['body'].unique()),
            'transmission': sorted(df['transmission'].unique()),
            'color': sorted(df['color'].unique()),
            'interior': sorted(df['interior'].unique()),
            'make': sorted(df['make'].unique()) if 'make' in df.columns else [],
            'model': sorted(df['model'].unique()) if 'model' in df.columns else [],
            'trim': sorted(df['trim'].unique()) if 'trim' in df.columns else []
        }

    def remove_outliers(self, df: pd.DataFrame, threshold: float = 1.5) -> tuple[pd.DataFrame, dict]:
        """Remove price outliers using IQR method"""
        initial_rows = len(df)
        Q1 = df['sellingprice'].quantile(0.25)
        Q3 = df['sellingprice'].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = (
            (df['sellingprice'] < (Q1 - threshold * IQR)) | 
            (df['sellingprice'] > (Q3 + threshold * IQR))
        )
        
        df_cleaned = df[~outlier_condition].copy()
        
        return df_cleaned, {
            'initial_rows': initial_rows,
            'final_rows': len(df_cleaned),
            'rows_removed': initial_rows - len(df_cleaned)
        }

    def prepare_data(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """Prepare data for modeling with optional sampling"""
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Handle categorical columns
        categorical_cols = ['state', 'body', 'transmission', 'color', 'interior']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')  # Use category dtype for memory efficiency
        
        # Drop unnecessary columns
        drop_cols = ['datetime', 'Day of Sale', 'Weekend', 'vin', 'seller', 'saledate']
        df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        
        # Handle missing values efficiently
        fill_values = {
            'transmission': 'unknown',
            'interior': 'unknown',
            'condition': df['condition'].median(),
            'odometer': df['odometer'].median()
        }
        df = df.fillna(fill_values)
        
        return df[df['sellingprice'] > 0]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features with caching"""
        cache_key = hash(str(df.head()))
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        df = df.copy()
        
        # Create age-related features
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        df['age_miles'] = df['vehicle_age'] * df['odometer']
        df['age_squared'] = df['vehicle_age'] ** 2
        
        # Log transform numeric features efficiently
        numeric_cols = ['odometer', 'condition']
        df_pos = df[numeric_cols] > 0
        for col in numeric_cols:
            if df_pos[col].all():
                df[f'{col}_log'] = np.log(df[col])
        
        # Create polynomial features
        df[['odometer_squared', 'year_squared', 'condition_squared']] = \
            df[['odometer', 'year', 'condition']].pow(2)
        
        # One-hot encode categorical features efficiently
        categorical_cols = ['body', 'transmission', 'state', 'color', 'interior']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=True)
        
        self._feature_cache[cache_key] = df
        return df

class CarPriceModel:
    """Main model class for car price prediction"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.models = self._initialize_models()
        self.feature_columns = None
        self.metrics = {}
        
    def _initialize_models(self) -> dict:
        """Initialize model instances with optimized parameters"""
        n_jobs = self.config.n_jobs if not self.config.fast_mode else 1
        return {
            'rf': RandomForestRegressor(
                n_estimators=50 if self.config.fast_mode else 100,
                max_depth=10 if self.config.fast_mode else None,
                random_state=self.config.random_state,
                n_jobs=n_jobs,
                verbose=0
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=50 if self.config.fast_mode else 100,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=0
            )
        }

    def filter_data(self, df: pd.DataFrame, make: str = None, model: str = None, trim: str = None) -> pd.DataFrame:
        """Filter data based on make, model, and trim selections"""
        if make and make != "All":
            df = df[df['make'] == make]
        if model and model != "All":
            df = df[df['model'] == model]
        if trim and trim != "All":
            df = df[df['trim'] == trim]
        return df

    def train(self, df: pd.DataFrame, make: str = None, model: str = None, trim: str = None) -> Dict[str, Any]:
        """Train all models with filtered data"""
        try:
            # Filter data based on selections
            df_filtered = self.filter_data(df, make, model, trim)
            
            # Preprocess data with sampling
            df_processed = self.preprocessor.prepare_data(
                df_filtered, 
                sample_size=self.config.max_samples
            )
            df_processed, outlier_stats = self.preprocessor.remove_outliers(df_processed)
            df_engineered = self.preprocessor.engineer_features(df_processed)
            
            # Prepare features
            X = df_engineered.drop(['sellingprice', 'mmr'] if 'mmr' in df_engineered.columns else ['sellingprice'], axis=1)
            y = df_engineered['sellingprice']
            self.feature_columns = X.columns.tolist()
            
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            X_train_scaled = self.preprocessor.scaler.fit_transform(X_train)
            X_test_scaled = self.preprocessor.scaler.transform(X_test)
            
            # Train models in parallel
            def train_model(name, model):
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                return name, {
                    'model': model,
                    'metrics': {
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'mape': mean_absolute_percentage_error(y_test, y_pred)
                    }
                }
            
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(train_model)(name, model) 
                for name, model in self.models.items()
            )
            
            # Process results
            metrics = {}
            for name, result in results:
                self.models[name] = result['model']
                metrics[name] = result['metrics']
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, input_data: dict) -> Dict[str, Any]:
        """Make predictions with all models"""
        try:
            # Prepare input data
            df = pd.DataFrame([input_data])
            df_processed = self.preprocessor.engineer_features(df)
            
            # Ensure all required columns are present
            missing_cols = set(self.feature_columns) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            
            df_processed = df_processed.reindex(columns=self.feature_columns, fill_value=0)
            X_scaled = self.preprocessor.scaler.transform(df_processed)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_scaled)[0]
            
            # Calculate aggregate statistics
            mean_pred = np.mean(list(predictions.values()))
            std_pred = np.std(list(predictions.values()))
            mape = np.mean([m['mape'] for m in self.metrics.values()])
            
            return {
                'predicted_price': mean_pred,
                'confidence_interval': (mean_pred - (1.96 * std_pred), mean_pred + (1.96 * std_pred)),
                'prediction_interval': (mean_pred * (1 - mape), mean_pred * (1 + mape)),
                'model_predictions': predictions,
                'mape': mape
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

class StreamlitApp:
    """Streamlit interface for the car price predictor"""
    def __init__(self):
        st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
        self.model = None
    
    def run(self):
        """Run the Streamlit application"""
        st.title("Car Price Predictor")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload Car Data CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    st.error("The uploaded file is empty.")
                    return
                
                # Model configuration
                st.sidebar.subheader("Model Settings")
                fast_mode = st.sidebar.checkbox("Fast Mode", value=True)
                max_samples = st.sidebar.number_input(
                    "Max Samples (0 for all)", 
                    value=10000,
                    min_value=0
                )
                n_jobs = st.sidebar.slider(
                    "CPU Cores (-1 for all)", 
                    min_value=1,
                    max_value=8,
                    value=4
                )
                
                # Initialize model
                config = ModelConfig(
                    fast_mode=fast_mode,
                    max_samples=max_samples if max_samples > 0 else None,
                    n_jobs=n_jobs
                )
                self.model = CarPriceModel(config)
                
                # Vehicle selection
                st.header("Select Vehicle")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    makes = ["All"] + sorted(df['make'].unique())
                    make = st.selectbox("Make", options=makes)
                
                with col2:
                    models = ["All"]
                    if make != "All":
                        models.extend(sorted(df[df['make'] == make]['model'].unique()))
                    model = st.selectbox("Model", options=models)
                
                with col3:
                    trims = ["All"]
                    if make != "All" and model != "All":
                        trims.extend(sorted(df[
                            (df['make'] == make) & 
                            (df['model'] == model)
                        ]['trim'].unique()))
                    trim = st.selectbox("Trim", options=trims)
                
                # Display sample size
                filtered_df = self.model.filter_data(df, 
                    make if make != "All" else None,
                    model if model != "All" else None,
                    trim if trim != "All" else None
                )
                st.info(f"Selected vehicle sample size: {len(filtered_df):,}")
                
                # Train model
                if st.button("Train Model"):
                    with st.spinner("Training models..."):
                        metrics = self.model.train(df, 
                            make if make != "All" else None,
                            model if model != "All" else None,
                            trim if trim != "All" else None
                        )
                        self._display_metrics(metrics)
                
                # Show prediction interface if model is trained
                if self.model and self.model.metrics:
                    self._show_prediction_interface()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("Please upload a CSV file to begin.")
            
            
            
            
            
            
                                             
    def _display_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Display model performance metrics"""
        st.header("Model Performance")
        
        avg_metrics = {
            'RMSE': np.mean([m['rmse'] for m in metrics.values()]),
            'R²': np.mean([m['r2'] for m in metrics.values()]),
            'Error %': np.mean([m['mape'] for m in metrics.values()]) * 100
        }
        
        # Display metrics in columns
        cols = st.columns(len(avg_metrics))
        for col, (metric, value) in zip(cols, avg_metrics.items()):
            col.metric(
                metric,
                f"${value:,.0f}" if metric == 'RMSE' else 
                f"{value:.1%}" if metric == 'R²' else
                f"{value:.1f}%"
            )
        
        # Create detailed metrics visualization
        fig = go.Figure()
        for model_name, model_metrics in metrics.items():
            fig.add_trace(go.Bar(
                name=model_name,
                x=['RMSE', 'R²', 'Error %'],
                y=[
                    model_metrics['rmse'],
                    model_metrics['r2'],
                    model_metrics['mape'] * 100
                ]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _show_prediction_interface(self):
        """Display the prediction interface"""
        st.header("Predict Price")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
            odometer = st.number_input("Odometer (miles)", min_value=0, value=50000)
            condition = st.slider("Condition", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            
        with col2:
            body = st.selectbox("Body Style", options=['SUV', 'Sedan', 'Truck', 'Coupe', 'Wagon', 'Van', 'Convertible'])
            transmission = st.selectbox("Transmission", options=['Automatic', 'Manual'])
            color = st.selectbox("Exterior Color", options=['Black', 'White', 'Silver', 'Gray', 'Blue', 'Red', 'Other'])
            interior = st.selectbox("Interior Color", options=['Black', 'Gray', 'Tan', 'Other'])
            state = st.selectbox("State", options=['CA', 'TX', 'FL', 'NY', 'Other'])
        
        if st.button("Predict Price"):
            input_data = {
                'year': year,
                'odometer': odometer,
                'condition': condition,
                'body': body,
                'transmission': transmission,
                'color': color,
                'interior': interior,
                'state': state
            }
            
            with st.spinner("Calculating prediction..."):
                prediction = self.model.predict(input_data)
                
                # Display prediction results
                st.subheader("Prediction Results")
                
                # Main prediction
                st.metric(
                    "Predicted Price",
                    f"${prediction['predicted_price']:,.2f}",
                    delta=f"±{prediction['mape']*100:.1f}% margin of error"
                )
                
                # Display confidence and prediction intervals
                col1, col2 = st.columns(2)
                with col1:
                    st.write("95% Confidence Interval:")
                    st.write(f"${prediction['confidence_interval'][0]:,.2f} - ${prediction['confidence_interval'][1]:,.2f}")
                
                with col2:
                    st.write("Prediction Interval:")
                    st.write(f"${prediction['prediction_interval'][0]:,.2f} - ${prediction['prediction_interval'][1]:,.2f}")
                
                # Individual model predictions
                st.subheader("Individual Model Predictions")
                for model_name, pred in prediction['model_predictions'].items():
                    st.metric(f"{model_name.upper()} Model", f"${pred:,.2f}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()