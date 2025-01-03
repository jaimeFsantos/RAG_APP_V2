"""
Main Application Module

Implements the main application logic and integrates all components
into a unified interface.

Environment:
    AWS EC2 Free Tier

Components:
    - Combined interface for all features
    - Security integration
    - File handling
    - State management
    - Visualization rendering
"""

# Standard Library Imports
import logging
import warnings
from datetime import datetime
from io import BytesIO
from typing import Dict, Any
import pytz
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split

# Third-Party Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from AI_Chat_Analyst_Script import EnhancedQASystem
from Pricing_Func import CarPricePredictor
from car_viz_dashboard import CarVizDashboard
from dashboard_config import DashboardConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enhanced_security_audit import (EnhancedSecurityManager,
    SecurityMode, 
    AuditEventType,
    audit_trail
)

class CombinedCarApp:
    def __init__(self):
        try:
            # Initialize basic session state
            if 'predictor' not in st.session_state:
                st.session_state.predictor = None
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            if 'qa_system' not in st.session_state:
                st.session_state.qa_system = None
            if 'chain' not in st.session_state:
                st.session_state.chain = None
            if 'model_trained' not in st.session_state:
                st.session_state.model_trained = False
                
            self._initialize_session_state()
                
            self.is_cloud = self._is_cloud_environment()
            self.initialize_services()
            # Initialize storage service using factory
            from storage_service import get_storage_service
            self.storage_service = get_storage_service()
            
            # Initialize security manager
            is_ec2 = os.getenv('AWS_EXECUTION_ENV', '').startswith('AWS_ECS')
            mode = SecurityMode.EC2 if is_ec2 else SecurityMode.LOCAL
            self.security_manager = EnhancedSecurityManager(mode=mode)
            
            self.initialize_security_components()
            self.setup_page_config()
            
            config = DashboardConfig()
            self.dashboard = CarVizDashboard(is_cloud=config.is_cloud)
            
            logger.info(f"Application initialized in {mode.value} mode")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            st.error(f"Error initializing app: {str(e)}")
            
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = {}
        if 'qa_initialized' not in st.session_state:
            st.session_state.qa_initialized = False
        if 'storage_initialized' not in st.session_state:
            st.session_state.storage_initialized = False

    def _is_cloud_environment(self) -> bool:
        """Detect if running in cloud environment"""
        # Check for AWS EC2 environment
        is_ec2 = os.getenv('AWS_EXECUTION_ENV', '').startswith('AWS_ECS')
        # Check for explicit cloud mode flag
        cloud_mode = os.getenv('CLOUD_MODE', '').lower() == 'true'
        return is_ec2 or cloud_mode

    def initialize_services(self):
        """Initialize services based on environment"""
        try:
            # Initialize storage service
            if self.is_cloud:
                from storage_service import CloudStorageService
                self.storage_service = CloudStorageService()
                logger.info("Cloud storage service initialized")
            else:
                from storage_service import LocalStorageService
                self.storage_service = LocalStorageService()
                logger.info("Local storage service initialized")
                
            st.session_state.storage_initialized = True
            
        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            raise

    def handle_file_upload(self, file_obj, file_type: str) -> str:
        """Handle file upload with environment awareness"""
        if not st.session_state.storage_initialized:
            raise RuntimeError("Storage not initialized")
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.is_cloud:
                # Cloud path
                file_path = f"uploads/{timestamp}_{file_obj.name}"
            else:
                # Local path
                file_path = f"local_storage/uploads/{timestamp}_{file_obj.name}"
                
            # Store file
            stored_path = self.storage_service.store_file(
                file_obj.getvalue(), 
                file_path
            )
            
            # Update session state
            st.session_state.uploaded_files[file_obj.name] = stored_path
            logger.info(f"Stored file at: {stored_path}")
            
            # Initialize/update QA system
            self.initialize_qa_system()
            
            return stored_path
            
        except Exception as e:
            logger.error(f"File upload error: {e}")
            raise

    # In CombinedCarApp class
    def initialize_qa_system(self):
        """Initialize QA system with cloud integration and proper error handling"""
        try:
            logger.info("Starting QA system initialization")
        
            if not hasattr(st.session_state, 'uploaded_files') or not st.session_state.uploaded_files:
                logger.error("No uploaded files found in session state")
                return False
            
            sources = []
            
            # Add CSV source if available with cloud path handling
            csv_files = {k: v for k, v in st.session_state.uploaded_files.items() if k.endswith('.csv')}
            if csv_files:
                most_recent_csv = max(csv_files.items(), key=lambda x: x[1])[1]
                source_path = most_recent_csv
                if self.is_cloud:
                    source_path = f"s3://{os.getenv('AWS_BUCKET_NAME')}/{most_recent_csv}"
                
                sources.append({
                    "path": source_path,
                    "type": "csv",
                    "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                            'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                            'seller', 'mmr', 'sellingprice', 'saledate']
                })
                logger.info(f"Added CSV source: {source_path}")
                
            if not sources:
                logger.error("No valid sources found")
                return False
                
            # Initialize QA system with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    qa_system = EnhancedQASystem(chunk_size=500, chunk_overlap=25)  # Reduced chunk size for EC2
                    chain = qa_system.create_chain(sources)
                    
                    if chain is None:
                        raise ValueError("Chain creation failed")
                        
                    st.session_state.qa_system = qa_system
                    st.session_state.chain = chain
                    logger.info("QA System initialized successfully")
                    return True
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to initialize QA system after {max_retries} attempts: {e}")
                        return False
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                    
        except Exception as e:
            logger.error(f"Error in QA system initialization: {e}")
            return False

    def initialize_security_components(self):
        """Initialize security components if not already done"""
        try:
            if 'security_manager' not in st.session_state:
                st.session_state.security_manager = self.security_manager
            
            # Initialize security session state
            if 'authenticated' not in st.session_state:
                st.session_state.authenticated = False
                st.session_state.login_attempts = 0
                st.session_state.last_activity = datetime.now()
                st.session_state.uploaded_files = {}
            
            logger.info("Security components initialized successfully")
                    
        except Exception as e:
            logger.error(f"Security initialization error: {str(e)}")
            st.error("Error initializing security components. Running in limited mode.")
            
    def cleanup_temp_files(self):
        """Cleanup temporary files periodically"""
        try:
            for file in os.listdir(self.temp_storage):
                if file.startswith('car_analysis_'):
                    os.remove(os.path.join(self.temp_storage, file))
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
    
    def should_initialize_security(self):
        """Check if security components should be initialized"""
        return os.getenv('ENABLE_SECURITY', 'false').lower() == 'true'
    def setup_page_config(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="🚗 Car Analysis Suite",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for consistent styling
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
            }
            .prediction-card {
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid #e0e0e0;
                margin: 1rem 0;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            .user-message {
                background-color: #f0f2f6;
            }
            .assistant-message {
                background-color: #e8f0fe;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and file upload"""
        st.sidebar.title("Navigation")
        
        
        pages = {
            "Home": "🏠",
            "Price Predictor": "💰",
            "AI Chat Assistant": "💭",
            "Data Analysis": "📊"
        }
        
        page_selection = st.sidebar.radio(
            "Go to",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]} {x}"
        )
        
        st.sidebar.header("Data Upload")
        
        # Create separate uploaders for CSV and PDF
        csv_file = st.sidebar.file_uploader("Upload Car Data (CSV)", type=['csv'], key='csv_uploader')
        pdf_files = st.sidebar.file_uploader("Upload Documentation (PDF)", 
                                        type=['pdf'], 
                                        accept_multiple_files=True,
                                        key='pdf_uploader')
        
        uploaded_files = []
        
        try:
            if csv_file is not None:
                if csv_file.size > 100 * 1024 * 1024:  # 100MB limit
                    st.sidebar.error("CSV file size too large. Maximum size is 100MB.")
                else:
                    file_path = self.storage_service.store_file(csv_file.getvalue(), csv_file.name)
                    st.session_state.uploaded_files[csv_file.name] = file_path
                    uploaded_files.append({"path": file_path, "type": "csv"})
                    logger.info(f"Stored CSV file: {file_path}")
            
            if pdf_files:
                for pdf_file in pdf_files:
                    if pdf_file.size > 50 * 1024 * 1024:  # 50MB limit per PDF
                        st.sidebar.error(f"PDF file {pdf_file.name} too large. Maximum size is 50MB.")
                        continue
                        
                    file_path = self.storage_service.store_file(pdf_file.getvalue(), pdf_file.name)
                    st.session_state.uploaded_files[pdf_file.name] = file_path
                    uploaded_files.append({"path": file_path, "type": "pdf"})
                    logger.info(f"Stored PDF file: {file_path}")
            
            # Show currently loaded files
            if st.session_state.uploaded_files:
                st.sidebar.subheader("Loaded Files")
                for filename, path in st.session_state.uploaded_files.items():
                    st.sidebar.text(f"✓ {filename}")
                    
        except Exception as e:
            logger.error(f"Error processing uploaded files: {e}")
            st.sidebar.error("Error processing uploaded files")
            
        return page_selection, uploaded_files
    
    def render_login(self):
        """Render login interface"""
        try:
            st.title("🔐 Login Required")
            
            if not hasattr(self, 'security_manager'):
                st.error("Security system not properly initialized")
                return False
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                # Initialize session state first
                if 'session_start' not in st.session_state:
                    st.session_state.session_start = datetime.now()
                st.session_state.last_activity = datetime.now()  # Update activity time
                
                if self.security_manager.config.mode == SecurityMode.LOCAL:
                    # Use local config credentials
                    is_valid = (username == 'admin' and password == 'admin')  # Simplified for testing
                else:
                    # Use environment variables for EC2
                    is_valid = (username == os.getenv("ADMIN_USERNAME") and 
                            self.security_manager.verify_password(password, 
                            os.getenv("ADMIN_PASSWORD_HASH")))

                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.last_activity = datetime.now()  # Update again after successful login
                    st.success("Login successful!")
                    time.sleep(1)  # Give a moment for the success message
                    st.experimental_rerun()  # Force a clean rerun after login
                    return True
                else:
                    st.error("Invalid credentials")
                    return False
                        
            return False
            
        except Exception as e:
            logger.error(f"Error in login process: {str(e)}")
            st.error("Login system error")
            return False

    def render_home(self):
        """Render the home page"""
        st.title("🚗 Car Analysis Suite")
        
        st.markdown("""
            ### Welcome to the Car Analysis Suite!
            
            This comprehensive platform combines three powerful tools:
            
            1. **💰 Price Prediction Engine**
               - Get accurate car valuations
               - Analyze price factors
               - View confidence intervals
            
            2. **💭 AI Chat Assistant**
               - Ask questions about cars and market trends
               - Get detailed insights
               - Explore market analysis
            
            3. **📊 Data Analysis Dashboard**
               - Visualize market trends
               - Compare models and makes
               - Track price patterns
            
            To begin, please upload your data using the sidebar.
        """)

    @audit_trail(AuditEventType.MODEL_TRAINING)
    def render_price_predictor(self, df: pd.DataFrame):
        """
        Render the price predictor interface with integrated chat analysis
        
        Args:
            df (pd.DataFrame): Input data for price prediction
        """
        try:
            st.header("💰 Car Price Predictor")
            
            if df is None:
                st.warning("Please upload data to use the price predictor.")
                return
                
            # Initialize predictor if needed
            if st.session_state.predictor is None:
                st.session_state.predictor = CarPricePredictor(
                    models=['rf', 'gbm'],
                    fast_mode=st.sidebar.checkbox("Fast Mode", value=True)
                )
            
            # Verify required columns
            required_columns = [
                'make', 'model', 'trim', 'body', 'transmission', 
                'state', 'condition', 'odometer', 'color', 'interior', 'sellingprice'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Data preparation
            try:
                # Fill missing values
                df = self._prepare_predictor_data(df)
                
                # Update unique values
                st.session_state.predictor.update_unique_values(df)
            except Exception as e:
                logger.error(f"Error preparing data: {e}")
                st.error("Error preparing data for prediction")
                return

            # Vehicle selection interface
            st.subheader("Select Vehicle")
            try:
                make, model, trim = self._render_vehicle_selector(df)
                filtered_data = self._get_filtered_data(df, make, model, trim)
                
                if len(filtered_data) == 0:
                    st.warning("No data available for the selected vehicle combination.")
                    return
                    
                st.info(f"Number of samples for this vehicle: {len(filtered_data)}")
            except Exception as e:
                logger.error(f"Error in vehicle selection: {e}")
                st.error("Error selecting vehicle")
                return

            # Model training section
            if len(filtered_data) > 5:  # Minimum samples needed
                if st.button("Train Models", type="primary"):
                    self._train_predictor_models(filtered_data)
            else:
                st.warning("Not enough samples to train models. Please select a different vehicle with more data.")

            # Display model performance metrics
            if st.session_state.model_trained and 'metrics' in st.session_state:
                self._display_model_metrics()

            # Price estimator section
            if st.session_state.model_trained:
                st.subheader("Price Estimator")
                price_estimate = self._render_price_estimator()
                
                # Display prediction if made
                if price_estimate:
                    self._display_prediction_results(price_estimate)
                    
                    # AI Chat Integration
                    st.markdown("---")
                    self._render_predictor_chat(price_estimate)
                    
        except Exception as e:
            logger.error(f"Error in price predictor: {e}")
            st.error("An error occurred. Please try again.")

    def _prepare_predictor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction with proper error handling"""
        categorical_columns = [
            'make', 'model', 'trim', 'body', 'transmission', 
            'state', 'color', 'interior', 'seller'
        ]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')

        numeric_columns = ['year', 'condition', 'odometer', 'sellingprice']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def _render_vehicle_selector(self, df: pd.DataFrame):
        """Render vehicle selection interface"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            make = st.selectbox("Make", options=sorted(df['make'].unique()))
        
        filtered_models = df[df['make'] == make]['model'].unique()
        with col2:
            model = st.selectbox("Model", options=sorted(filtered_models))
        
        filtered_trims = df[
            (df['make'] == make) & 
            (df['model'] == model)
        ]['trim'].unique()
        with col3:
            trim = st.selectbox("Trim", options=sorted(filtered_trims))
            
        return make, model, trim

    def _get_filtered_data(self, df: pd.DataFrame, make: str, model: str, trim: str) -> pd.DataFrame:
        """Get filtered data for selected vehicle"""
        filter_condition = (
            (df['make'].fillna('').eq(make)) &
            (df['model'].fillna('').eq(model)) &
            (df['trim'].fillna('').eq(trim))
        )
        return pd.DataFrame(df[filter_condition])

    def _train_predictor_models(self, filtered_data: pd.DataFrame):
        """Train prediction models with proper error handling"""
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Prepare and engineer features
                df_processed = st.session_state.predictor.prepare_data(filtered_data)
                df_engineered = st.session_state.predictor.engineer_features(df_processed)
                
                # Split features and target
                drop_cols = ['sellingprice']
                if 'mmr' in df_engineered.columns:
                    drop_cols.append('mmr')
                
                X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])
                y = df_engineered['sellingprice']
                
                # Remove multicollinearity
                X = st.session_state.predictor.remove_multicollinearity(X)
                
                # Train test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Fit and evaluate models
                st.session_state.predictor.fit(X_train, y_train)
                metrics, predictions = st.session_state.predictor.evaluate(X_test, y_test)
                
                st.session_state.metrics = metrics
                st.session_state.model_trained = True
                
                st.success("Models trained successfully!")
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                st.error(f"Error during model training: {str(e)}")

    def _display_model_metrics(self):
        """Display model performance metrics"""
        st.subheader("Model Performance")
        
        avg_metrics = {
            'RMSE': np.mean([m['rmse'] for m in st.session_state.metrics.values()]),
            'R²': np.mean([m['r2'] for m in st.session_state.metrics.values()]),
            'Error %': np.mean([m['mape'] for m in st.session_state.metrics.values()]) * 100
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Error", f"{avg_metrics['Error %']:.1f}%")
        with col2:
            st.metric("RMSE", f"${avg_metrics['RMSE']:,.0f}")
        with col3:
            st.metric("R² Score", f"{avg_metrics['R²']:.3f}")

    def _render_price_estimator(self):
        """Render price estimation interface"""
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
            condition = st.number_input("Condition (1-50)", min_value=1.0, max_value=50.0, value=25.0, step=1.0)
            odometer = st.number_input("Mileage", min_value=0, value=50000, step=1000)
            state = st.selectbox("State", options=st.session_state.predictor.unique_values['state'])
        
        with col2:
            body = st.selectbox("Body Style", options=st.session_state.predictor.unique_values['body'])
            transmission = st.selectbox("Transmission", options=st.session_state.predictor.unique_values['transmission'])
            color = st.selectbox("Color", options=st.session_state.predictor.unique_values['color'])
            interior = st.selectbox("Interior", options=st.session_state.predictor.unique_values['interior'])
        
        if st.button("Get Price Estimate", type="primary"):
            return self._generate_price_estimate(
                year, condition, odometer, state, 
                body, transmission, color, interior
            )
        return None

    def _generate_price_estimate(self, year, condition, odometer, state, 
                            body, transmission, color, interior):
        """Generate price estimate with error handling"""
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
            
            return st.session_state.predictor.create_what_if_prediction(input_data)
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            st.error("Error generating price estimate")
            return None

    def _display_prediction_results(self, prediction_result: dict):
        """Display prediction results"""
        mean_price = prediction_result['predicted_price']
        low_estimate, high_estimate = prediction_result['prediction_interval']
        
        st.subheader("Price Estimates")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Low Estimate", f"${low_estimate:,.0f}")
        with col2:
            st.metric("Best Estimate", f"${mean_price:,.0f}")
        with col3:
            st.metric("High Estimate", f"${high_estimate:,.0f}")
        
        st.info(f"Estimated error margin: ±{prediction_result['mape']*100:.1f}%")
        
        # Store prediction result for chat context
        st.session_state.last_prediction_result = prediction_result

    def _render_predictor_chat(self, prediction_result: dict):
        """Render chat interface for prediction analysis"""
        st.subheader("💭 AI Analysis")
        
        # Initialize QA system if needed
        if not hasattr(st.session_state, 'qa_system') or st.session_state.qa_system is None:
            self._initialize_qa_system()
        
        # Create chat interface
        if st.session_state.qa_system and hasattr(st.session_state.qa_system, 'chain'):
            chat_col, context_col = st.columns([2, 1])
            
            with chat_col:
                self._render_chat_messages()
                
            with context_col:
                self._render_prediction_context(prediction_result)
                            
                            
    @audit_trail(AuditEventType.CHAT_INTERACTION)
    def render_chat_assistant(self):
        """Render the AI chat assistant interface with visualizations"""
        st.header("💭 AI Chat Assistant")
        
        # Initialize QA system if needed
        if st.session_state.qa_system is None:
            with st.spinner("Initializing chat system..."):
                if not self.initialize_qa_system(): 
                    st.error("Could not initialize chat system. Please try again.")
                    return

        # Check if chain is properly initialized
        if st.session_state.chain is None:
            st.error("Chat system not properly initialized. Please refresh the page.")
            return

        # Create two columns - one for chat, one for visualizations
        chat_col, viz_col = st.columns([2, 1])

        with chat_col:
            # Chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            if prompt := st.chat_input("Ask me anything about cars..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response_data = st.session_state.qa_system.generate_response(prompt)
                            if isinstance(response_data, dict):
                                st.markdown(response_data['text'])
                                if response_data.get('visualization'):
                                    with viz_col:
                                        st.plotly_chart(
                                            response_data['visualization'],
                                            use_container_width=True
                                        )
                                response_text = response_data['text']
                            else:
                                # Handle case where we got a direct string response
                                st.markdown(response_data)
                                response_text = response_data
                                
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text
                            })
                        
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            st.error("Error generating response. Please try again.")

        # Initialize visualization column
        with viz_col:
            st.empty()  # Placeholder for visualizations that will be updated
            
            
    def _render_chat_messages(self):
        """Render chat message history and input"""
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about this prediction or market insights..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    response = st.session_state.qa_system.chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    st.error("Failed to generate response. Please try again.")

    def _render_prediction_context(self, prediction_result: dict):
        """Display prediction context in sidebar"""
        st.info("Current Prediction Context")
        st.write(f"Predicted Price: ${prediction_result['predicted_price']:,.2f}")
        st.write(f"Confidence: ±{prediction_result['mape']*100:.1f}%")
        
        if 'model_predictions' in prediction_result:
            st.write("Model Predictions:")
            for model, price in prediction_result['model_predictions'].items():
                st.write(f"- {model}: ${price:,.2f}")

    def _determine_visualization_type(self, query: str) -> str:
        """Determine the type of visualization needed based on the query"""
        if any(keyword in query for keyword in ['trend', 'price', 'cost', 'value', 'historical']):
            return 'price_trends'
        elif any(keyword in query for keyword in ['feature', 'factor', 'impact', 'influence']):
            return 'feature_importance'
        elif any(keyword in query for keyword in ['market', 'segment', 'compare', 'analysis']):
            return 'market_analysis'
        return None

    def _update_visualization(self, viz_type: str, viz_container):
        """Update the visualization based on the query type"""
        try:
            if viz_type == 'price_trends':
                data = self._get_price_trends_data()
                self._render_price_trends(data, viz_container)
            elif viz_type == 'feature_importance':
                if hasattr(st.session_state, 'predictor') and st.session_state.predictor:
                    data = self._get_feature_importance_data()
                    self._render_feature_importance(data, viz_container)
            elif viz_type == 'market_analysis':
                data = self._get_market_analysis_data()
                self._render_market_analysis(data, viz_container)
        except Exception as e:
            viz_container.error(f"Error updating visualization: {str(e)}")

    def _get_price_trends_data(self):
        """Extract price trends data from the dataset"""
        if not hasattr(st.session_state, 'total_data'):
            return None
            
        df = st.session_state.total_data
        df['date'] = pd.to_datetime(df['saledate'])
        monthly_prices = df.groupby(df['date'].dt.strftime('%Y-%m'))[['sellingprice']].mean()
        return monthly_prices.to_dict()['sellingprice']

    def _get_feature_importance_data(self):
        """Get feature importance data from the trained model"""
        if not hasattr(st.session_state, 'predictor') or not st.session_state.predictor:
            return None
            
        predictor = st.session_state.predictor
        if 'rf' in predictor.best_models:
            return {
                feature: importance 
                for feature, importance in zip(
                    predictor.feature_columns,
                    predictor.best_models['rf'].feature_importances_
                )
            }
        return None

    def _get_market_analysis_data(self):
        """Get market analysis data"""
        if not hasattr(st.session_state, 'total_data'):
            return None
            
        df = st.session_state.total_data
        return {
            'make': df['make'].value_counts().head(10).to_dict(),
            'body_style': df['body'].value_counts().to_dict(),
            'transmission': df['transmission'].value_counts().to_dict()
        }

    def _render_price_trends(self, data, container):
        """Render price trends visualization"""
        if not data:
            return
            
        fig = go.Figure()
        dates = list(data.keys())
        prices = list(data.values())
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Average Price'
        ))
        
        fig.update_layout(
            title='Price Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Average Price ($)',
            height=400
        )
        
        container.plotly_chart(fig, use_container_width=True)

    def _render_feature_importance(self, data, container):
        """Render feature importance visualization"""
        if not data:
            return
            
        # Sort features by importance
        sorted_features = dict(sorted(data.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=list(sorted_features.keys()),
            x=list(sorted_features.values()),
            orientation='h'
        ))
        
        fig.update_layout(
            title='Top 10 Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400
        )
        
        container.plotly_chart(fig, use_container_width=True)
        
# In Main_App.py, modify the render_data_analysis method:

    def render_data_analysis(self, df):
        """Render data analysis dashboard with integrated chat analysis"""
        st.header("📊 Data Analysis Dashboard")
        
        if df is None:
            st.warning("Please upload data using the sidebar to view analytics.")
            return
            
        try:
            # Show loading state while preparing data
            with st.spinner("Processing data for visualization..."):
                # Generate cloud storage path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                storage_path = f"analytics/car_data_{timestamp}.csv"
                
                # Upload data to storage
                try:
                    # Convert DataFrame to CSV string
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    # Upload to storage service
                    file_path = self.storage_service.store_file(csv_data, storage_path)
                    
                    # Create tabs for Dashboard and Chat
                    dash_tab, chat_tab = st.tabs(["📈 Dashboard", "💬 AI Analysis"])
                    
                    with dash_tab:
                        # Display basic stats
                        st.write("Dataset Overview:")
                        cols = st.columns(3)
                        cols[0].metric("Total Records", f"{len(df):,}")
                        cols[1].metric("Average Price", f"${df['sellingprice'].mean():,.2f}")
                        cols[2].metric("Unique Models", f"{df['model'].nunique():,}")
                        
                        # Render dashboard with cloud path
                        self.dashboard.render_dashboard(file_path)
                    
                    with chat_tab:
                        # Initialize QA system if needed
                        if not hasattr(st.session_state, 'qa_system') or st.session_state.qa_system is None:
                            self._initialize_qa_system()
                        
                        # Convert saledate to datetime and handle numeric data safely
                        try:
                            df['saledate'] = pd.to_datetime(df['saledate'])
                            date_range = f"{df['saledate'].min().strftime('%Y-%m-%d')} to {df['saledate'].max().strftime('%Y-%m-%d')}"
                        except Exception as e:
                            logger.warning(f"Error processing dates: {e}")
                            date_range = "Date range unavailable"
                        
                        # Safely get numeric values
                        try:
                            price_min = df['sellingprice'].min()
                            price_max = df['sellingprice'].max()
                            price_range = f"${price_min:,.2f} to ${price_max:,.2f}"
                        except Exception as e:
                            logger.warning(f"Error processing prices: {e}")
                            price_range = "Price range unavailable"
                        
                        # Add dashboard metrics to QA context
                        dashboard_context = {
                            'total_records': len(df),
                            'avg_price': float(df['sellingprice'].mean()),
                            'unique_models': int(df['model'].nunique()),
                            'date_range': date_range,
                            'price_range': price_range
                        }
                        
                        # Create chat columns
                        chat_col, viz_col = st.columns([2, 1])
                        
                        with chat_col:
                            self._render_chat_messages()
                            
                        with viz_col:
                            # Display dashboard context
                            st.info("Dashboard Insights")
                            st.write("Dataset Summary:")
                            st.write(f"• Records: {dashboard_context['total_records']:,}")
                            st.write(f"• Average Price: ${dashboard_context['avg_price']:,.2f}")
                            st.write(f"• Models: {dashboard_context['unique_models']:,}")
                            st.write(f"• Date Range: {dashboard_context['date_range']}")
                            st.write(f"• Price Range: {dashboard_context['price_range']}")
                    
                except Exception as e:
                    logger.error(f"Storage error: {str(e)}")
                    st.error("Error storing visualization data")
                    return
                    
        except Exception as e:
            logger.error(f"Dashboard error: {str(e)}")
            st.error("Error rendering dashboard")

    def _render_market_analysis(self, data, container):
        """Render market analysis visualization"""
        if not data:
            return
            
        # Create tabs for different market aspects
        tabs = container.tabs(['Makes', 'Body Styles', 'Transmissions'])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(data['make'].keys()),
                y=list(data['make'].values())
            ))
            fig.update_layout(title='Top Makes by Count', height=400)
            container.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(data['body_style'].keys()),
                y=list(data['body_style'].values())
            ))
            fig.update_layout(title='Body Styles Distribution', height=400)
            container.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(data['transmission'].keys()),
                y=list(data['transmission'].values())
            ))
            fig.update_layout(title='Transmission Types Distribution', height=400)
            container.plotly_chart(fig, use_container_width=True)
            
    def __del__(self):
        """Cleanup when app instance is deleted"""
        try:
            self.storage_manager.cleanup_cache()
        except:
            pass

    def run(self):
        """Main application loop with optional security"""
        # Initialize session state if needed
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.authenticated = False
            st.session_state.last_activity = datetime.now()
        
        # Only check authentication if security is enabled
        if hasattr(self, 'security_manager'):
            # First check if already authenticated
            if not st.session_state.authenticated:
                if not self.render_login():
                    return
            else:
                # Only check timeout if already authenticated
                if (datetime.now() - st.session_state.last_activity) > timedelta(hours=12):
                    st.warning("Session expired. Please login again.")
                    st.session_state.authenticated = False
                    st.session_state.last_activity = datetime.now()
                    st.experimental_rerun()
                    return
                
                # Update activity timestamp
                st.session_state.last_activity = datetime.now()

        # Regular app flow
        page, uploaded_files = self.render_sidebar()
        
    # Rest of your existing run() code...
        # Load data if uploaded
        df = None
        if uploaded_files:
            try:
                # Find the CSV file in uploaded files
                csv_file = next((f for f in uploaded_files if f["type"] == "csv"), None)
                if csv_file:
                    df = pd.read_csv(csv_file["path"])
                    logger.info(f"Successfully loaded data with shape: {df.shape}")
                    
                    # Initialize QA system if not already done
                    if not st.session_state.initialized:
                        with st.spinner("Initializing AI system..."):
                            success = self.initialize_qa_system()
                            if success:
                                st.session_state.initialized = True
                                st.success("System initialized successfully!")
                            else:
                                st.error("Failed to initialize system. Please try again.")
                                return
                                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error loading data: {str(e)}")
                return
        
        # Render selected page
        if page == "Home":
            self.render_home()
        elif page == "Price Predictor":
            self.render_price_predictor(df)
        elif page == "AI Chat Assistant":
            self.render_chat_assistant()
        elif page == "Data Analysis":
            self.render_data_analysis(df)

if __name__ == "__main__":
    app = CombinedCarApp()
    app.run()