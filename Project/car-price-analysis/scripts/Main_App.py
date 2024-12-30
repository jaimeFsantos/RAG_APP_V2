import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import from AI Chat and Price Prediction scripts
from security_storage_utils import SecurityManager, StorageManager, FileValidator
import os
from dotenv import load_dotenv
from AI_Chat_Analyst_Script import QASystem
from Pricing_Func import CarPricePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedCarApp:
    def __init__(self):
        try:
            # Initialize core components
            self.setup_basic_session_state()
            self.setup_page_config()
            
            # Initialize security components only if needed
            if self.should_initialize_security():
                self.initialize_security_components()
        except Exception as e:
            st.error(f"Error initializing app: {str(e)}")
            logging.error(f"Initialization error: {str(e)}")
    
    def should_initialize_security(self):
        """Check if security components should be initialized"""
        return os.getenv('ENABLE_SECURITY', 'false').lower() == 'true'
    
    def setup_basic_session_state(self):
        """Initialize basic session state variables"""
        # Common state variables
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
            
        # AI Chat state variables
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = None
            
        if 'chain' not in st.session_state:
            st.session_state.chain = None
            
        # Price Predictor state variables
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None
            
        if 'analyst' not in st.session_state:
            st.session_state.analyst = None
            
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
            
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
    
    def initialize_security_components(self):
        """Initialize security components if enabled"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Initialize security managers
            self.security_manager = SecurityManager()
            self.storage_manager = StorageManager()
            self.file_validator = FileValidator()
            
            # Initialize security session state
            st.session_state.authenticated = False
            st.session_state.login_attempts = 0
            st.session_state.last_activity = datetime.now()
            st.session_state.uploaded_files = {}
            
            logger.info("Security components initialized successfully")
                    
        except Exception as e:
            logger.error(f"Security initialization error: {str(e)}")
            st.error("Error initializing security components. Running in limited mode.")
            
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

    def initialize_qa_system(self):
        """Initialize the QA system with provided sources"""
        try:
            # Define sources with proper paths
            sources = [
                {"path": "Project\car-price-analysis\Sources\mmv.pdf", "type": "pdf"},
                {"path": "Project\car-price-analysis\Sources\autoconsumer.pdf", "type": "pdf"},
                {"path": "Project\car-price-analysis\Sources\car_prices.csv", "type": "csv",
                "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                            'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                            'seller', 'mmr', 'sellingprice', 'saledate']}
            ]

            # Create new QA system instance
            qa_system = QASystem(chunk_size=500, chunk_overlap=25)  # Reduced size for memory efficiency
            
            # Create the chain
            chain = qa_system.create_chain(sources)
            
            if chain is None:
                logger.error("Failed to create QA chain")
                return False
                
            # Only update session state if chain creation was successful
            st.session_state.qa_system = qa_system
            st.session_state.chain = chain
            logger.info("QA System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing QA System: {e}")
            return False

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
        uploaded_file = st.sidebar.file_uploader("Upload Car Data CSV", type=['csv'])
        
        # If security is enabled and file is uploaded
        if uploaded_file is not None and hasattr(self, 'security_manager'):
            try:
                # Validate file
                is_valid, message = self.file_validator.validate_file(uploaded_file)
                
                if not is_valid:
                    st.sidebar.error(message)
                    return page_selection, None
                
                # Store file securely
                file_path = self.storage_manager.upload_file(
                    uploaded_file, 
                    folder="car_data"
                )
                
                if file_path:
                    st.session_state.uploaded_files[uploaded_file.name] = file_path
                else:
                    st.sidebar.error("Error uploading file")
                    return page_selection, None
                    
            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")
                return page_selection, None
        
        return page_selection, uploaded_file
    
    def render_login(self):
        """Render login interface"""
        try:
            st.title("🔐 Login Required")
            
            if not hasattr(self, 'security_manager'):
                st.error("Security system not properly initialized")
                return False
            
            if not self.security_manager.can_attempt_login():
                wait_time = (st.session_state.next_attempt - datetime.now()).seconds
                st.error(f"Too many failed attempts. Please wait {wait_time} seconds.")
                return False
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if username == os.getenv("ADMIN_USERNAME") and \
                self.security_manager.verify_password(password, os.getenv("ADMIN_PASSWORD_HASH")):
                    st.session_state.authenticated = True
                    st.session_state.login_attempts = 0
                    st.success("Login successful!")
                    return True
                else:
                    self.security_manager.update_failed_login()
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

    def render_price_predictor(self, df: pd.DataFrame):
        """Render the price predictor interface from Pricing_Func"""
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
            
        try:
            # Verify required columns exist
            required_columns = ['make', 'model', 'trim', 'body', 'transmission', 
                            'state', 'condition', 'odometer', 'color', 'interior', 
                            'sellingprice']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Fill NA values for categorical columns
            categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 
                                 'state', 'color', 'interior', 'seller']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('unknown')
            
            # Fill NA values for numeric columns with median
            numeric_columns = ['year', 'condition', 'odometer', 'sellingprice']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            # Update unique values for the predictor
            st.session_state.predictor.update_unique_values(df)
            
            # Vehicle selection interface
            st.subheader("Select Vehicle")
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
            
            # Filter data for selected vehicle and ensure it's a DataFrame
            filter_condition = (
                (df['make'].fillna('').eq(make)) &
                (df['model'].fillna('').eq(model)) &
                (df['trim'].fillna('').eq(trim))
            )
            
            filtered_data = pd.DataFrame(df[filter_condition])
            
            if len(filtered_data) == 0:
                st.warning("No data available for the selected vehicle combination.")
                return
                
            st.info(f"Number of samples for this vehicle: {len(filtered_data)}")
            # Model training section
            if len(filtered_data) > 5:  # Minimum samples needed for training
                if st.button("Train Models", type="primary"):
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            # Convert filtered_data to DataFrame if it's not already
                            filtered_data = pd.DataFrame(filtered_data)
                            
                            # Prepare and engineer features
                            df_processed = st.session_state.predictor.prepare_data(filtered_data)
                            
                            # Ensure df_processed is a DataFrame
                            if not isinstance(df_processed, pd.DataFrame):
                                df_processed = pd.DataFrame(df_processed)
                                
                            # Engineer features
                            df_engineered = st.session_state.predictor.engineer_features(df_processed)
                            
                            # Ensure proper column handling for feature selection
                            drop_cols = ['sellingprice']
                            if 'mmr' in df_engineered.columns:
                                drop_cols.append('mmr')
                            
                            # Split features and target
                            X = df_engineered.drop(columns=[col for col in drop_cols if col in df_engineered.columns])
                            y = df_engineered['sellingprice']
                            
                            # Remove multicollinearity
                            X = st.session_state.predictor.remove_multicollinearity(X)
                            
                            # Train-test split
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Fit and evaluate models
                            st.session_state.predictor.fit(X_train, y_train)
                            metrics, predictions = st.session_state.predictor.evaluate(X_test, y_test)
                            
                            # Store in session state
                            st.session_state.metrics = metrics
                            st.session_state.model_trained = True
                            
                            st.success("Models trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
            else:
                st.warning("Not enough samples to train models. Please select a different vehicle with more data.")
            
            # Display model performance metrics if trained
            if st.session_state.model_trained and 'metrics' in st.session_state:
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
            
            # Price estimator section
            if st.session_state.model_trained:
                st.subheader("Price Estimator")
                
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
                    with st.spinner("Calculating price estimate..."):
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
                            
                            prediction_result = st.session_state.predictor.create_what_if_prediction(input_data)
                            
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
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error in price prediction: {str(e)}")

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
                            # Get response and determine visualization type
                            response = st.session_state.chain.invoke(prompt)
                            if response is None:
                                st.error("Unable to generate response. Please try again.")
                                return
                            
                            # Analyze query to determine visualization type
                            viz_type = self._determine_visualization_type(prompt.lower())
                            
                            # Generate response
                            st.markdown(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            
                            # Update visualization if needed
                            if viz_type:
                                self._update_visualization(viz_type, viz_col)
                                
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            st.error("Error generating response. Please try again.")

        # Initialize visualization column
        with viz_col:
            st.empty()  # Placeholder for visualizations

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
        # Only check authentication if security is enabled
        if hasattr(self, 'security_manager'):
            if not st.session_state.authenticated:
                if not self.render_login():
                    return
            
            # Check session timeout
            if self.security_manager.check_session_timeout():
                st.warning("Session expired. Please login again.")
                st.session_state.authenticated = False
                return

        # Regular app flow
        page, uploaded_file = self.render_sidebar()
        
        # Load data if uploaded
        df = None
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                logger.info(f"Successfully loaded data with shape: {df.shape}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
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