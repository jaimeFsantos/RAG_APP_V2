import streamlit as st
import logging
from AI_Chat_Analyst_Script import QASystem
import json  # Add this import
import os
import boto3


# Set up logging
logging.basicConfig(level=logging.DEBUG)    
logger = logging.getLogger(__name__)

def create_visualization_data(qa_system, query, response):
    """Create visualization data based on the query and QA system state"""
    try:
        if not hasattr(qa_system, 'market_insights'):
            return None

        market_data = {
            'market_summary': qa_system.market_insights.get('market_summary', {}),
            'feature_impact': qa_system.market_insights.get('feature_impact', {})
        }

        # Process market summary data
        if 'market_summary' in market_data:
            summary = market_data['market_summary']
            if isinstance(summary, dict):
                market_data['market_summary'] = {
                    'median_price': float(summary.get('median_price', 0)),
                    'popular_colors': summary.get('popular_colors', {}),
                    'transmission_split': summary.get('transmission_split', {})
                }

        # Process feature impact data
        if 'feature_impact' in market_data:
            feature_impact = {}
            for feature, analysis in market_data['feature_impact'].items():
                if analysis:
                    impact_values = []
                    for segment in analysis[:5]:  # Top 5 segments
                        for value, impact in segment.get('impacts', {}).items():
                            impact_values.append({
                                'value': value,
                                'price_premium': impact.get('price_premium', 0)
                            })
                    feature_impact[feature] = impact_values
            market_data['feature_impact'] = feature_impact

        return market_data

    except Exception as e:
        logger.error(f"Error creating visualization data: {e}")
        return None

# Initialize Streamlit session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
    st.session_state.chain = None

if 'last_query' not in st.session_state:
    st.session_state.last_query = None

# Initialize the QASystem
def initialize_qa_system():
    """Initialize the QA system with error handling and storage fallback"""
    try:
        # Configure sources with fallback paths
        sources = []
        
        # Add cloud sources if available
        s3_client = get_s3_client()  # Helper function to get S3 client
        if s3_client:
            try:
                # Try to get sources from S3
                sources.extend(get_cloud_sources(s3_client))
            except Exception as cloud_err:
                logger.warning(f"Could not access cloud sources: {cloud_err}")
                
        # Add local sources as fallback
        local_sources = [
            {"path": "local_storage/car_prices.csv", "type": "csv",
             "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                        'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                        'seller', 'mmr', 'sellingprice', 'saledate']},
            {"path": "local_storage/mmv.pdf", "type": "pdf"},
            {"path": "local_storage/autoconsumer.pdf", "type": "pdf"}
        ]
        sources.extend([s for s in local_sources if os.path.exists(s["path"])])
        
        if not sources:
            raise ValueError("No valid sources found in cloud or local storage")

        # Initialize QA system with proper checks
        st.session_state.qa_system = QASystem(chunk_size=1000, chunk_overlap=50)
        
        # Create chain with explicit error handling
        chain = st.session_state.qa_system.create_chain(sources)
        if chain is None:
            raise ValueError("Failed to create QA chain")
            
        st.session_state.chain = chain
        logger.info("QA System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing QA System: {e}")
        st.error("Failed to initialize QA system. Please check logs.")
        return False

def get_s3_client():
    """Get S3 client with proper error handling"""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
    except Exception as e:
        logger.warning(f"Could not initialize S3 client: {e}")
        return None

def get_cloud_sources(s3_client):
    """Get sources from S3 with error handling"""
    cloud_sources = []
    try:
        # List objects in the source bucket
        bucket = os.getenv('S3_BUCKET_NAME')
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='sources/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):
                cloud_sources.append({
                    "path": f"s3://{bucket}/{key}",
                    "type": "csv",
                    "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                               'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                               'seller', 'mmr', 'sellingprice', 'saledate']
                })
            elif key.endswith('.pdf'):
                cloud_sources.append({
                    "path": f"s3://{bucket}/{key}",
                    "type": "pdf"
                })
    except Exception as e:
        logger.error(f"Error getting cloud sources: {e}")
        
    return cloud_sources

def get_cloud_sources(s3_client):
    """Get sources from S3 with error handling"""
    cloud_sources = []
    try:
        # List objects in the source bucket
        bucket = os.getenv('S3_BUCKET_NAME')
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='sources/')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):
                cloud_sources.append({
                    "path": f"s3://{bucket}/{key}",
                    "type": "csv",
                    "columns": ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                               'vin', 'state', 'condition', 'odometer', 'color', 'interior', 
                               'seller', 'mmr', 'sellingprice', 'saledate']
                })
            elif key.endswith('.pdf'):
                cloud_sources.append({
                    "path": f"s3://{bucket}/{key}",
                    "type": "pdf"
                })
    except Exception as e:
        logger.error(f"Error getting cloud sources: {e}")
        
    return cloud_sources

# Page configuration
st.set_page_config(
    page_title="🚗 AI Chat Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Main title with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🚗 AI Chat Assistant</h1>
    """, unsafe_allow_html=True)

# Initialize QA system if not done
if st.session_state.qa_system is None:
    with st.spinner("Initializing AI system..."):
        try:
            initialize_qa_system()
            st.success("System initialized successfully! Ready to chat about cars!")
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")

# Create layout containers
st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
chat_container = st.container()
input_container = st.container()

# Chat interface in the chat container
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Add visualization for assistant messages
            if msg["role"] == "assistant":
                viz_data = create_visualization_data(
                    st.session_state.qa_system,
                    st.session_state.last_query,
                    msg["content"]
                )
                
                if viz_data:
                    # Create container for visualization
                    viz_container = st.container()
                    with viz_container:
                        # Use ChatVisualizer component
                        st.markdown(
                            """
                            <div id="chat-visualization" 
                                 data-query="{query}" 
                                 data-market-data='{market_data}'></div>
                            """.format(
                                query=st.session_state.last_query,
                                market_data=json.dumps(viz_data)
                            ),
                            unsafe_allow_html=True
                        )

# Input interface in the input container
with input_container:
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    
    # Chat input with placeholder text
    if user_input := st.chat_input("Ask me anything about cars..."):
        # Store the query
        st.session_state.last_query = user_input
        
        # Log user input
        logger.info(f"Received user input: {user_input}")
        
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display AI response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                response = st.session_state.chain.invoke(user_input)
                placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                logger.info("Successfully generated response")
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                placeholder.error(error_msg)
                logger.error(error_msg)

# Add bottom spacing
st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
