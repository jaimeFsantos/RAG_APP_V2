"""
Enhanced Car Visualization Dashboard

This module provides comprehensive visualization capabilities for car data analysis,
optimized for both local development and AWS EC2 free tier deployment.

Key Features:
- Memory-efficient data loading and processing
- Incremental visualization rendering
- Cloud storage integration with S3
- Caching for improved performance
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import boto3
from io import BytesIO
import json
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In car_viz_dashboard.py

class DataLoader:
    def __init__(self, is_cloud: bool = True):
        self.is_cloud = is_cloud
        if is_cloud:
            self.s3_client = boto3.client('s3')
            self.bucket = os.getenv('S3_BUCKET_NAME')
        self.cache = {}

    @lru_cache(maxsize=32)
    def load_data(self, source: str) -> pd.DataFrame:
        """
        Load data efficiently with caching
        
        Args:
            source: Data source path/key
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if source in self.cache:
                return self.cache[source]
                
            if self.is_cloud:
                data = self._load_from_s3(source)
            else:
                data = pd.read_csv(source)  # Direct read for local files
                
            self.cache[source] = data
            logger.info(f"Successfully loaded data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return pd.DataFrame()
            
    def _load_from_s3(self, key: str) -> pd.DataFrame:
        """Load data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return pd.read_csv(BytesIO(response['Body'].read()))
        except Exception as e:
            logger.error(f"S3 loading error: {e}")
            return pd.DataFrame()
            
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for visualization"""
        try:
            # Handle missing values
            df = df.fillna({
                'make': 'Unknown',
                'model': 'Unknown',
                'trim': 'Unknown'
            })
            
            # Convert date columns
            if 'saledate' in df.columns:
                df['saledate'] = pd.to_datetime(df['saledate'])
                
            return df
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return df

# In car_viz_dashboard.py

class ChartBuilder:
    """Handles chart creation and configuration"""
    
    @staticmethod
    def create_market_distribution(df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create market distribution visualizations"""
        figures = {}
        
        # Make distribution
        make_counts = df['make'].value_counts().head(10)
        figures['makes'] = go.Figure(go.Bar(
            x=make_counts.index,
            y=make_counts.values,
            name='Count'
        ))
        figures['makes'].update_layout(
            title='Top 10 Manufacturers',
            xaxis_title='Manufacturer',
            yaxis_title='Number of Vehicles',
            height=400
        )
        
        # Body style distribution
        body_counts = df['body'].value_counts()
        figures['body_styles'] = go.Figure(go.Bar(
            x=body_counts.index,
            y=body_counts.values,
            name='Count'
        ))
        figures['body_styles'].update_layout(
            title='Vehicle Body Types',
            xaxis_title='Body Style',
            yaxis_title='Number of Vehicles',
            height=400
        )
        
        # Transmission distribution
        trans_counts = df['transmission'].value_counts()
        figures['transmission'] = go.Figure(go.Bar(
            x=trans_counts.index,
            y=trans_counts.values,
            name='Count'
        ))
        figures['transmission'].update_layout(
            title='Transmission Types',
            xaxis_title='Transmission',
            yaxis_title='Number of Vehicles',
            height=400
        )
        
        return figures


class CarVizDashboard:
    """Main dashboard class for car data visualization"""
    
    def __init__(self, is_cloud: bool = True):
        self.data_loader = DataLoader(is_cloud)
        self.chart_builder = ChartBuilder()
        
    def render_dashboard(self, data_source: str):
        """Render the complete dashboard"""
        try:
            # Load data
            df = self.data_loader.load_data(data_source)
            if df.empty:
                st.error("No data available for visualization")
                return
            
            # Create market analysis section
            st.subheader("Market Analysis")
            
            # Get market distribution charts
            market_figures = self.chart_builder.create_market_distribution(df)
            
            # Create tabs for different aspects
            tab1, tab2, tab3 = st.tabs(['Makes', 'Body Styles', 'Transmissions'])
            
            with tab1:
                st.plotly_chart(market_figures['makes'], use_container_width=True)
                
            with tab2:
                st.plotly_chart(market_figures['body_styles'], use_container_width=True)
                
            with tab3:
                st.plotly_chart(market_figures['transmission'], use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error rendering market analysis: {e}")
            st.error("Error creating market analysis visualizations")
            if os.getenv('DEBUG', 'false').lower() == 'true':
                st.write("Debug info:", str(e))
            
    def _render_summary_metrics(self, df: pd.DataFrame):
        """Render summary metrics section"""
        cols = st.columns(4)
        
        metrics = {
            "Total Listings": len(df),
            "Average Price": f"${df['sellingprice'].mean():,.2f}",
            "Median Price": f"${df['sellingprice'].median():,.2f}",
            "Price Range": f"${df['sellingprice'].min():,.0f} - ${df['sellingprice'].max():,.0f}"
        }
        
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, value)
            
    def _render_price_analysis(self, df: pd.DataFrame):
        """Render price analysis section"""
        st.subheader("Price Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = self.chart_builder.create_price_trend(df)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Add additional price analysis chart
            fig = px.box(df, x='year', y='sellingprice', title='Price Distribution by Year')
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_market_distribution(self, df: pd.DataFrame):
        """Render market distribution section"""
        st.subheader("Market Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = self.chart_builder.create_make_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Add body style distribution
            body_counts = df['body'].value_counts()
            fig = px.pie(
                values=body_counts.values,
                names=body_counts.index,
                title='Vehicle Body Types'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_feature_impact(self, df: pd.DataFrame):
        """Render feature impact section"""
        st.subheader("Feature Impact")
        
        # Add feature selector
        feature = st.selectbox(
            "Select Feature",
            options=['transmission', 'color', 'interior', 'body']
        )
        
        fig = self.chart_builder.create_price_by_feature(df, feature)
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_geographic_insights(self, df: pd.DataFrame):
        """Render geographic insights section"""
        st.subheader("Geographic Insights")
        
        fig = self.chart_builder.create_price_by_feature(df, 'state')
        st.plotly_chart(fig, use_container_width=True)
