import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import boto3
from io import BytesIO

logger = logging.getLogger(__name__)

class VisualizationDashboard:
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from either local storage or S3"""
        try:
            if self.s3_bucket != 'local':
                # Load from S3
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket, 
                    Key=file_path
                )
                return pd.read_csv(BytesIO(response['Body'].read()))
            else:
                # Load from local storage
                return pd.read_csv(file_path)
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def render_dashboard(self, file_path: str):
        """Main dashboard rendering function"""
        st.header("📊 Car Market Analysis Dashboard")
        
        df = self._load_data(file_path)
        if df.empty:
            st.error("No data available")
            return
    def __init__(self, s3_bucket: str):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.cache = {}

    def _load_data_from_s3(self, key: str) -> pd.DataFrame:
        """Load data from S3 with caching"""
        if key in self.cache:
            return self.cache[key]
        
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            df = pd.read_csv(BytesIO(response['Body'].read()))
            self.cache[key] = df
            return df
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
            return pd.DataFrame()

    def render_dashboard(self, data_key: str):
        """Main dashboard rendering function"""
        st.header("📊 Car Market Analysis Dashboard")
        
        if df.empty:
            df = self._load_data_from_s3(data_key)
            st.error("No data available")
            return

        # Create tabs for different visualizations
        tabs = st.tabs([
            "Price Trends", 
            "Market Distribution", 
            "Feature Analysis",
            "Geographic Analysis"
        ])

        with tabs[0]:
            self._render_price_trends(df)

        with tabs[1]:
            self._render_market_distribution(df)

        with tabs[2]:
            self._render_feature_analysis(df)

        with tabs[3]:
            self._render_geographic_analysis(df)

    def _render_price_trends(self, df: pd.DataFrame):
        """Render price trend visualizations"""
        st.subheader("Price Trends Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series of average prices
            df['date'] = pd.to_datetime(df['saledate'])
            monthly_avg = df.groupby(df['date'].dt.strftime('%Y-%m'))['sellingprice'].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg.values,
                mode='lines+markers',
                name='Average Price'
            ))
            fig.update_layout(
                title='Average Price Trends',
                xaxis_title='Month',
                yaxis_title='Price ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Price distribution by year
            fig = px.box(df, 
                x='year', 
                y='sellingprice',
                title='Price Distribution by Year'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _render_market_distribution(self, df: pd.DataFrame):
        """Render market distribution visualizations"""
        st.subheader("Market Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top manufacturers
            make_counts = df['make'].value_counts().head(10)
            fig = go.Figure(go.Bar(
                x=make_counts.values,
                y=make_counts.index,
                orientation='h'
            ))
            fig.update_layout(
                title='Top 10 Manufacturers',
                xaxis_title='Count',
                yaxis_title='Manufacturer',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Body style distribution
            body_counts = df['body'].value_counts()
            fig = px.pie(
                values=body_counts.values,
                names=body_counts.index,
                title='Vehicle Body Types Distribution'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _render_feature_analysis(self, df: pd.DataFrame):
        """Render feature analysis visualizations"""
        st.subheader("Feature Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mileage vs Price scatter
            fig = px.scatter(
                df,
                x='odometer',
                y='sellingprice',
                color='year',
                title='Price vs Mileage by Year',
                labels={'odometer': 'Mileage', 'sellingprice': 'Price ($)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Average price by transmission
            avg_price_trans = df.groupby('transmission')['sellingprice'].mean()
            fig = go.Figure(go.Bar(
                x=avg_price_trans.index,
                y=avg_price_trans.values
            ))
            fig.update_layout(
                title='Average Price by Transmission Type',
                xaxis_title='Transmission',
                yaxis_title='Average Price ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_geographic_analysis(self, df: pd.DataFrame):
        """Render geographic analysis visualizations"""
        st.subheader("Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average price by state
            state_prices = df.groupby('state')['sellingprice'].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(
                x=state_prices.index,
                y=state_prices.values
            ))
            fig.update_layout(
                title='Average Price by State',
                xaxis_title='State',
                yaxis_title='Average Price ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Popular models by region
            region_models = df.groupby(['state', 'model']).size().reset_index(name='count')
            top_models = region_models.sort_values('count', ascending=False).head(10)
            
            fig = px.treemap(
                top_models,
                path=['state', 'model'],
                values='count',
                title='Popular Models by State'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def create_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary metrics for dashboard"""
        return {
            'total_listings': len(df),
            'avg_price': df['sellingprice'].mean(),
            'median_price': df['sellingprice'].median(),
            'price_range': (df['sellingprice'].min(), df['sellingprice'].max()),
            'popular_makes': df['make'].value_counts().head(5).to_dict(),
            'popular_models': df['model'].value_counts().head(5).to_dict()
        }

    def export_visualization(self, fig, filename: str):
        """Export visualization to S3"""
        try:
            img_bytes = fig.to_image(format="png")
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"visualizations/{filename}",
                Body=img_bytes
            )
            return f"s3://{self.s3_bucket}/visualizations/{filename}"
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return None