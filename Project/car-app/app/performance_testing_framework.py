"""
Local Testing Framework for Car Analysis Suite
Features both local performance analysis and integration testing
"""

import psutil
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalPerformanceAnalyzer:
    """Analyzes system performance locally"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = []
        self.start_time = None
        
    def monitor_resources(self, duration: int = 60, interval: int = 1) -> pd.DataFrame:
        """
        Monitor system resources for a specified duration
        
        Args:
            duration: Monitoring duration in seconds
            interval: Sampling interval in seconds
        """
        metrics = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                metric = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_used': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                    'memory_percent': psutil.virtual_memory().percent
                }
                metrics.append(metric)
                time.sleep(interval)
                
            df = pd.DataFrame(metrics)
            self._save_metrics(df, 'resource_usage.csv')
            self._plot_resource_usage(df)
            return df
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return pd.DataFrame()

    def profile_function(self, func, *args, **kwargs) -> Dict[str, float]:
        """
        Profile a specific function's performance
        
        Args:
            func: Function to profile
            args: Function arguments
            kwargs: Function keyword arguments
        """
        start_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            end_memory = psutil.Process().memory_info().rss
            end_time = time.time()
            
            metrics = {
                'execution_time': end_time - start_time,
                'memory_change': (end_memory - start_memory) / 1024 / 1024,  # MB
                'peak_memory': psutil.Process().memory_info().peak_wset / 1024 / 1024  # MB
            }
            
            self._save_metrics(pd.DataFrame([metrics]), f'profile_{func.__name__}.csv')
            return metrics
            
        except Exception as e:
            logger.error(f"Error profiling function {func.__name__}: {e}")
            return {}

    def _save_metrics(self, df: pd.DataFrame, filename: str):
        """Save metrics to CSV"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics to {output_path}")

    def _plot_resource_usage(self, df: pd.DataFrame):
        """Create resource usage plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # CPU Usage
        sns.lineplot(data=df, x=df.index, y='cpu_percent', ax=ax1)
        ax1.set_title('CPU Usage')
        ax1.set_ylabel('CPU %')
        
        # Memory Usage
        sns.lineplot(data=df, x=df.index, y='memory_used', ax=ax2)
        ax2.set_title('Memory Usage')
        ax2.set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage.png')
        plt.close()

class TestDataGenerator:
    """Generates test data matching CarPricePredictor requirements"""
    
    @staticmethod
    def create_sample_data(num_records: int = 1000) -> pd.DataFrame:
        """
        Create sample car data that matches predictor requirements
        
        Args:
            num_records: Number of test records to generate
        """
        # Define realistic value ranges for each feature
        makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']
        models = ['Camry', 'Civic', 'Focus', '3 Series', 'C-Class']
        trims = ['Base', 'Sport', 'Limited', 'Premium', 'Standard']
        bodies = ['Sedan', 'SUV', 'Coupe', 'Truck', 'Van']
        transmissions = ['Automatic', 'Manual', 'CVT']
        states = ['CA', 'NY', 'TX', 'FL', 'WA']
        colors = ['Black', 'White', 'Silver', 'Blue', 'Red']
        interiors = ['Black', 'Beige', 'Gray', 'Brown', 'White']
        sellers = ['Dealer', 'Private', 'Fleet']
        
        # Generate data matching required schema
        data = {
            'make': np.random.choice(makes, num_records),
            'model': np.random.choice(models, num_records),
            'trim': np.random.choice(trims, num_records),
            'body': np.random.choice(bodies, num_records),
            'transmission': np.random.choice(transmissions, num_records),
            'state': np.random.choice(states, num_records),
            'condition': np.random.uniform(1, 50, num_records),  # 1-50 scale
            'odometer': np.random.randint(0, 150000, num_records),
            'color': np.random.choice(colors, num_records),
            'interior': np.random.choice(interiors, num_records),
            'seller': np.random.choice(sellers, num_records),
            'sellingprice': np.random.uniform(15000, 50000, num_records)
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic year values
        df['year'] = np.random.randint(2015, 2024, num_records)
        
        # Ensure no missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(df[col].mean())
        
        return df

def test_price_predictor():
    """Test the price prediction functionality"""
    from Pricing_Func import CarPricePredictor
    
    # Generate test data with correct schema
    logger.info("Generating test data...")
    df = TestDataGenerator.create_sample_data(num_records=1000)
    
    # Initialize analyzer
    analyzer = LocalPerformanceAnalyzer()
    logger.info("Initializing predictor...")
    
    # Initialize predictor with minimal models for testing
    predictor = CarPricePredictor(
        models=['rf'],  # Use only Random Forest for testing
        fast_mode=True,
        max_samples=1000
    )
    
    def train_model():
        logger.info("Preparing data...")
        df_processed = predictor.prepare_data(df)
        logger.info(f"Processed data shape: {df_processed.shape}")
        
        logger.info("Engineering features...")
        df_engineered = predictor.engineer_features(df_processed)
        logger.info(f"Engineered features shape: {df_engineered.shape}")
        
        logger.info("Splitting features...")
        X = df_engineered.drop(['sellingprice'], axis=1)
        y = df_engineered['sellingprice']
        
        logger.info("Training model...")
        predictor.fit(X, y)
    
    metrics = analyzer.profile_function(train_model)
    logger.info(f"Training metrics: {metrics}")
    
    # Verify model trained successfully
    assert predictor.is_trained, "Model failed to train"
    
    # Test prediction
    if predictor.is_trained:
        logger.info("Testing prediction...")
        test_input = {
            'year': 2020,
            'condition': 25,
            'odometer': 50000,
            'state': 'CA',
            'body': 'Sedan',
            'transmission': 'Automatic',
            'color': 'Black',
            'interior': 'Black'
        }
        prediction = predictor.create_what_if_prediction(test_input)
        logger.info(f"Test prediction result: {prediction}")

    return predictor

def test_qa_system():
    """Test the QA system functionality"""
    from AI_Chat_Analyst_Script import QASystem
    
    analyzer = LocalPerformanceAnalyzer()
    qa_system = QASystem()
    
    def test_query():
        return qa_system.ask("What factors affect car prices?")
    
    metrics = analyzer.profile_function(test_query)
    logger.info(f"QA system metrics: {metrics}")

def main():
    """Run all local tests"""
    logger.info("Starting local testing...")
    
    # Create test directory
    test_dir = Path("test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Run performance analysis
    analyzer = LocalPerformanceAnalyzer()
    logger.info("Monitoring system resources...")
    resource_metrics = analyzer.monitor_resources(duration=60)
    
    # Run integration tests
    logger.info("Running integration tests...")
    test_price_predictor()
    test_qa_system()
    
    logger.info("Testing complete. Check test_results directory for outputs.")

if __name__ == "__main__":
    main()