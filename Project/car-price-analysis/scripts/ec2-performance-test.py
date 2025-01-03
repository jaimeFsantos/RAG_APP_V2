"""
EC2 Free Tier Performance Test Suite

Tests application performance within EC2 free tier constraints:
- 1 vCPU
- 1GB RAM
- 8GB Storage

Usage:
    python ec2_performance_test.py
"""

import psutil
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import threading
import queue
import gc
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EC2ResourceMonitor:
    """Monitors resource usage within EC2 free tier limits"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_queue = queue.Queue()
        self.monitoring = False
        
        # EC2 t2.micro limits
        self.RAM_LIMIT_MB = 1024  # 1GB
        self.CPU_CORE_COUNT = 1
        
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring thread and save results"""
        self.monitoring = False
        self.monitor_thread.join()
        
        # Get all metrics from queue
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
            
        # Save and analyze results
        if metrics:
            df = pd.DataFrame(metrics)
            self._save_metrics(df)
            self._analyze_results(df)
        
    def _monitor_resources(self, interval: float):
        """Monitor system resources"""
        while self.monitoring:
            try:
                metric = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_used': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
                
                # Check for EC2 free tier violations
                if metric['memory_used'] > self.RAM_LIMIT_MB:
                    logger.warning(f"Memory usage exceeded EC2 free tier limit: {metric['memory_used']:.2f}MB")
                
                if metric['cpu_percent'] > 95:  # Allow some headroom
                    logger.warning(f"High CPU usage detected: {metric['cpu_percent']}%")
                    
                self.metrics_queue.put(metric)
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                
            time.sleep(interval)
            
    def _save_metrics(self, df: pd.DataFrame):
        """Save metrics to CSV"""
        try:
            output_path = self.output_dir / 'ec2_resource_usage.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Saved metrics to {output_path}")
            
            # Create visualization
            self._create_visualization(df)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            
    def _create_visualization(self, df: pd.DataFrame):
        """Create resource usage visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # CPU Usage
            sns.lineplot(data=df, x=df.index, y='cpu_percent', ax=ax1)
            ax1.axhline(y=95, color='r', linestyle='--', label='EC2 CPU Limit')
            ax1.set_title('CPU Usage')
            ax1.set_ylabel('CPU %')
            
            # Memory Usage
            sns.lineplot(data=df, x=df.index, y='memory_used', ax=ax2)
            ax2.axhline(y=self.RAM_LIMIT_MB, color='r', linestyle='--', label='EC2 RAM Limit')
            ax2.set_title('Memory Usage')
            ax2.set_ylabel('Memory (MB)')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ec2_resource_usage.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            
    def _analyze_results(self, df: pd.DataFrame):
        """Analyze resource usage and provide recommendations"""
        analysis = {
            'max_memory': df['memory_used'].max(),
            'avg_memory': df['memory_used'].mean(),
            'max_cpu': df['cpu_percent'].max(),
            'avg_cpu': df['cpu_percent'].mean(),
            'memory_limit_exceeded': df['memory_used'].max() > self.RAM_LIMIT_MB,
            'high_cpu_percentage': (df['cpu_percent'] > 95).mean() * 100
        }
        
        # Generate recommendations
        recommendations = []
        
        if analysis['memory_limit_exceeded']:
            recommendations.append(
                f"WARNING: Peak memory usage ({analysis['max_memory']:.2f}MB) exceeds EC2 free tier limit"
            )
            
        if analysis['high_cpu_percentage'] > 10:
            recommendations.append(
                f"WARNING: High CPU usage detected for {analysis['high_cpu_percentage']:.1f}% of the time"
            )
            
        # Save analysis
        with open(self.output_dir / 'ec2_analysis.txt', 'w') as f:
            f.write("Resource Usage Analysis\n")
            f.write("======================\n\n")
            
            f.write(f"Memory Usage:\n")
            f.write(f"- Peak: {analysis['max_memory']:.2f}MB\n")
            f.write(f"- Average: {analysis['avg_memory']:.2f}MB\n")
            f.write(f"- EC2 Limit: {self.RAM_LIMIT_MB}MB\n\n")
            
            f.write(f"CPU Usage:\n")
            f.write(f"- Peak: {analysis['max_cpu']:.1f}%\n")
            f.write(f"- Average: {analysis['avg_cpu']:.1f}%\n\n")
            
            if recommendations:
                f.write("Recommendations:\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                    
        logger.info(f"Analysis saved to {self.output_dir / 'ec2_analysis.txt'}")

def test_qa_system():
    """Test QA system performance"""
    from AI_Chat_Analyst_Script import EnhancedQASystem
    
    monitor = EC2ResourceMonitor()
    qa_system = EnhancedQASystem(chunk_size=500, chunk_overlap=25)  # Reduced chunk size for EC2
    
    test_queries = [
        "What factors affect car prices?",
        "Compare Toyota and Honda resale values",
        "Analyze market trends for SUVs",
        "What are the most popular car colors?",
        "How does mileage affect price?"
    ]
    
    monitor.start_monitoring()
    
    try:
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            response = qa_system.ask(query)
            
            # Force garbage collection between queries
            gc.collect()
            
            # Small delay to prevent resource spikes
            time.sleep(2)
            
    except Exception as e:
        logger.error(f"Error testing QA system: {e}")
        
    finally:
        monitor.stop_monitoring()

def test_price_predictor():
    """Test price predictor performance"""
    from Pricing_Func import CarPricePredictor
    
    monitor = EC2ResourceMonitor()
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000  # Reduced sample size for testing
    
    test_data = pd.DataFrame({
        'year': np.random.randint(2010, 2024, n_samples),
        'condition': np.random.uniform(1, 50, n_samples),
        'odometer': np.random.randint(0, 150000, n_samples),
        'sellingprice': np.random.uniform(15000, 50000, n_samples),
        'state': np.random.choice(['CA', 'NY', 'TX'], n_samples),
        'body': np.random.choice(['Sedan', 'SUV', 'Truck'], n_samples),
        'transmission': np.random.choice(['Auto', 'Manual'], n_samples),
        'color': np.random.choice(['Black', 'White', 'Silver'], n_samples),
        'interior': np.random.choice(['Black', 'Tan', 'Gray'], n_samples)
    })
    
    predictor = CarPricePredictor(
        models=['rf'],  # Use only Random Forest for testing
        fast_mode=True
    )
    
    monitor.start_monitoring()
    
    try:
        # Test model training
        logger.info("Testing model training...")
        processed_data = predictor.prepare_data(test_data)
        features = predictor.engineer_features(processed_data)
        
        X = features.drop(['sellingprice'], axis=1)
        y = features['sellingprice']
        
        predictor.fit(X, y)
        
        # Test predictions
        logger.info("Testing predictions...")
        test_input = {
            'year': 2020,
            'condition': 25,
            'odometer': 50000,
            'state': 'CA',
            'body': 'Sedan',
            'transmission': 'Auto',
            'color': 'Black',
            'interior': 'Black'
        }
        
        for _ in range(5):  # Test multiple predictions
            prediction = predictor.create_what_if_prediction(test_input)
            gc.collect()
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error testing price predictor: {e}")
        
    finally:
        monitor.stop_monitoring()

def main():
    """Run all performance tests"""
    logger.info("Starting EC2 performance tests...")
    
    # Create test directory
    test_dir = Path("test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Run tests
    logger.info("Testing QA system...")
    test_qa_system()
    
    logger.info("Testing price predictor...")
    test_price_predictor()
    
    logger.info("Testing complete. Check test_results directory for analysis.")

if __name__ == "__main__":
    main()