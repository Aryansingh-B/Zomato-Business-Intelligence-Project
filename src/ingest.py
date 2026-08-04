"""
ingest.py
---------
Data ingestion module for loading all 12 raw CSV datasets into memory
and the database. Handles initial data profiling and quality checks.

Author: Data Science Team
Date: 2024
"""

import os
import logging
from typing import Dict, Optional
import pandas as pd
from pathlib import Path
from src.utils import setup_logger, load_csv_safe, log_data_quality, standardize_column_names


class ZomatoDataIngestor:
    """
    Handles loading and initial profiling of all Zomato datasets.
    """
    
    # Expected datasets and their row counts (approximate)
    EXPECTED_DATASETS = {
        'customers': {'rows': 12000, 'key': 'CustomerID'},
        'restaurants': {'rows': 1200, 'key': 'RestaurantID'},
        'orders': {'rows': 20000, 'key': 'OrderID'},
        'order_items': {'rows': 45000, 'key': 'OrderItemID'},
        'menu': {'rows': 9000, 'key': 'FoodItemID'},
        'delivery_partners': {'rows': 2000, 'key': 'DeliveryPartnerID'},
        'customer_feedback': {'rows': 14000, 'key': 'FeedbackID'},
        'payments': {'rows': 20000, 'key': 'PaymentID'},
        'promotions': {'rows': 300, 'key': 'PromotionID'},
        'cities': {'rows': 25, 'key': 'CityID'},
        'weather': {'rows': 18000, 'key': 'WeatherID'},
        'traffic': {'rows': 18000, 'key': 'TrafficID'},
    }
    
    def __init__(self, raw_data_path: str = './data/raw/', 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the data ingestor.
        
        Args:
            raw_data_path (str): Path to raw CSV files
            logger (logging.Logger, optional): Logger instance
        """
        self.raw_data_path = Path(raw_data_path)
        self.logger = logger or setup_logger(__name__)
        self.datasets = {}
        self.quality_reports = {}
        
        # Verify raw data path exists
        if not self.raw_data_path.exists():
            self.logger.warning(f"Raw data path does not exist: {raw_data_path}")
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 12 CSV datasets from raw data folder.
        
        Returns:
            dict: Dictionary of loaded DataFrames
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("LOADING ALL DATASETS")
        self.logger.info("="*70)
        
        for dataset_name in self.EXPECTED_DATASETS.keys():
            filepath = self.raw_data_path / f"{dataset_name}.csv"
            
            if filepath.exists():
                try:
                    df = load_csv_safe(str(filepath), self.logger)
                    # Standardize column names
                    df = standardize_column_names(df)
                    self.datasets[dataset_name] = df
                except Exception as e:
                    self.logger.error(f"✗ Failed to load {dataset_name}: {str(e)}")
            else:
                self.logger.warning(f"⚠ File not found: {filepath}")
        
        self.logger.info(f"\n✓ Loaded {len(self.datasets)}/{len(self.EXPECTED_DATASETS)} datasets")
        return self.datasets
    
    def profile_all_datasets(self) -> Dict[str, dict]:
        """
        Profile all loaded datasets for data quality issues.
        
        Returns:
            dict: Quality reports for each dataset
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PROFILING ALL DATASETS")
        self.logger.info("="*70)
        
        for dataset_name, df in self.datasets.items():
            quality_report = log_data_quality(df, dataset_name, self.logger)
            self.quality_reports[dataset_name] = quality_report
        
        return self.quality_reports
    
    def check_dataset_integrity(self) -> Dict[str, dict]:
        """
        Check for known data quality issues in each dataset.
        
        Returns:
            dict: Integrity check results
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("DATA INTEGRITY CHECKS")
        self.logger.info("="*70)
        
        integrity_issues = {}
        
        # Check customers
        if 'customers' in self.datasets:
            df = self.datasets['customers']
            issues = {
                'missing_age': df['age'].isnull().sum(),
                'missing_email': df['email'].isnull().sum() if 'email' in df.columns else 0,
                'missing_phone': df['phone'].isnull().sum() if 'phone' in df.columns else 0,
                'duplicate_ids': df['customerid'].duplicated().sum(),
            }
            integrity_issues['customers'] = issues
            self.logger.info(f"Customers: {issues}")
        
        # Check orders
        if 'orders' in self.datasets:
            df = self.datasets['orders']
            issues = {
                'missing_delivery_time': df['deliverytimeminutes'].isnull().sum(),
                'negative_cost': (df['foodcost'] < 0).sum() if 'foodcost' in df.columns else 0,
                'duplicate_ids': df['orderid'].duplicated().sum(),
            }
            integrity_issues['orders'] = issues
            self.logger.info(f"Orders: {issues}")
        
        # Check restaurants
        if 'restaurants' in self.datasets:
            df = self.datasets['restaurants']
            issues = {
                'missing_rating': df['rating'].isnull().sum() if 'rating' in df.columns else 0,
                'duplicate_ids': df['restaurantid'].duplicated().sum(),
            }
            integrity_issues['restaurants'] = issues
            self.logger.info(f"Restaurants: {issues}")
        
        # Check payments
        if 'payments' in self.datasets:
            df = self.datasets['payments']
            issues = {
                'duplicate_ids': df['paymentid'].duplicated().sum(),
                'failed_payments': (df['paymentstatus'] == 'Failed').sum() if 'paymentstatus' in df.columns else 0,
            }
            integrity_issues['payments'] = issues
            self.logger.info(f"Payments: {issues}")
        
        self.logger.info("="*70 + "\n")
        return integrity_issues
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """
        Create a summary table of all loaded datasets.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        summary_data = []
        
        for dataset_name, df in self.datasets.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory (MB)': df.memory_usage(deep=True).sum() / 1024**2,
                'Missing %': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            })
        
        summary_df = pd.DataFrame(summary_data)
        self.logger.info("\n" + "="*70)
        self.logger.info("DATASET SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"\n{summary_df.to_string(index=False)}\n")
        
        return summary_df
    
    def get_column_info(self, dataset_name: str) -> pd.DataFrame:
        """
        Get detailed column information for a specific dataset.
        
        Args:
            dataset_name (str): Name of dataset
        
        Returns:
            pd.DataFrame: Column details
        """
        if dataset_name not in self.datasets:
            self.logger.warning(f"Dataset not found: {dataset_name}")
            return None
        
        df = self.datasets[dataset_name]
        column_info = []
        
        for col in df.columns:
            column_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': df[col].count(),
                'Null %': (df[col].isnull().sum() / len(df)) * 100,
                'Unique': df[col].nunique(),
            })
        
        return pd.DataFrame(column_info)
    
    def get_foreign_key_mapping(self) -> Dict[str, list]:
        """
        Identify potential foreign key relationships between datasets.
        
        Returns:
            dict: Mapping of relationships
        """
        relationships = {}
        
        # Define expected relationships based on schema
        expected_fks = {
            'orders': {
                'customerid': ('customers', 'customerid'),
                'restaurantid': ('restaurants', 'restaurantid'),
                'deliverypartnerid': ('delivery_partners', 'deliverypartnerid'),
            },
            'order_items': {
                'orderid': ('orders', 'orderid'),
                'fooditemid': ('menu', 'fooditemid'),
            },
            'menu': {
                'restaurantid': ('restaurants', 'restaurantid'),
            },
            'customer_feedback': {
                'orderid': ('orders', 'orderid'),
            },
            'payments': {
                'orderid': ('orders', 'orderid'),
            },
        }
        
        for table_name, fks in expected_fks.items():
            if table_name in self.datasets:
                table_df = self.datasets[table_name]
                for fk_col, (ref_table, ref_col) in fks.items():
                    if fk_col in table_df.columns and ref_table in self.datasets:
                        ref_df = self.datasets[ref_table]
                        if ref_col in ref_df.columns:
                            # Count orphaned records
                            orphaned = table_df[~table_df[fk_col].isin(ref_df[ref_col])][fk_col].count()
                            relationships[f"{table_name}.{fk_col}"] = {
                                'references': f"{ref_table}.{ref_col}",
                                'orphaned_count': orphaned,
                            }
        
        return relationships


# ============================================================================
# STANDALONE LOADING FUNCTION
# ============================================================================

def load_zomato_data(raw_data_path: str = './data/raw/') -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all Zomato datasets.
    
    Args:
        raw_data_path (str): Path to raw CSV files
    
    Returns:
        dict: Dictionary of loaded DataFrames
    """
    logger = setup_logger(__name__)
    ingestor = ZomatoDataIngestor(raw_data_path, logger)
    datasets = ingestor.load_all_datasets()
    ingestor.profile_all_datasets()
    ingestor.check_dataset_integrity()
    ingestor.get_dataset_summary()
    
    return datasets
