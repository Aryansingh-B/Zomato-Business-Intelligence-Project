"""
Zomato Business Intelligence & Delivery Time Prediction Platform
Source package containing data processing and ML modules.

Modules:
    - ingest: Data loading from CSV files
    - clean: Data cleaning and transformation
    - features: Feature engineering
    - model_delivery: Delivery time prediction models
    - model_churn: Customer churn classification models
    - utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from . import utils, ingest, clean, features

__all__ = [
    'utils',
    'ingest',
    'clean',
    'features',
]
