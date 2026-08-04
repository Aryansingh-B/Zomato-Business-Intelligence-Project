"""
utils.py
--------
Utility functions for the Zomato BI project including database connections,
logging, and common helper functions.

Author: Data Science Team
Date: 2024
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
import psycopg2
from psycopg2.extras import RealDictCursor


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name (str): Logger name (typically __name__)
        log_file (str, optional): Path to log file. If None, logs to console only.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseConnection:
    """
    Manages database connections for both raw SQL operations and SQLAlchemy ORM.
    Supports PostgreSQL and MySQL.
    """
    
    def __init__(self, db_type: str = "postgresql", 
                 host: str = "localhost", 
                 port: int = 5432,
                 username: str = "postgres", 
                 password: str = "password",
                 database: str = "zomato_db",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize database connection parameters.
        
        Args:
            db_type (str): 'postgresql' or 'mysql'
            host (str): Database host
            port (int): Database port
            username (str): Database username
            password (str): Database password
            database (str): Database name
            logger (logging.Logger, optional): Logger instance
        """
        self.db_type = db_type
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.logger = logger or setup_logger(__name__)
        self.engine = None
        self.connection = None
    
    def connect(self) -> Engine:
        """
        Establish SQLAlchemy engine connection.
        
        Returns:
            sqlalchemy.engine.Engine: Database engine
        """
        try:
            if self.db_type == "postgresql":
                connection_string = (
                    f"postgresql+psycopg2://{self.username}:{self.password}"
                    f"@{self.host}:{self.port}/{self.database}"
                )
            elif self.db_type == "mysql":
                connection_string = (
                    f"mysql+pymysql://{self.username}:{self.password}"
                    f"@{self.host}:{self.port}/{self.database}"
                )
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self.engine = create_engine(connection_string, echo=False)
            self.logger.info(f"✓ Connected to {self.db_type} database: {self.database}")
            return self.engine
        
        except Exception as e:
            self.logger.error(f"✗ Database connection failed: {str(e)}")
            raise
    
    def execute_query(self, query: str, fetch: str = "all") -> list:
        """
        Execute a raw SQL query.
        
        Args:
            query (str): SQL query string
            fetch (str): 'all', 'one', or 'none'
        
        Returns:
            list: Query results
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query)
                if fetch == "all":
                    return result.fetchall()
                elif fetch == "one":
                    return result.fetchone()
                else:
                    return None
        except Exception as e:
            self.logger.error(f"✗ Query execution failed: {str(e)}")
            raise
    
    def get_table_list(self) -> list:
        """Get list of all tables in the database."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        self.logger.info(f"✓ Found {len(tables)} tables in database")
        return tables
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("✓ Database connection closed")


# ============================================================================
# DATA QUALITY FUNCTIONS
# ============================================================================

def log_data_quality(df: pd.DataFrame, table_name: str, 
                    logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Profile a DataFrame and return quality metrics.
    
    Args:
        df (pd.DataFrame): DataFrame to profile
        table_name (str): Name of the table for logging
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        dict: Quality metrics dictionary
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    quality_report = {
        'table_name': table_name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA QUALITY REPORT: {table_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Rows: {quality_report['total_rows']:,}")
    logger.info(f"Columns: {quality_report['total_columns']}")
    logger.info(f"Duplicates: {quality_report['duplicate_rows']:,}")
    logger.info(f"Memory: {quality_report['memory_usage_mb']:.2f} MB")
    logger.info(f"Missing Values:\n{pd.Series(quality_report['missing_values'])}")
    logger.info(f"{'='*60}\n")
    
    return quality_report


def detect_data_type_mismatches(df: pd.DataFrame, 
                                expected_dtypes: Dict[str, str],
                                logger: Optional[logging.Logger] = None) -> Dict[str, list]:
    """
    Detect columns with unexpected data types.
    
    Args:
        df (pd.DataFrame): DataFrame to check
        expected_dtypes (dict): Mapping of column_name to expected dtype
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        dict: Mismatched columns and their current types
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    mismatches = {}
    for col, expected_type in expected_dtypes.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                mismatches[col] = {
                    'expected': expected_type,
                    'actual': actual_type
                }
    
    if mismatches:
        logger.warning(f"Type mismatches detected: {mismatches}")
    
    return mismatches


# ============================================================================
# FILE I/O HELPERS
# ============================================================================

def load_csv_safe(filepath: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Safely load a CSV file with error handling.
    
    Args:
        filepath (str): Path to CSV file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        pd.DataFrame: Loaded data
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"✓ Loaded {filepath} ({len(df):,} rows)")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to load {filepath}: {str(e)}")
        raise


def save_csv_safe(df: pd.DataFrame, filepath: str, 
                 logger: Optional[logging.Logger] = None) -> None:
    """
    Safely save a DataFrame to CSV with error handling.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Destination filepath
        logger (logging.Logger, optional): Logger instance
    """
    if logger is None:
        logger = setup_logger(__name__)
    
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✓ Saved {filepath} ({len(df):,} rows)")
    except Exception as e:
        logger.error(f"✗ Failed to save {filepath}: {str(e)}")
        raise


# ============================================================================
# DATA TRANSFORMATION HELPERS
# ============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase with underscores.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    df.columns = (df.columns
                  .str.lower()
                  .str.strip()
                  .str.replace(' ', '_')
                  .str.replace('-', '_'))
    return df


def remove_leading_trailing_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading/trailing whitespace from all string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with cleaned strings
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df
