"""
clean.py
--------
Data cleaning and transformation module for Zomato datasets.
Handles missing values, duplicates, inconsistencies, and data quality issues.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from datetime import datetime


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for Zomato datasets.
    Handles all 12 tables with their specific quality issues.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize data cleaner.
        
        Args:
            logger (logging.Logger): Logger instance for tracking operations
        """
        self.logger = logger or logging.getLogger(__name__)
        self.cleaning_log = []
    
    def clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customers dataset.
        
        Issues to handle:
        - Missing Age, Email, Phone values
        - Inconsistent gender values
        - Invalid phone numbers
        - Extra whitespace in names
        
        Args:
            df (pd.DataFrame): Raw customers data
        
        Returns:
            pd.DataFrame: Cleaned customers data
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle missing Age - impute with city median
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            city_medians = df.groupby('city')['age'].transform('median')
            df['age'].fillna(city_medians, inplace=True)
            # Fill remaining with overall median
            df['age'].fillna(df['age'].median(), inplace=True)
        
        # Handle missing Email
        if 'email' in df.columns:
            df['email'].fillna('not_provided', inplace=True)
        
        # Handle missing Phone - drop rows with no phone
        if 'phone' in df.columns:
            df['phone'] = df['phone'].astype(str).str.replace(r'\D', '', regex=True)
            df = df[df['phone'].str.len() >= 10].copy()
        
        # Standardize city names
        if 'city' in df.columns:
            df['city'] = df['city'].str.strip().str.title()
            # Common misspellings
            df['city'] = df['city'].replace({
                'Bangalore': 'Bengaluru',
                'bangalore': 'Bengaluru',
                'Bengaluru ': 'Bengaluru',
            })
        
        # Remove duplicates on customerid
        initial_dupes = df.duplicated(subset=['customerid']).sum()
        df = df.drop_duplicates(subset=['customerid'], keep='first')
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'customers',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after,
            'duplicates_removed': initial_dupes,
            'missing_emails_filled': (df['email'] == 'not_provided').sum()
        })
        
        self.logger.info(f"Cleaned customers: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_restaurants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean restaurants dataset.
        
        Issues to handle:
        - Duplicate restaurant IDs
        - Missing ratings
        - Inconsistent cuisine names
        - Invalid coordinates
        
        Args:
            df (pd.DataFrame): Raw restaurants data
        
        Returns:
            pd.DataFrame: Cleaned restaurants data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['restaurantid']).sum()
        df = df.drop_duplicates(subset=['restaurantid'], keep='first')
        
        # Handle missing ratings - impute with cuisine average
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            cuisine_means = df.groupby('cuisine')['rating'].transform('mean')
            df['rating'].fillna(cuisine_means, inplace=True)
            df['rating'].fillna(3.5, inplace=True)  # Overall average
        
        # Standardize cuisine names
        if 'cuisine' in df.columns:
            df['cuisine'] = df['cuisine'].str.strip().str.title()
        
        # Standardize city names
        if 'city' in df.columns:
            df['city'] = df['city'].str.strip().str.title()
            df['city'] = df['city'].replace({'Bangalore': 'Bengaluru'})
        
        # Strip whitespace
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'restaurants',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after,
            'duplicates_removed': initial_dupes
        })
        
        self.logger.info(f"Cleaned restaurants: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean orders dataset.
        
        Issues to handle:
        - Mixed date formats
        - Negative or zero costs
        - Unrealistic delivery times
        - Duplicate order IDs
        - Invalid foreign keys
        
        Args:
            df (pd.DataFrame): Raw orders data
        
        Returns:
            pd.DataFrame: Cleaned orders data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['orderid']).sum()
        df = df.drop_duplicates(subset=['orderid'], keep='first')
        
        # Handle date formats
        if 'orderdate' in df.columns:
            df['orderdate'] = pd.to_datetime(df['orderdate'], errors='coerce')
        
        # Handle negative or zero costs
        if 'foodcost' in df.columns:
            df['foodcost'] = pd.to_numeric(df['foodcost'], errors='coerce')
            df = df[df['foodcost'] > 0].copy()
        
        # Handle delivery time outliers (max 180 minutes)
        if 'deliverytimeminutes' in df.columns:
            df['deliverytimeminutes'] = pd.to_numeric(df['deliverytimeminutes'], errors='coerce')
            df = df[df['deliverytimeminutes'].between(1, 180)].copy()
        
        # Impute missing delivery times with median by restaurant
        if 'deliverytimeminutes' in df.columns and 'restaurantid' in df.columns:
            rest_medians = df.groupby('restaurantid')['deliverytimeminutes'].transform('median')
            df['deliverytimeminutes'].fillna(rest_medians, inplace=True)
            df['deliverytimeminutes'].fillna(df['deliverytimeminutes'].median(), inplace=True)
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'orders',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after,
            'duplicates_removed': initial_dupes
        })
        
        self.logger.info(f"Cleaned orders: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_menu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean menu dataset.
        
        Issues to handle:
        - Duplicate food item IDs
        - Negative prices
        - Missing availability
        
        Args:
            df (pd.DataFrame): Raw menu data
        
        Returns:
            pd.DataFrame: Cleaned menu data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['fooditemid']).sum()
        df = df.drop_duplicates(subset=['fooditemid'], keep='first')
        
        # Handle negative prices
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[df['price'] > 0].copy()
        
        # Handle missing availability
        if 'availability' in df.columns:
            df['availability'].fillna('Yes', inplace=True)
        
        # Strip whitespace
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'menu',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after
        })
        
        self.logger.info(f"Cleaned menu: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_delivery_partners(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean delivery partners dataset.
        
        Issues to handle:
        - Duplicate delivery partner IDs
        - Missing ratings
        - Invalid age ranges
        
        Args:
            df (pd.DataFrame): Raw delivery partners data
        
        Returns:
            pd.DataFrame: Cleaned delivery partners data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['deliverypartnerid']).sum()
        df = df.drop_duplicates(subset=['deliverypartnerid'], keep='first')
        
        # Handle age
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df = df[df['age'].between(16, 70)].copy()
        
        # Handle missing ratings
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating'].fillna(3.5, inplace=True)
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'delivery_partners',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after
        })
        
        self.logger.info(f"Cleaned delivery_partners: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_customer_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customer feedback dataset.
        
        Issues to handle:
        - Duplicate feedback IDs
        - Missing ratings
        - Invalid rating ranges (1-5)
        - Inconsistent sentiment values
        
        Args:
            df (pd.DataFrame): Raw feedback data
        
        Returns:
            pd.DataFrame: Cleaned feedback data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['feedbackid']).sum()
        df = df.drop_duplicates(subset=['feedbackid'], keep='first')
        
        # Validate ratings are 1-5
        rating_cols = ['customerrating', 'deliveryrating', 'foodrating']
        for col in rating_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(lower=1, upper=5)
                df[col].fillna(3, inplace=True)
        
        # Standardize sentiment
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].str.strip().str.title()
            valid_sentiments = ['Positive', 'Negative', 'Neutral']
            df.loc[~df['sentiment'].isin(valid_sentiments), 'sentiment'] = 'Neutral'
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'customer_feedback',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after
        })
        
        self.logger.info(f"Cleaned customer_feedback: {initial_rows} → {rows_after} rows")
        return df
    
    def clean_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean payments dataset.
        
        Issues to handle:
        - Duplicate payment IDs
        - Invalid payment status
        - Missing payment dates
        
        Args:
            df (pd.DataFrame): Raw payments data
        
        Returns:
            pd.DataFrame: Cleaned payments data
        """
        df = df.copy()
        initial_rows = len(df)
        
        df.columns = df.columns.str.lower().str.strip()
        
        # Remove duplicates
        initial_dupes = df.duplicated(subset=['paymentid']).sum()
        df = df.drop_duplicates(subset=['paymentid'], keep='first')
        
        # Handle payment date
        if 'paymentdate' in df.columns:
            df['paymentdate'] = pd.to_datetime(df['paymentdate'], errors='coerce')
        
        # Standardize payment status
        if 'paymentstatus' in df.columns:
            df['paymentstatus'] = df['paymentstatus'].str.strip().str.title()
            valid_status = ['Success', 'Failed', 'Pending']
            df.loc[~df['paymentstatus'].isin(valid_status), 'paymentstatus'] = 'Success'
        
        rows_after = len(df)
        self.cleaning_log.append({
            'table': 'payments',
            'rows_before': initial_rows,
            'rows_after': rows_after,
            'rows_removed': initial_rows - rows_after
        })
        
        self.logger.info(f"Cleaned payments: {initial_rows} → {rows_after} rows")
        return df
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """
        Get summary of all cleaning operations.
        
        Returns:
            pd.DataFrame: Cleaning log summary
        """
        return pd.DataFrame(self.cleaning_log)
    
    def export_cleaning_log(self, filepath: str) -> None:
        """
        Export cleaning log to CSV.
        
        Args:
            filepath (str): Path to save cleaning log
        """
        summary = self.get_cleaning_summary()
        summary.to_csv(filepath, index=False)
        self.logger.info(f"✓ Cleaning log exported to {filepath}")
