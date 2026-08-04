"""
features.py
-----------
Feature engineering module for Zomato project.
Creates 11+ derived features for ML modeling and analysis.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline.
    Creates derived features for delivery time prediction and churn analysis.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize feature engineer.
        
        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.features_created = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from order date/time.
        
        Features:
        - peak_hour: 1 if lunch (12-2pm) or dinner (7-10pm)
        - weekend_order: 1 if Saturday or Sunday
        - day_of_week: 0-6 (Monday-Sunday)
        - hour_of_day: 0-23
        - month: 1-12
        - is_weekend: Boolean weekend flag
        
        Args:
            df (pd.DataFrame): DataFrame with orderdate and ordertime
        
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        df = df.copy()
        
        # Ensure datetime columns
        if 'orderdate' in df.columns:
            df['orderdate'] = pd.to_datetime(df['orderdate'], errors='coerce')
        
        if 'ordertime' in df.columns:
            # Extract hour from ordertime
            if df['ordertime'].dtype == 'object':
                df['hour_of_day'] = pd.to_datetime(df['ordertime'], format='%H:%M:%S', errors='coerce').dt.hour
            else:
                df['hour_of_day'] = pd.to_datetime(df['ordertime'], errors='coerce').dt.hour
        
        # Peak hours (12-14 = lunch, 19-22 = dinner)
        df['peak_hour'] = ((df['hour_of_day'].isin([12, 13, 14])) | 
                          (df['hour_of_day'].isin([19, 20, 21, 22]))).astype(int)
        
        # Weekend flag
        df['day_of_week'] = df['orderdate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['weekend_order'] = df['is_weekend']  # Alias for clarity
        
        # Month
        df['month'] = df['orderdate'].dt.month
        
        # Day of month
        df['day_of_month'] = df['orderdate'].dt.day
        
        # Quarter
        df['quarter'] = df['orderdate'].dt.quarter
        
        self.features_created.extend([
            'peak_hour', 'weekend_order', 'day_of_week', 'hour_of_day',
            'month', 'day_of_month', 'quarter', 'is_weekend'
        ])
        
        self.logger.info("✓ Temporal features created")
        return df
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level features.
        
        Features:
        - customer_tenure: Days since registration
        - average_basket_value: Mean order amount
        - customer_lifetime_value: Cumulative spending
        - order_frequency: Orders per 30 days
        
        Args:
            df (pd.DataFrame): DataFrame with customer order data
        
        Returns:
            pd.DataFrame: DataFrame with customer features
        """
        df = df.copy()
        
        if 'customerid' not in df.columns or 'orderdate' not in df.columns:
            self.logger.warning("Missing customerid or orderdate for customer features")
            return df
        
        # Customer tenure (days since registration)
        if 'registrationdate' in df.columns:
            df['registrationdate'] = pd.to_datetime(df['registrationdate'], errors='coerce')
            df['customer_tenure'] = (df['orderdate'] - df['registrationdate']).dt.days
            df['customer_tenure'] = df['customer_tenure'].clip(lower=0)
        
        # Average basket value per customer
        if 'finalamount' in df.columns:
            customer_avg = df.groupby('customerid')['finalamount'].transform('mean')
            df['average_basket_value'] = customer_avg
        
        # Customer lifetime value (cumulative spending)
        if 'finalamount' in df.columns:
            customer_clv = df.groupby('customerid')['finalamount'].transform('sum')
            df['customer_lifetime_value'] = customer_clv
        
        # Order frequency (orders per 30 days per customer)
        customer_order_count = df.groupby('customerid').cumcount() + 1
        df['customer_order_count'] = customer_order_count
        
        self.features_created.extend([
            'customer_tenure', 'average_basket_value', 
            'customer_lifetime_value', 'customer_order_count'
        ])
        
        self.logger.info("✓ Customer features created")
        return df
    
    def create_restaurant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create restaurant-level features.
        
        Features:
        - restaurant_order_count: Total orders
        - restaurant_avg_rating: Average customer rating
        - restaurant_popularity: Rank within city
        
        Args:
            df (pd.DataFrame): DataFrame with restaurant data
        
        Returns:
            pd.DataFrame: DataFrame with restaurant features
        """
        df = df.copy()
        
        if 'restaurantid' not in df.columns:
            self.logger.warning("Missing restaurantid for restaurant features")
            return df
        
        # Restaurant order count
        rest_order_count = df.groupby('restaurantid').cumcount() + 1
        df['restaurant_order_count'] = rest_order_count
        
        # Restaurant average rating
        if 'rating' in df.columns:
            rest_avg_rating = df.groupby('restaurantid')['rating'].transform('mean')
            df['restaurant_avg_rating'] = rest_avg_rating
        
        # Restaurant popularity (rank by order count within city)
        if 'city' in df.columns:
            df['restaurant_popularity'] = (df.groupby('city')['restaurantid']
                                          .transform(lambda x: x.nunique()) - 
                                          df.groupby(['city', 'restaurantid'])['orderid']
                                          .transform('count').rank())
        
        self.features_created.extend([
            'restaurant_order_count', 'restaurant_avg_rating', 'restaurant_popularity'
        ])
        
        self.logger.info("✓ Restaurant features created")
        return df
    
    def create_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delivery-related features.
        
        Features:
        - delivery_efficiency: Partner's avg time / city avg time
        - partner_rating_impact: Partner rating influence
        
        Args:
            df (pd.DataFrame): DataFrame with delivery data
        
        Returns:
            pd.DataFrame: DataFrame with delivery features
        """
        df = df.copy()
        
        if 'deliverypartnerid' not in df.columns or 'deliverytimeminutes' not in df.columns:
            self.logger.warning("Missing delivery data for delivery features")
            return df
        
        # Partner average delivery time
        partner_avg_time = df.groupby('deliverypartnerid')['deliverytimeminutes'].transform('mean')
        df['partner_avg_delivery_time'] = partner_avg_time
        
        # Delivery efficiency (partner avg / city avg)
        if 'city' in df.columns:
            city_avg_time = df.groupby('city')['deliverytimeminutes'].transform('mean')
            df['delivery_efficiency'] = (partner_avg_time / city_avg_time).fillna(1.0)
        
        # Partner rating (if available)
        if 'rating' in df.columns:
            # Assuming there's a delivery partner rating column
            df['partner_rating_impact'] = df.get('rating', 3.5)
        
        self.features_created.extend([
            'partner_avg_delivery_time', 'delivery_efficiency', 'partner_rating_impact'
        ])
        
        self.logger.info("✓ Delivery features created")
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features.
        
        Features:
        - rain_impact: Score based on rainfall
        - weather_severity: Combined weather impact score
        
        Args:
            df (pd.DataFrame): DataFrame with weather data
        
        Returns:
            pd.DataFrame: DataFrame with weather features
        """
        df = df.copy()
        
        # Rain impact (0-10 scale)
        if 'rainfall' in df.columns:
            df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce').fillna(0)
            df['rain_impact'] = df['rainfall'].apply(
                lambda x: 0 if x < 2 else 3 if x < 5 else 6 if x < 10 else 10
            )
        else:
            df['rain_impact'] = 0
        
        # Weather severity (combined impact)
        if 'weathercondition' in df.columns:
            severity_map = {
                'Clear': 0,
                'Cloudy': 1,
                'Rainy': 5,
                'Heavy Rain': 8,
                'Foggy': 3,
                'Thunderstorm': 10,
                'Sunny': 0
            }
            df['weather_severity'] = df['weathercondition'].map(severity_map).fillna(2)
        else:
            df['weather_severity'] = df.get('rain_impact', 0)
        
        self.features_created.extend(['rain_impact', 'weather_severity'])
        
        self.logger.info("✓ Weather features created")
        return df
    
    def create_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create traffic-related features.
        
        Features:
        - traffic_score: Numeric score from traffic level
        - traffic_impact: Impact on delivery time
        
        Args:
            df (pd.DataFrame): DataFrame with traffic data
        
        Returns:
            pd.DataFrame: DataFrame with traffic features
        """
        df = df.copy()
        
        if 'trafficlevel' in df.columns:
            # Traffic score mapping
            traffic_map = {
                'Low': 1,
                'Medium': 5,
                'High': 8,
                'Very High': 10,
                'Moderate': 5,
                'Heavy': 8
            }
            df['traffic_score'] = df['trafficlevel'].map(traffic_map).fillna(5)
        else:
            df['traffic_score'] = 5
        
        # Traffic impact on delivery (correlation-based multiplier)
        df['traffic_impact'] = df['traffic_score'] * 0.8  # Empirical multiplier
        
        self.features_created.extend(['traffic_score', 'traffic_impact'])
        
        self.logger.info("✓ Traffic features created")
        return df
    
    def create_promotional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create promotion/discount features.
        
        Features:
        - discount_percentage: Percentage discount applied
        - promotion_used: Binary flag for coupon use
        - discount_impact: Impact of discount on order
        
        Args:
            df (pd.DataFrame): DataFrame with order/promotion data
        
        Returns:
            pd.DataFrame: DataFrame with promotional features
        """
        df = df.copy()
        
        # Discount percentage
        if 'discount' in df.columns and 'foodcost' in df.columns:
            df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)
            df['foodcost'] = pd.to_numeric(df['foodcost'], errors='coerce')
            df['discount_percentage'] = (df['discount'] / df['foodcost'] * 100).fillna(0)
            df['discount_percentage'] = df['discount_percentage'].clip(0, 100)
        else:
            df['discount_percentage'] = 0
        
        # Promotion used
        if 'couponcode' in df.columns:
            df['promotion_used'] = df['couponcode'].notna().astype(int)
        else:
            df['promotion_used'] = 0
        
        self.features_created.extend(['discount_percentage', 'promotion_used'])
        
        self.logger.info("✓ Promotional features created")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        self.logger.info("Starting comprehensive feature engineering...")
        
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_restaurant_features(df)
        df = self.create_delivery_features(df)
        df = self.create_weather_features(df)
        df = self.create_traffic_features(df)
        df = self.create_promotional_features(df)
        
        self.logger.info(f"✓ Total features created: {len(self.features_created)}")
        return df
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of all engineered features.
        
        Returns:
            List[str]: List of feature names
        """
        return self.features_created
    
    def get_feature_importance_baseline(self) -> pd.DataFrame:
        """
        Get baseline feature importance descriptions.
        
        Returns:
            pd.DataFrame: Feature descriptions and expected importance
        """
        feature_info = [
            {'Feature': 'peak_hour', 'Category': 'Temporal', 'Expected_Importance': 'High'},
            {'Feature': 'weekend_order', 'Category': 'Temporal', 'Expected_Importance': 'Medium'},
            {'Feature': 'traffic_score', 'Category': 'Traffic', 'Expected_Importance': 'High'},
            {'Feature': 'rain_impact', 'Category': 'Weather', 'Expected_Importance': 'High'},
            {'Feature': 'delivery_efficiency', 'Category': 'Delivery', 'Expected_Importance': 'High'},
            {'Feature': 'customer_tenure', 'Category': 'Customer', 'Expected_Importance': 'Medium'},
            {'Feature': 'average_basket_value', 'Category': 'Customer', 'Expected_Importance': 'Low'},
            {'Feature': 'promotion_used', 'Category': 'Promotion', 'Expected_Importance': 'Medium'},
            {'Feature': 'restaurant_popularity', 'Category': 'Restaurant', 'Expected_Importance': 'Medium'},
            {'Feature': 'customer_lifetime_value', 'Category': 'Customer', 'Expected_Importance': 'Medium'},
            {'Feature': 'hour_of_day', 'Category': 'Temporal', 'Expected_Importance': 'High'},
        ]
        return pd.DataFrame(feature_info)
