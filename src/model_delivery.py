"""
model_delivery.py
-----------------
Delivery time prediction models using regression algorithms.
Target: DeliveryTimeMinutes

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DeliveryTimePredictor:
    """
    Machine learning pipeline for delivery time prediction.
    Trains and compares multiple regression algorithms.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize delivery time predictor.
        
        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.scaler = StandardScaler()
        self.results = []
        self.best_model = None
        self.best_model_name = None
    
    def prepare_data(self, df: pd.DataFrame, 
                    target: str = 'deliverytimeminutes',
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for modeling.
        
        Args:
            df (pd.DataFrame): Input data with features and target
            target (str): Target column name
            test_size (float): Test set proportion
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Identify feature columns (exclude IDs, dates, and target)
        exclude_cols = ['orderid', 'customerid', 'restaurantid', 'deliverypartnerid',
                       'orderdate', 'ordertime', target, 'feedbackid', 'paymentid']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.dropna()
        X = X.loc[y.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"✓ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: pd.Series) -> LinearRegression:
        """
        Train Linear Regression baseline model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            LinearRegression: Trained model
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        self.logger.info("✓ Linear Regression trained")
        return model
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: pd.Series) -> DecisionTreeRegressor:
        """
        Train Decision Tree Regressor.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            DecisionTreeRegressor: Trained model
        """
        model = DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Decision Tree'] = model
        self.logger.info("✓ Decision Tree Regressor trained")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest Regressor.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            RandomForestRegressor: Trained model
        """
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        self.logger.info("✓ Random Forest Regressor trained")
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: pd.Series) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting Regressor.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            GradientBoostingRegressor: Trained model
        """
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        self.logger.info("✓ Gradient Boosting Regressor trained")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: pd.Series) -> Dict:
        """
        Train all regression models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            Dict: Dictionary of trained models
        """
        self.train_linear_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        self.logger.info(f"✓ All {len(self.models)} models trained")
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all models on test set.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
        
        Returns:
            pd.DataFrame: Model performance metrics
        """
        results = []
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'Test_Set_Size': len(y_test)
            })
            
            self.logger.info(f"{model_name:25} | R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
        self.results = pd.DataFrame(results)
        
        # Select best model (highest R²)
        best_idx = self.results['R2'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        self.logger.info(f"\n✓ Best Model: {self.best_model_name} (R² = {self.results.loc[best_idx, 'R2']:.4f})")
        
        return self.results
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            feature_names (List[str]): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if self.best_model is None:
            self.logger.warning("No model trained yet")
            return pd.DataFrame()
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            self.logger.info(f"{self.best_model_name} does not have feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> None:
        """
        Save best model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if self.best_model is None:
            self.logger.warning("No trained model to save")
            return
        
        joblib.dump(self.best_model, filepath)
        self.logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        self.best_model = joblib.load(filepath)
        self.logger.info(f"✓ Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with best model.
        
        Args:
            X (np.ndarray): Features for prediction
        
        Returns:
            np.ndarray: Predicted delivery times
        """
        if self.best_model is None:
            self.logger.error("No trained model available")
            return None
        
        return self.best_model.predict(X)
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of modeling results.
        
        Returns:
            Dict: Model summary with best model info and metrics
        """
        return {
            'Best_Model': self.best_model_name,
            'Models_Trained': len(self.models),
            'Results': self.results.to_dict(),
            'Best_R2': self.results['R2'].max() if len(self.results) > 0 else None,
            'Best_RMSE': self.results[self.results['Model'] == self.best_model_name]['RMSE'].values[0] if self.best_model_name else None
        }
