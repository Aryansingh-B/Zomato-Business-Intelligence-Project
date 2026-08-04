"""
model_churn.py
--------------
Customer churn prediction models using classification algorithms.
Target: Binary churn flag (no orders in 60 days)

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)


class ChurnPredictor:
    """
    Machine learning pipeline for customer churn prediction.
    Trains and compares multiple classification algorithms.
    Handles class imbalance with class weights.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize churn predictor.
        
        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.scaler = StandardScaler()
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.class_weights = 'balanced'
    
    def create_churn_target(self, df: pd.DataFrame, 
                           days_threshold: int = 60) -> pd.Series:
        """
        Create binary churn target variable.
        Customer is churned if no orders in last N days.
        
        Args:
            df (pd.DataFrame): DataFrame with customer orders
            days_threshold (int): Days without orders to define churn
        
        Returns:
            pd.Series: Binary churn target (1=churned, 0=active)
        """
        # Calculate days since last order per customer
        if 'orderdate' in df.columns:
            df['orderdate'] = pd.to_datetime(df['orderdate'], errors='coerce')
            max_date = df['orderdate'].max()
            
            last_order = df.groupby('customerid')['orderdate'].max()
            days_since_order = (max_date - last_order).dt.days
            
            churn = (days_since_order > days_threshold).astype(int)
            return churn
        else:
            self.logger.warning("No orderdate column found")
            return pd.Series([0] * len(df))
    
    def prepare_data(self, df: pd.DataFrame, 
                    target: str = 'churn',
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for classification modeling.
        
        Args:
            df (pd.DataFrame): Input data with features and target
            target (str): Target column name
            test_size (float): Test set proportion
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Identify feature columns
        exclude_cols = ['orderid', 'customerid', 'restaurantid', 'deliverypartnerid',
                       'orderdate', 'ordertime', target, 'feedbackid', 'paymentid', 'churn']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.dropna()
        X = X.loc[y.index]
        
        # Check class distribution
        class_counts = y.value_counts()
        self.logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"✓ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
        """
        Train Logistic Regression baseline model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            LogisticRegression: Trained model
        """
        model = LogisticRegression(
            class_weight=self.class_weights,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        self.logger.info("✓ Logistic Regression trained")
        return model
    
    def train_decision_tree(self, X_train: np.ndarray, y_train: pd.Series) -> DecisionTreeClassifier:
        """
        Train Decision Tree Classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            DecisionTreeClassifier: Trained model
        """
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=self.class_weights,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Decision Tree'] = model
        self.logger.info("✓ Decision Tree Classifier trained")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest Classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            RandomForestClassifier: Trained model
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        self.logger.info("✓ Random Forest Classifier trained")
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: pd.Series) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting Classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            GradientBoostingClassifier: Trained model
        """
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        self.logger.info("✓ Gradient Boosting Classifier trained")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: pd.Series) -> Dict:
        """
        Train all classification models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
        
        Returns:
            Dict: Dictionary of trained models
        """
        self.train_logistic_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        self.logger.info(f"✓ All {len(self.models)} models trained")
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all models on test set with comprehensive metrics.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
        
        Returns:
            pd.DataFrame: Model performance metrics
        """
        results = []
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC_AUC': roc_auc,
                'Test_Set_Size': len(y_test)
            })
            
            self.logger.info(
                f"{model_name:25} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}"
            )
        
        self.results = pd.DataFrame(results)
        
        # Select best model (highest F1-score)
        best_idx = self.results['F1'].idxmax()
        self.best_model_name = self.results.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        self.logger.info(f"\n✓ Best Model: {self.best_model_name} (F1 = {self.results.loc[best_idx, 'F1']:.4f})")
        
        return self.results
    
    def get_confusion_matrix(self, X_test: np.ndarray, y_test: pd.Series) -> np.ndarray:
        """
        Get confusion matrix for best model.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
        
        Returns:
            np.ndarray: Confusion matrix
        """
        if self.best_model is None:
            self.logger.warning("No model trained yet")
            return None
        
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        return cm
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            feature_names (List[str]): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance ranking (top 10)
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
            }).sort_values('Importance', ascending=False).head(10)
            
            return importance_df
        else:
            self.logger.info(f"{self.best_model_name} does not have feature importance")
            return pd.DataFrame()
    
    def get_churn_segments(self, X: np.ndarray, 
                          segment_count: int = 3) -> np.ndarray:
        """
        Segment customers by churn risk probability.
        
        Args:
            X (np.ndarray): Features for prediction
            segment_count (int): Number of risk segments
        
        Returns:
            np.ndarray: Risk segments (0=low, 1=medium, 2=high)
        """
        if self.best_model is None:
            self.logger.warning("No model trained yet")
            return None
        
        churn_proba = self.best_model.predict_proba(X)[:, 1]
        segments = pd.qcut(churn_proba, q=segment_count, labels=False, duplicates='drop')
        
        return segments.values
    
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
    
    def predict_churn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict churn probability and binary classification.
        
        Args:
            X (np.ndarray): Features for prediction
        
        Returns:
            Tuple: (predictions, probabilities)
        """
        if self.best_model is None:
            self.logger.error("No trained model available")
            return None, None
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of modeling results.
        
        Returns:
            Dict: Model summary with best model info and metrics
        """
        return {
            'Best_Model': self.best_model_name,
            'Models_Trained': len(self.models),
            'Results': self.results.to_dict() if len(self.results) > 0 else {},
            'Best_F1': self.results['F1'].max() if len(self.results) > 0 else None,
            'Best_AUC': self.results[self.results['Model'] == self.best_model_name]['ROC_AUC'].values[0] if self.best_model_name else None,
            'Best_Accuracy': self.results[self.results['Model'] == self.best_model_name]['Accuracy'].values[0] if self.best_model_name else None
        }
