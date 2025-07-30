
"""
Model training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from typing import Dict, Tuple, List, Any
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating machine learning models"""
    
    def __init__(self):
        """ModelTrainer class constructor"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initializes default models
        
        Returns:
            Dict[str, Any]: Model dictionary
        """
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=2000, 
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "SVM": SVC(
                probability=True, 
                random_state=42
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5
            ),
            "Naive Bayes": GaussianNB()
        }
        
        logger.info(f"{len(self.models)} models initialized")
        return self.models
    
    def add_model(self, name: str, model: Any) -> None:
        """
        Adds a new model
        
        Args:
            name (str): Model name
            model (Any): Scikit-learn model
        """
        self.models[name] = model
        logger.info(f"'{name}' model added")
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Trains and evaluates all models
        
        Args:
            X_train: Training features
            y_train: Training target variable
            X_test: Test features
            y_test: Test target variable
            
        Returns:
            Dict[str, Dict]: Model results
        """
        if not self.models:
            self.initialize_models()
        
        logger.info("Model training started...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Model training
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            metrics['cv_score_mean'] = cv_scores.mean()
            metrics['cv_score_std'] = cv_scores.std()
            
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'y_test': y_test
            }
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f} ± {metrics['cv_score_std']:.4f}")
        
        # Find best model
        self._find_best_model()
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Calculates model performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dict[str, float]: Metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def _find_best_model(self) -> None:
        """Determines best model based on AUC score"""
        best_auc = 0
        for name, result in self.results.items():
            auc_score = result['metrics'].get('auc', 0)
            if auc_score > best_auc:
                best_auc = auc_score
                self.best_model_name = name
                self.best_model = result['model']
        
        if self.best_model_name:
            logger.info(f"Best model: {self.best_model_name} (AUC: {best_auc:.4f})")
    
    def hyperparameter_tuning(self, model_name: str, param_grid: Dict, 
                            X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Performs hyperparameter optimization for specified model
        
        Args:
            model_name (str): Model name
            param_grid (Dict): Parameter grid
            X_train: Training features
            y_train: Training target variable
            
        Returns:
            Any: Optimized model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found!")
        
        logger.info(f"Hyperparameter optimization started for {model_name}...")
        
        grid_search = GridSearchCV(
            self.models[model_name], 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update best model
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_classification_report(self, model_name: str) -> str:
        """
        Returns detailed classification report for specified model
        
        Args:
            model_name (str): Model name
            
        Returns:
            str: Classification report
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not trained yet!")
        
        result = self.results[model_name]
        return classification_report(
            result['y_test'], 
            result['y_pred'],
            target_names=['Malignant', 'Benign']
        )
    
    def get_feature_importance(self, model_name: str = None) -> pd.Series:
        """
        Returns model feature importance
        
        Args:
            model_name (str, optional): Model name. Uses best model if None.
            
        Returns:
            pd.Series: Feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.results:
                raise ValueError(f"Model '{model_name}' not found!")
            model = self.results[model_name]['model']
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"{model_name} model does not support feature importance!")
        
        # Use placeholder for feature names
        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
        
        importance_series = pd.Series(
            model.feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False)
        
        return importance_series
    
    def save_model(self, model_name: str, file_path: str) -> None:
        """
        Saves model to file
        
        Args:
            model_name (str): Model name
            file_path (str): Save path
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found!")
        
        model = self.results[model_name]['model']
        joblib.dump(model, file_path)
        logger.info(f"{model_name} model saved to '{file_path}'")
    
    def load_model(self, file_path: str, model_name: str = None) -> Any:
        """
        Loads model from file
        
        Args:
            file_path (str): Model file path
            model_name (str, optional): Model name
            
        Returns:
            Any: Loaded model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model = joblib.load(file_path)
        
        if model_name:
            self.models[model_name] = model
            logger.info(f"Model loaded as '{model_name}'")
        
        return model
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Returns summary table of all model results
        
        Returns:
            pd.DataFrame: Results summary
        """
        if not self.results:
            raise ValueError("No models trained yet!")
        
        summary_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics.get('auc', 'N/A'),
                'CV Score (Mean)': metrics['cv_score_mean'],
                'CV Score (Std)': metrics['cv_score_std']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('AUC', ascending=False) if 'AUC' in df.columns else df.sort_values('Accuracy', ascending=False)
        
        return df
