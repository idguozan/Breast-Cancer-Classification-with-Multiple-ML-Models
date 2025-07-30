"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing breast cancer dataset"""
    
    def __init__(self, file_path: str):
        """
        DataProcessor class constructor
        
        Args:
            file_path (str): Path to CSV file
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads data from CSV file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise
    
    def preprocess_data(self, target_column: str = "diagnosis") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses the data
        
        Args:
            target_column (str): Name of target variable column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target variable
        """
        if self.data is None:
            raise ValueError("Load data first!")
        
        # Check target column
        if target_column not in self.data.columns:
            raise ValueError(f"'{target_column}' column not found in dataset!")
        
        # Check missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        # Encode target variable (M: 0 - Malignant, B: 1 - Benign)
        target_mapping = {"M": 0, "B": 1}
        if self.data[target_column].dtype == 'object':
            self.data["target"] = self.data[target_column].map(target_mapping)
        else:
            self.data["target"] = self.data[target_column]
        
        # Remove unnecessary columns
        columns_to_drop = [target_column, "target"]
        if "id" in self.data.columns:
            columns_to_drop.append("id")
        if "Unnamed: 32" in self.data.columns:  # Common empty column in breast cancer dataset
            columns_to_drop.append("Unnamed: 32")
            
        self.X = self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns])
        self.y = self.data["target"]
        
        # Clean non-numeric columns
        numeric_columns = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_columns]
        
        logger.info(f"Preprocessing completed. Number of features: {self.X.shape[1]}")
        logger.info(f"Class distribution:\n{self.y.value_counts()}")
        
        return self.X, self.y
    
    def get_data_info(self) -> dict:
        """
        Returns basic information about the data
        
        Returns:
            dict: Data information
        """
        if self.data is None:
            raise ValueError("Load data first!")
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "target_distribution": self.y.value_counts().to_dict() if self.y is not None else None
        }
        
        return info
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42, 
                   scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into training and test sets
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed
            scale_features (bool): Whether to scale features
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if self.X is None or self.y is None:
            raise ValueError("Preprocess data first!")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y
        )
        
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            logger.info("Features scaled (StandardScaler)")
        
        logger.info(f"Data split - Training: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
