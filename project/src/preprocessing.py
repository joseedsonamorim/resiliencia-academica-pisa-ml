import pandas as pd
import numpy as np
import logging
import gc
from sklearn.preprocessing import StandardScaler
from pathlib import Path

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(__name__)
        self.allowed_features = [
            'ESCS', 'HOMEPOS', 'HISCED', 'ICTRES', 'FLCONICT',
            'JOYREAD', 'OPENPS', 'ENTUSE', 'ST004D01T'
        ]

    def identify_features(self) -> list:
        """Identify allowed features present in dataset."""
        available_features = [f for f in self.allowed_features if f in self.df.columns]
        self.logger.info(f"Found {len(available_features)} allowed features")
        return available_features

    def handle_missing_values(self, features: list, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values with memory-efficient approach."""
        try:
            df = self.df.copy()

            for feature in features:
                if feature in df.columns:
                    missing_count = df[feature].isna().sum()
                    if missing_count > 0:
                        if strategy == 'mean':
                            fill_value = df[feature].mean()
                            df[feature].fillna(fill_value, inplace=True)
                        elif strategy == 'median':
                            fill_value = df[feature].median()
                            df[feature].fillna(fill_value, inplace=True)
                        self.logger.info(f"{feature}: {missing_count} missing values handled")

            gc.collect()
            return df

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def create_feature_matrix(self, features: list) -> tuple:
        """Create feature matrix X and target y."""
        try:
            if 'Creative_Resilience' not in self.df.columns:
                raise ValueError("Creative_Resilience target not found")

            X = self.df[features].copy()
            y = self.df['Creative_Resilience'].copy()

            # Convert to float32 for memory efficiency
            for col in X.columns:
                if X[col].dtype == 'float64':
                    X[col] = X[col].astype('float32')
                elif X[col].dtype == 'int64':
                    if X[col].min() >= np.iinfo(np.int32).min and X[col].max() <= np.iinfo(np.int32).max:
                        X[col] = X[col].astype('int32')

            self.logger.info(f"Feature matrix created: {X.shape[0]} samples, {X.shape[1]} features")
            gc.collect()
            return X, y

        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {str(e)}")
            raise

    def get_data_info(self) -> dict:
        """Get dataset information."""
        return {
            'n_samples': len(self.df),
            'n_features': self.df.shape[1],
            'missing_values': self.df.isna().sum().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Preprocessor module loaded successfully")
