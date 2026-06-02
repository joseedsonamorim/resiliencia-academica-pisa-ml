"""Data loading, validation, and profiling layer."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
from src.config import get_config
from src.utils import compute_file_checksum, save_checksum, setup_logger, get_memory_usage

logger = setup_logger(__name__)


class DataValidator:
    """Validates dataset integrity and structure."""

    def __init__(self, config=None):
        """Initialize validator.

        Args:
            config: ConfigManager instance (uses global if None)
        """
        self.config = config or get_config()
        self.forbidden_features = self.config.get('data.forbidden_features', [])
        self.target_column = self.config.get('data.target_column')
        self.crt_column = self.config.get('data.crt_score_column')
        self.escs_column = self.config.get('data.escs_column')

    def validate_shape(self, df: pd.DataFrame, expected_rows: int = None):
        """Validate dataset dimensions.

        Args:
            df: DataFrame to validate
            expected_rows: Expected number of rows (optional)
        """
        rows, cols = df.shape
        logger.info(f" Dataset shape: {rows:,} × {cols}")

        if expected_rows and rows != expected_rows:
            logger.warning(f"  Expected {expected_rows:,} rows, got {rows:,}")

        return {'rows': rows, 'cols': cols}

    def validate_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Validate data types.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary of dtype issues
        """
        issues = {}

        # Check numeric columns
        for col in df.columns:
            try:
                # Try converting to numeric
                pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                issues[col] = str(e)

        if not issues:
            logger.info(f" All {len(df.columns)} columns are valid numeric/string")
        else:
            logger.warning(f"  {len(issues)} columns have type issues")

        return issues

    def check_leakage_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check for forbidden features that cause leakage.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary of leakage issues
        """
        leakage_found = {}

        for forbidden_col in self.forbidden_features:
            if forbidden_col in df.columns:
                leakage_found[forbidden_col] = True
                logger.warning(f"  LEAKAGE FEATURE DETECTED: '{forbidden_col}' must be removed")
            else:
                leakage_found[forbidden_col] = False

        return leakage_found

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check missing value percentages.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary of columns with missing %
        """
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        max_missing = self.config.get('data.max_missing_pct', 0.8)

        high_missing = missing_pct[missing_pct > (max_missing * 100)]

        if len(high_missing) > 0:
            logger.warning(f"  {len(high_missing)} columns have >{max_missing*100:.0f}% missing:")
            for col, pct in high_missing.head(10).items():
                logger.warning(f"    - {col}: {pct:.2f}%")
        else:
            logger.info(f" No columns with >{max_missing*100:.0f}% missing")

        return dict(missing_pct[missing_pct > 0])

    def check_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate rows.

        Args:
            df: DataFrame to check

        Returns:
            Number of duplicates
        """
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"  {n_duplicates} duplicate rows detected")
        else:
            logger.info(" No duplicate rows")

        return n_duplicates

    def validate_target(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate target variable distribution.

        Args:
            df: DataFrame with target column

        Returns:
            Dictionary of target statistics
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        target = df[self.target_column]
        unique_vals = target.unique()
        value_counts = target.value_counts()

        logger.info(f" Target variable '{self.target_column}':")
        for val, count in value_counts.items():
            pct = count / len(target) * 100
            logger.info(f"    - {val}: {count:,} ({pct:.2f}%)")

        return {
            'unique_values': int(len(unique_vals)),
            'class_distribution': value_counts.to_dict(),
            'imbalance_ratio': float(value_counts.max() / value_counts.min())
        }

    def validate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with all validation results
        """
        logger.info("=" * 80)
        logger.info(" STARTING DATA VALIDATION")
        logger.info("=" * 80)

        results = {
            'shape': self.validate_shape(df),
            'dtypes_issues': self.validate_dtypes(df),
            'leakage_features': self.check_leakage_features(df),
            'duplicates': self.check_duplicates(df),
            'missing_values': self.check_missing_values(df),
            'target_stats': self.validate_target(df)
        }

        # Summary
        logger.info("=" * 80)
        logger.info(" VALIDATION COMPLETE")
        logger.info("=" * 80)

        return results


class DataLoader:
    """Handles dataset loading and preprocessing."""

    def __init__(self, config=None):
        """Initialize loader.

        Args:
            config: ConfigManager instance (uses global if None)
        """
        self.config = config or get_config()
        self.dtype_numeric = self.config.get('data.dtype_numeric', 'float32')

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw PISA dataset.

        Returns:
            DataFrame with raw data
        """
        data_path = self.config.get_path('data_raw')

        logger.info(f" Loading data from: {data_path}")

        # Load with float32 for memory optimization
        df = pd.read_csv(
            data_path,
            dtype_backend='numpy_nullable'  # Use nullable dtypes
        )

        # Convert numeric columns to float32
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(np.float32)

        logger.info(f" Loaded {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f" Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

        return df

    def compute_data_checksums(self, data_path: Path) -> str:
        """Compute checksum of data file.

        Args:
            data_path: Path to data file

        Returns:
            Checksum string
        """
        logger.info(f" Computing data checksum...")
        checksum = compute_file_checksum(data_path)
        logger.info(f" Checksum: {checksum[:16]}...")
        return checksum


class DataProfiler:
    """Generates dataset profile metadata."""

    def __init__(self, config=None):
        """Initialize profiler.

        Args:
            config: ConfigManager instance (uses global if None)
        """
        self.config = config or get_config()

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete dataset profile.

        Args:
            df: DataFrame to profile

        Returns:
            Dictionary with profile information
        """
        logger.info(" Generating dataset profile...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        profile = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'dtypes': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols)
            },
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024**2)),
            'missing_values': {
                col: float(df[col].isnull().sum() / len(df) * 100)
                for col in df.columns
                if df[col].isnull().sum() > 0
            },
            'numeric_stats': {},
            'categorical_stats': {}
        }

        # Numeric statistics
        for col in numeric_cols:
            profile['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q1': float(df[col].quantile(0.25)),
                'q2': float(df[col].quantile(0.50)),
                'q3': float(df[col].quantile(0.75))
            }

        # Categorical statistics
        for col in categorical_cols:
            unique_count = df[col].nunique()
            value_counts = df[col].value_counts(normalize=True).head(10).to_dict()
            profile['categorical_stats'][col] = {
                'unique_count': unique_count,
                'top_values': value_counts
            }

        logger.info(f" Profile generated: {len(profile)} sections")

        return profile

    def save_profile(self, profile: Dict[str, Any], output_path: Path):
        """Save profile to JSON file.

        Args:
            profile: Profile dictionary
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)

        logger.info(f" Profile saved to {output_path}")

    def generate_and_save(self, df: pd.DataFrame):
        """Generate profile and save to standard location.

        Args:
            df: DataFrame to profile
        """
        profile = self.profile_dataset(df)
        output_path = self.config.get_path('data_processed') / 'dataset_profile.json'
        self.save_profile(profile, output_path)
