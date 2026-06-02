"""Preprocessing pipeline with 5 sequential deterministic stages."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import psutil

from src.config import get_config
from src.utils import setup_logger, save_metadata, Timer

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """Deterministic preprocessing pipeline with 5 stages and joblib caching."""

    def __init__(self, config=None, random_state=42):
        """Initialize pipeline.

        Args:
            config: ConfigManager instance (uses global if None)
            random_state: Random seed for reproducibility
        """
        self.config = config or get_config()
        self.random_state = random_state
        self.cache_dir = self.config.get_path('data_cache')
        self.output_dir = self.config.get_path('data_processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.transformers = {}
        self.feature_names = None
        self.transformation_log = []

    def stage_1_exclude_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1: Remove features that cause data leakage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with leakage features removed
        """
        logger.info("\n STAGE 1: Excluding leakage features")

        forbidden_features = self.config.get('data.forbidden_features', [])
        cols_to_drop = [col for col in forbidden_features if col in df.columns]

        if cols_to_drop:
            logger.info(f"   Removing {len(cols_to_drop)} leakage features:")
            for col in cols_to_drop:
                logger.info(f"      - {col}")
            df = df.drop(columns=cols_to_drop)

        self.transformation_log.append({
            'stage': 1,
            'operation': 'exclude_leakage',
            'removed_columns': len(cols_to_drop),
            'remaining_columns': df.shape[1]
        })

        logger.info(f"   Result: {df.shape[0]:,} × {df.shape[1]} (removed {len(cols_to_drop)})")

        return df

    def stage_2_impute_missing(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Stage 2: Impute missing values using KNN.

        Args:
            X: Feature matrix
            y: Target variable (optional, for stratification)

        Returns:
            Imputed feature matrix
        """
        logger.info("\n STAGE 2: Imputing missing values")

        # Drop columns with >80% missing
        max_missing_pct = self.config.get('data.max_missing_pct', 0.8)
        cols_missing_pct = X.isnull().sum() / len(X)
        cols_to_drop = cols_missing_pct[cols_missing_pct > max_missing_pct].index.tolist()

        if cols_to_drop:
            logger.info(f"    Dropping {len(cols_to_drop)} columns with >{max_missing_pct*100:.0f}% missing")
            X = X.drop(columns=cols_to_drop)

        # Impute remaining missing values
        n_neighbors = self.config.get('data.knn_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)

        X_imputed = imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        self.transformers['imputer'] = imputer

        self.transformation_log.append({
            'stage': 2,
            'operation': 'impute_missing',
            'method': 'knn',
            'k_neighbors': n_neighbors,
            'columns_dropped': len(cols_to_drop),
            'columns_imputed': X_imputed.shape[1]
        })

        missing_pct = X_imputed.isnull().sum().sum() / (X_imputed.shape[0] * X_imputed.shape[1]) * 100
        logger.info(f"   Result: {missing_pct:.2f}% missing remaining")

        return X_imputed

    def stage_3_scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Scale features using StandardScaler.

        Args:
            X: Feature matrix

        Returns:
            Scaled feature matrix
        """
        logger.info("\n STAGE 3: Scaling features")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        self.transformers['scaler'] = scaler

        self.transformation_log.append({
            'stage': 3,
            'operation': 'scale_features',
            'method': 'standard_scaler',
            'features_scaled': X_scaled.shape[1]
        })

        logger.info(f"   Mean: {X_scaled.mean().mean():.6f}, Std: {X_scaled.std().mean():.6f}")

        return X_scaled

    def stage_4_select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Stage 4: Feature selection using RFE.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Selected feature matrix
        """
        logger.info("\n STAGE 4: Feature selection (RFE)")

        n_features = self.config.get('preprocessing.n_features_to_select', 150)

        if n_features >= X.shape[1]:
            logger.info(f"    n_features ({n_features}) >= total features ({X.shape[1]})")
            logger.info(f"  → Keeping all {X.shape[1]} features")
            selected_cols = X.columns.tolist()
        else:
            # RFE with LogisticRegression
            estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )

            logger.info(f"   Selecting {n_features} from {X.shape[1]} features using RFE")

            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)

            selected_cols = X.columns[selector.support_].tolist()
            self.transformers['selector'] = selector

        X_selected = X[selected_cols]

        self.transformation_log.append({
            'stage': 4,
            'operation': 'feature_selection',
            'method': 'rfe',
            'features_before': X.shape[1],
            'features_after': X_selected.shape[1],
            'selected_features': selected_cols[:10]  # First 10 for logging
        })

        logger.info(f"   Result: {X_selected.shape[1]} features selected")

        # Store feature names for later use
        self.feature_names = selected_cols

        return X_selected

    def create_stratified_folds(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """Create stratified K-Fold splits.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            List of fold dictionaries with train/test indices
        """
        logger.info("\n Creating stratified K-Fold splits")

        n_splits = self.config.get('preprocessing.n_splits', 5)
        cv_shuffle = self.config.get('preprocessing.cv_shuffle', True)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=cv_shuffle,
            random_state=self.random_state
        )

        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_info = {
                'fold_id': fold_idx,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_positive_ratio': y.iloc[train_idx].mean(),
                'test_positive_ratio': y.iloc[test_idx].mean()
            }
            folds.append(fold_info)

            logger.info(f"  Fold {fold_idx}: "
                       f"train {len(train_idx):,} (pos: {fold_info['train_positive_ratio']:.2%}), "
                       f"test {len(test_idx):,} (pos: {fold_info['test_positive_ratio']:.2%})")

        return folds

    def apply_smote_train_only(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Stage 5: Apply SMOTE to training fold ONLY.

        Args:
            X_train: Training feature matrix
            y_train: Training target variable

        Returns:
            Resampled (X_train, y_train)
        """
        logger.info(f"\n STAGE 5: Applying SMOTE (training fold only)")

        smote_enabled = self.config.get('preprocessing.smote_enabled', True)

        if not smote_enabled:
            logger.info("    SMOTE disabled in config")
            return X_train, y_train

        logger.info(f"  Original class distribution: {y_train.value_counts().to_dict()}")

        smote = SMOTE(
            k_neighbors=self.config.get('preprocessing.smote_k_neighbors', 5),
            random_state=self.random_state,
            sampling_strategy='auto'
        )

        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)

        self.transformers['smote'] = smote

        logger.info(f"   Resampled class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        logger.info(f"   New size: {len(X_train_resampled):,} samples")

        return X_train_resampled, y_train_resampled

    def preprocess_full_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Run full 4-stage preprocessing (stages 1-4).

        Note: Stage 5 (SMOTE) is applied per-fold during CV, not here.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Preprocessed (X, y)
        """
        logger.info("\n" + "=" * 80)
        logger.info(" STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 80)

        with Timer("Full preprocessing", logger=logger):
            # Stage 1: Exclude leakage
            X = self.stage_1_exclude_leakage(X)

            # Stage 2: Impute missing
            X = self.stage_2_impute_missing(X, y)

            # Stage 3: Scale
            X = self.stage_3_scale_features(X)

            # Stage 4: Select features
            X = self.stage_4_select_features(X, y)

        logger.info("\n" + "=" * 80)
        logger.info(" PREPROCESSING COMPLETE")
        logger.info("=" * 80)

        return X, y

    def save_pipeline(self):
        """Save transformers and metadata to cache."""
        logger.info("\n Saving preprocessing pipeline...")

        # Save transformers
        pipeline_path = self.output_dir / 'preprocessing_pipeline.pkl'
        joblib.dump(self.transformers, pipeline_path)
        logger.info(f"   Transformers saved to {pipeline_path}")

        # Save feature names
        feature_names_path = self.output_dir / 'feature_names.pkl'
        joblib.dump(self.feature_names, feature_names_path)
        logger.info(f"   Feature names saved to {feature_names_path}")

        # Save transformation log
        log_path = self.output_dir / 'transformation_log.json'
        save_metadata(self.transformation_log, log_path, "Preprocessing transformation log")

    def load_pipeline(self):
        """Load pre-trained transformers."""
        logger.info("\n Loading preprocessing pipeline...")

        pipeline_path = self.output_dir / 'preprocessing_pipeline.pkl'
        if not pipeline_path.exists():
            logger.warning(f"    Pipeline not found: {pipeline_path}")
            return False

        self.transformers = joblib.load(pipeline_path)

        feature_names_path = self.output_dir / 'feature_names.pkl'
        if feature_names_path.exists():
            self.feature_names = joblib.load(feature_names_path)

        logger.info("   Pipeline loaded successfully")
        return True
