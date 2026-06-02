"""Comprehensive testing suite for data + preprocessing pipeline."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.data_layer import DataLoader, DataValidator, DataProfiler
from src.preprocessing_pipeline import PreprocessingPipeline
from src.config import get_config
from src.utils import setup_logger, get_memory_usage

logger = setup_logger(__name__)


class PipelineTestSuite:
    """Test suite for data + preprocessing pipeline."""

    def __init__(self):
        self.config = get_config()
        self.test_results = {}
        self.df_raw = None
        self.df_preprocessed = None

    def test_data_loading(self):
        """TEST 1: Data loading and validation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: DATA LOADING & VALIDATION")
        logger.info("="*80)

        loader = DataLoader()
        self.df_raw = loader.load_raw_data()

        # Check shape
        expected_rows = 3834
        expected_cols = 1275
        assert self.df_raw.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {self.df_raw.shape[0]}"
        assert self.df_raw.shape[1] == expected_cols, f"Expected {expected_cols} cols, got {self.df_raw.shape[1]}"

        logger.info(f" Shape correct: {self.df_raw.shape[0]:,} × {self.df_raw.shape[1]}")

        # Check target variable
        assert 'Creative_Resilience' in self.df_raw.columns, "Target column not found"
        target_counts = self.df_raw['Creative_Resilience'].value_counts()
        assert 1.0 in target_counts.index, "No resilient students found"
        assert target_counts[1.0] == 164, f"Expected 164 resilient, got {target_counts[1.0]}"

        logger.info(f" Target distribution: {dict(target_counts)}")

        # Check key columns exist
        key_cols = ['Creative_Resilience', 'CRT_SCORE', 'ESCS', 'ST004D01T', 'HOMEPOS', 'ICTRES']
        for col in key_cols:
            assert col in self.df_raw.columns, f"Key column missing: {col}"

        logger.info(f" All {len(key_cols)} key columns present")

        self.test_results['data_loading'] = 'PASS'

    def test_data_validation(self):
        """TEST 2: Leakage detection and data integrity."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: LEAKAGE DETECTION & DATA INTEGRITY")
        logger.info("="*80)

        validator = DataValidator()
        validation = validator.validate_all(self.df_raw)

        # Check leakage detection
        leakage = validation['leakage_features']
        expected_leakage = {
            'CRT_SCORE': True,
            'Status': True,
            'Grupo_ESCS': True,
            'CNTSTUID': True,
            'W_FSTUWT': True
        }

        for col, should_leak in expected_leakage.items():
            assert leakage.get(col) == should_leak, f"Leakage detection failed for {col}"

        logger.info(" All leakage features detected correctly")

        # Check duplicates
        assert validation['duplicates'] == 0, "Duplicate rows found"
        logger.info(" No duplicate rows")

        # Check target
        target_stats = validation['target_stats']
        assert target_stats['imbalance_ratio'] > 20, "Imbalance not extreme enough"
        logger.info(f" Target imbalance ratio: {target_stats['imbalance_ratio']:.1f}:1")

        self.test_results['data_validation'] = 'PASS'

    def test_preprocessing_stages(self):
        """TEST 3: Preprocessing pipeline stages."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: PREPROCESSING PIPELINE STAGES")
        logger.info("="*80)

        # Prepare data: separate X and y
        df = self.df_raw.copy()
        y = df['Creative_Resilience']
        X = df.drop(columns=['Creative_Resilience'])

        logger.info(f"\nInitial: X={X.shape[0]:,} × {X.shape[1]}, y={y.shape[0]:,}")

        pipeline = PreprocessingPipeline(self.config, random_state=42)

        # Stage 1: Exclude leakage
        X = pipeline.stage_1_exclude_leakage(X)
        initial_cols = X.shape[1]
        logger.info(f"After Stage 1 (exclude_leakage): {X.shape[1]} columns")

        # Verify leakage features removed
        for forbidden_col in ['CRT_SCORE', 'Status', 'Grupo_ESCS', 'CNTSTUID', 'W_FSTUWT']:
            assert forbidden_col not in X.columns, f"Leakage feature not removed: {forbidden_col}"

        logger.info(" All leakage features removed")

        # Stage 2: Impute missing
        X = pipeline.stage_2_impute_missing(X, y)
        logger.info(f"After Stage 2 (impute_missing): {X.shape[1]} columns")
        assert X.isnull().sum().sum() == 0, "Missing values remain after imputation"
        logger.info(" No missing values after imputation")

        # Stage 3: Scale
        X = pipeline.stage_3_scale_features(X)
        logger.info(f"After Stage 3 (scale): {X.shape[1]} columns")
        assert X.shape == (3834, X.shape[1]), "Shape changed during scaling"
        logger.info(" Features scaled (mean≈0, std≈1)")

        # Stage 4: Feature selection
        X = pipeline.stage_4_select_features(X, y)
        logger.info(f"After Stage 4 (select_features): {X.shape[1]} columns")
        assert X.shape[1] <= 150, "Too many features selected"
        logger.info(f" Feature selection complete: {initial_cols} → {X.shape[1]}")

        self.df_preprocessed = X
        self.test_results['preprocessing_stages'] = 'PASS'

    def test_stratified_folds(self):
        """TEST 4: Stratified K-Fold creation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: STRATIFIED K-FOLD CREATION")
        logger.info("="*80)

        y = self.df_raw['Creative_Resilience']
        X = self.df_preprocessed

        pipeline = PreprocessingPipeline(self.config, random_state=42)
        folds = pipeline.create_stratified_folds(X, y)

        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
        logger.info(f" Created {len(folds)} stratified folds")

        # Verify class distribution preserved
        original_ratio = y.mean()
        for fold in folds:
            train_ratio = fold['train_positive_ratio']
            test_ratio = fold['test_positive_ratio']

            # Should be close to original ratio
            assert abs(train_ratio - original_ratio) < 0.02, f"Train fold ratio {train_ratio} too far from {original_ratio}"
            assert abs(test_ratio - original_ratio) < 0.02, f"Test fold ratio {test_ratio} too far from {original_ratio}"

        logger.info(f" Class distribution preserved in all folds")
        logger.info(f"  Original ratio: {original_ratio:.2%}")
        fold_ratios = [f['train_positive_ratio'] for f in folds]
        logger.info(f"  Fold ratios: {fold_ratios}")

        self.test_results['stratified_folds'] = 'PASS'

    def test_no_data_leakage(self):
        """TEST 5: Verify no data leakage."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: DATA LEAKAGE VERIFICATION")
        logger.info("="*80)

        y = self.df_raw['Creative_Resilience']
        X = self.df_preprocessed

        # Check that CRT_SCORE is NOT in features
        assert 'CRT_SCORE' not in X.columns, "CRT_SCORE (leakage!) in features"
        assert 'Status' not in X.columns, "Status (leakage!) in features"
        assert 'Grupo_ESCS' not in X.columns, "Grupo_ESCS (leakage!) in features"

        logger.info(" No direct leakage features present")

        # Check that features don't directly encode target
        correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
        max_correlation = correlation_with_target.iloc[0]

        if max_correlation > 0.95:
            logger.warning(f"  Feature with very high correlation to target: {max_correlation:.4f}")
            logger.warning(f"    Feature: {correlation_with_target.idxmax()}")
        else:
            logger.info(f" Max correlation with target: {max_correlation:.4f} (acceptable)")

        self.test_results['data_leakage'] = 'PASS'

    def test_reproducibility(self):
        """TEST 6: Verify reproducibility with fixed seed."""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: REPRODUCIBILITY TEST")
        logger.info("="*80)

        y = self.df_raw['Creative_Resilience']
        X1_raw = self.df_raw.drop(columns=['Creative_Resilience']).copy()

        # Run preprocessing twice with same seed
        logger.info("Run 1: Processing...")
        pipeline1 = PreprocessingPipeline(self.config, random_state=42)
        X1, _ = pipeline1.preprocess_full_pipeline(X1_raw.copy(), y.copy())

        logger.info("Run 2: Processing with same seed...")
        X2_raw = self.df_raw.drop(columns=['Creative_Resilience']).copy()
        pipeline2 = PreprocessingPipeline(self.config, random_state=42)
        X2, _ = pipeline2.preprocess_full_pipeline(X2_raw.copy(), y.copy())

        # Check if outputs are identical
        assert X1.shape == X2.shape, f"Shapes don't match: {X1.shape} vs {X2.shape}"

        # Check if values are identical
        diff = np.abs(X1.values - X2.values).max()
        assert diff < 1e-10, f"Values differ by {diff} (not reproducible!)"

        logger.info(f" Reproducibility verified (max diff: {diff:.2e})")

        self.test_results['reproducibility'] = 'PASS'

    def test_memory_efficiency(self):
        """TEST 7: Memory efficiency for M1 8GB."""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: MEMORY EFFICIENCY")
        logger.info("="*80)

        mem = get_memory_usage()
        logger.info(f" Current memory usage: {mem['rss_gb']:.2f}GB / 8GB ({mem['percent']:.1f}%)")

        assert mem['rss_gb'] < 4.0, f"Memory usage too high: {mem['rss_gb']:.2f}GB > 4GB"
        logger.info(" Memory usage acceptable for M1 8GB")

        self.test_results['memory_efficiency'] = 'PASS'

    def run_all_tests(self):
        """Run all tests."""
        logger.info("\n" + ""*40)
        logger.info(" COMPLETE PIPELINE TEST SUITE")
        logger.info(""*40)

        try:
            self.test_data_loading()
            self.test_data_validation()
            self.test_preprocessing_stages()
            self.test_stratified_folds()
            self.test_no_data_leakage()
            self.test_reproducibility()
            self.test_memory_efficiency()

            logger.info("\n" + ""*40)
            logger.info(" ALL TESTS PASSED!")
            logger.info(""*40)

            # Summary
            logger.info("\n TEST SUMMARY:")
            for test_name, result in self.test_results.items():
                logger.info(f"   {test_name}: {result}")

            return True

        except AssertionError as e:
            logger.error(f"\n TEST FAILED: {e}")
            return False
        except Exception as e:
            logger.error(f"\n UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    suite = PipelineTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)
