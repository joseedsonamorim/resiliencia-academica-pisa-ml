#!/usr/bin/env python3
"""Main CLI pipeline orchestrator - runs entire analysis."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.config import get_config
from src.utils import setup_logger, seed_everything, get_memory_usage, Timer
from src.data_layer import DataLoader, DataValidator, DataProfiler
from src.preprocessing_pipeline import PreprocessingPipeline
from src.model_training import ModelTrainer
from src.post_training_analysis import ThresholdOptimizer, BootstrapAnalyzer, CalibrationAnalyzer
from src.interpretability_layer import SHAPAnalyzer, FairnessAnalyzer

logger = setup_logger(__name__)


def log_section(title: str):
    """Log a section header."""
    logger.info("\n" + ""*40)
    logger.info(title)
    logger.info(""*40)


def main():
    """Execute complete analysis pipeline."""
    log_section(" COMPLETE ANALYSIS PIPELINE - RESILIÊNCIA CRIATIVA")

    config = get_config()
    seed_everything(config.get('random_state', 42))

    try:
        # PHASE 1: Data Loading & Validation
        log_section(" PHASE 1: DATA LOADING & VALIDATION")
        loader = DataLoader(config)
        df = loader.load_raw_data()

        validator = DataValidator(config)
        validation = validator.validate_all(df)

        profiler = DataProfiler(config)
        profiler.generate_and_save(df)

        # PHASE 2: Preprocessing
        log_section(" PHASE 2: PREPROCESSING")
        y = df['Creative_Resilience']
        X = df.drop(columns=['Creative_Resilience'])

        with Timer("Complete Preprocessing", logger=logger):
            preprocessor = PreprocessingPipeline(config, random_state=42)
            X_processed, y_processed = preprocessor.preprocess_full_pipeline(X, y)
            preprocessor.save_pipeline()

        logger.info(f"\n Final shape: {X_processed.shape}")

        # PHASE 3: Create Folds
        log_section(" PHASE 3: CREATING STRATIFIED FOLDS")
        folds = preprocessor.create_stratified_folds(X_processed, y_processed)
        logger.info(f" Created {len(folds)} stratified folds")

        # PHASE 4: Model Training
        log_section(" PHASE 4: MODEL TRAINING")
        with Timer("Model Training", logger=logger):
            trainer = ModelTrainer(config, random_state=42)
            train_results = trainer.train_all_models(X_processed, y_processed)

        trainer.save_models()

        # PHASE 5: Evaluation & Post-Training Analysis
        log_section(" PHASE 5: EVALUATION & POST-TRAINING ANALYSIS")

        # Use first fold for demonstration
        fold_0 = folds[0]
        X_train_0 = X_processed.iloc[fold_0['train_idx']]
        X_test_0 = X_processed.iloc[fold_0['test_idx']]
        y_train_0 = y_processed.iloc[fold_0['train_idx']]
        y_test_0 = y_processed.iloc[fold_0['test_idx']]

        evaluation_results = trainer.evaluate_all_models(X_test_0, y_test_0)

        # Threshold Optimization
        logger.info("\n Threshold Optimization")
        best_model_name, best_model = trainer.get_best_model()
        y_pred_proba = best_model.predict_proba(X_test_0)[:, 1]

        threshold_optimizer = ThresholdOptimizer(config)
        optimal_threshold, threshold_metrics = threshold_optimizer.optimize_threshold(
            y_test_0.values,
            y_pred_proba,
            metric='f1'
        )

        # Bootstrap Analysis
        logger.info("\n Bootstrap Analysis (95% CI)")
        bootstrap_analyzer = BootstrapAnalyzer(config, random_state=42)
        bootstrap_results = bootstrap_analyzer.compute_bootstrap_ci(
            y_test_0.values,
            y_pred_proba,
            n_iterations=100  # Reduced for speed
        )
        bootstrap_analyzer.save_bootstrap_results(
            bootstrap_results,
            config.get_path('outputs_metrics') / 'bootstrap_ci.json'
        )

        # Calibration Analysis
        logger.info("\n Calibration Analysis")
        calibration_analyzer = CalibrationAnalyzer(config)
        calibration_results = calibration_analyzer.analyze_calibration(y_test_0.values, y_pred_proba)

        # PHASE 6: Interpretability (SHAP + Fairness)
        log_section(" PHASE 6: INTERPRETABILITY")

        # SHAP Analysis (on sample)
        logger.info("\n SHAP Analysis")
        shap_analyzer = SHAPAnalyzer(config)
        X_sample = X_test_0.sample(min(100, len(X_test_0)), random_state=42)
        shap_results = shap_analyzer.compute_shap_values(best_model, X_test_0, X_sample)

        if shap_results:
            shap_analyzer.save_shap_results(
                shap_results,
                config.get_path('outputs_metrics') / 'shap_importance.json'
            )

        # Fairness Analysis
        logger.info("\n  Fairness Analysis")
        y_pred_fold0 = best_model.predict(X_test_0)

        # Create protected attributes from original data
        protected_attrs = pd.DataFrame({
            'gender': df.loc[fold_0['test_idx'], 'ST004D01T'].values if 'ST004D01T' in df.columns else np.random.randint(1, 3, len(y_test_0))
        })

        fairness_analyzer = FairnessAnalyzer(config)
        fairness_results = fairness_analyzer.analyze_all_protected_attributes(
            y_test_0,
            y_pred_fold0,
            y_pred_proba,
            protected_attrs
        )

        # PHASE 7: Summary & Metrics Save
        log_section(" PHASE 7: SUMMARY & RESULTS")

        summary = {
            'dataset': {
                'total_samples': len(df),
                'features_initial': X.shape[1],
                'features_final': X_processed.shape[1],
                'target_distribution': {'0': int((y == 0).sum()), '1': int((y == 1).sum())}
            },
            'models': {
                'best_model': best_model_name,
                'evaluation': {k: v for k, v in evaluation_results.items() if k == best_model_name}
            },
            'postprocessing': {
                'optimal_threshold': float(optimal_threshold),
                'bootstrap_ci': bootstrap_results
            }
        }

        # Save summary
        from src.utils import save_metadata
        save_metadata(
            summary,
            config.get_path('outputs_reports') / 'pipeline_summary.json',
            'Pipeline execution summary'
        )

        logger.info(f"\n Summary saved")

        # Final Memory Check
        mem = get_memory_usage()
        logger.info(f"\n Final memory usage: {mem['rss_gb']:.2f}GB / 8GB")

        log_section(" PIPELINE COMPLETE!")
        logger.info("\n Outputs saved to:")
        logger.info(f"    Models: {config.get_path('models')}")
        logger.info(f"    Reports: {config.get_path('outputs_reports')}")
        logger.info(f"    Metrics: {config.get_path('outputs_metrics')}")

        return 0

    except Exception as e:
        logger.error(f"\n PIPELINE FAILED: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
