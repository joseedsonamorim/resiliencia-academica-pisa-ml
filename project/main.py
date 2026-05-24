import logging
import gc
import sys
import time
import psutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_data import DataLoader
from target_resilience import TargetBuilder
from preprocessing import Preprocessor
from classification import ClassificationTrainer
from threshold import ThresholdOptimizer
from overfitting import OverfittingAnalyzer
from shap_analysis import SHAPAnalyzer
from fairness import FairnessAnalyzer
from clustering import ClusteringAnalyzer
from audit import ScientificAudit
from report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory():
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    logger.info(f"Memory usage: {rss_mb:.2f} MB")

def main():
    """Main pipeline execution."""
    start_time = time.time()
    log_memory()

    try:
        # Step 1: Load and merge data
        logger.info("="*60)
        logger.info("Step 1: Loading and merging datasets")
        logger.info("="*60)

        loader = DataLoader("project/data")
        cog_df = loader.load_stu_cog()
        qqq_df = loader.load_stu_qqq()
        merged_df = loader.merge_datasets(cog_df, qqq_df)
        merged_df = loader.rename_crt_columns(merged_df)
        loader.save_merged(merged_df)
        log_memory()

        # Step 2: Build target variable
        logger.info("\n" + "="*60)
        logger.info("Step 2: Building Creative_Resilience target")
        logger.info("="*60)

        target_builder = TargetBuilder(merged_df)
        df_with_target, target_stats = target_builder.build_target()
        target_builder.save_target_report(target_stats)
        log_memory()

        # Step 3: Preprocessing
        logger.info("\n" + "="*60)
        logger.info("Step 3: Preprocessing and feature engineering")
        logger.info("="*60)

        preprocessor = Preprocessor(df_with_target)
        features = preprocessor.identify_features()
        df_preprocessed = preprocessor.handle_missing_values(features)
        X, y = preprocessor.create_feature_matrix(features)
        log_memory()

        # Step 4: Train-test split
        logger.info("\n" + "="*60)
        logger.info("Step 4: Train-test split")
        logger.info("="*60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        log_memory()

        # Step 5: Model training
        logger.info("\n" + "="*60)
        logger.info("Step 5: Training classification models")
        logger.info("="*60)

        trainer = ClassificationTrainer(X_train, X_test, y_train, y_test)
        results = trainer.train_all_models(apply_smote=True)
        best_model_name, best_model, best_metrics = trainer.get_best_model()
        logger.info(f"Best model: {best_model_name} (F1: {best_metrics['f1']:.4f})")
        log_memory()

        # Step 6: Threshold optimization
        logger.info("\n" + "="*60)
        logger.info("Step 6: Optimizing decision threshold")
        logger.info("="*60)

        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        threshold_opt = ThresholdOptimizer(y_test, y_pred_proba)
        optimal_threshold, threshold_results = threshold_opt.optimize_threshold(metric='f1_recall')
        logger.info(f"Optimal threshold: {optimal_threshold:.2f}")

        # Re-evaluate with optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        best_metrics_optimal = trainer.evaluate_model(best_model_name, best_model, optimal_threshold)
        log_memory()

        # Step 7: Overfitting analysis
        logger.info("\n" + "="*60)
        logger.info("Step 7: Analyzing overfitting")
        logger.info("="*60)

        overfitting_analyzer = OverfittingAnalyzer(results)
        overfitting_analysis = overfitting_analyzer.analyze_overfitting()
        log_memory()

        # Step 8: Fairness analysis
        logger.info("\n" + "="*60)
        logger.info("Step 8: Fairness analysis")
        logger.info("="*60)

        fairness_analyzer = FairnessAnalyzer(df_preprocessed.iloc[X_test.index], y_pred_optimal,
                                            y_pred_proba, y_test.values)
        fairness_report = fairness_analyzer.get_fairness_report()
        log_memory()

        # Step 9: SHAP analysis
        logger.info("\n" + "="*60)
        logger.info("Step 9: SHAP explainability analysis")
        logger.info("="*60)

        shap_analyzer = SHAPAnalyzer(best_model, X_train, X_test, features)
        explainer, X_sample = shap_analyzer.create_explainer(sample_size=500)
        shap_values, X_test_sample = shap_analyzer.calculate_shap_values(explainer)
        shap_analyzer.plot_summary(explainer, shap_values, X_test_sample)
        shap_importance = shap_analyzer.get_feature_importance(shap_values)
        logger.info("SHAP analysis completed")
        log_memory()

        # Step 10: Clustering
        logger.info("\n" + "="*60)
        logger.info("Step 10: Clustering analysis")
        logger.info("="*60)

        clustering = ClusteringAnalyzer(X_train, n_clusters=3)
        X_pca = clustering.apply_pca(n_components=20)
        clusters = clustering.apply_kmeans(X_pca)
        clustering_metrics = clustering.calculate_metrics(X_pca)
        clustering.plot_clusters(X_pca)
        logger.info("Clustering completed")
        log_memory()

        # Step 11: Scientific audit
        logger.info("\n" + "="*60)
        logger.info("Step 11: Scientific audit")
        logger.info("="*60)

        audit = ScientificAudit(results, overfitting_analysis, fairness_report)
        audit_results = audit.audit()
        logger.info(f"Audit result - Publishable: {audit_results['publishable']}")
        log_memory()

        # Step 12: Generate reports
        logger.info("\n" + "="*60)
        logger.info("Step 12: Generating reports")
        logger.info("="*60)

        report_gen = ReportGenerator()
        report_gen.generate_model_report(results, best_model_name)
        report_gen.generate_audit_report(audit_results)

        # Prepare summary data
        summary_data = {
            'n_samples': len(df_with_target),
            'n_resilient': int(target_stats['n_resilient']),
            'pct_resilient': float(target_stats['pct_resilient']),
            'n_features': len(features),
            'best_model': best_model_name,
            'best_f1': float(best_metrics['f1']),
            'best_recall': float(best_metrics['recall']),
            'best_roc_auc': float(best_metrics['roc_auc']),
            'optimal_threshold': float(optimal_threshold),
            'publishable': bool(audit_results['publishable']),
            'audit_risk': audit_results.get('overall_risk', 'UNKNOWN'),
            'gender_disparity': fairness_report.get('gender_f1_disparity', None),
            'escs_disparity': fairness_report.get('escs_f1_disparity', None),
            'silhouette': clustering_metrics.get('silhouette_score', None),
            'davies_bouldin': clustering_metrics.get('davies_bouldin_score', None),
            'recommendations_text': '\n'.join(audit_results.get('recommendations', []))
        }

        report_gen.generate_summary_report(summary_data)
        report_gen.save_json_results(summary_data)

        # Execution time
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        log_memory()

        return summary_data

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
