"""Interpretability: SHAP analysis and fairness metrics."""
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.config import get_config
from src.utils import setup_logger, save_metadata

logger = setup_logger(__name__)


class SHAPAnalyzer:
    """SHAP values for model interpretability."""

    def __init__(self, config=None):
        """Initialize analyzer.

        Args:
            config: ConfigManager instance
        """
        self.config = config or get_config()
        self.max_samples = self.config.get('interpretability.shap_max_samples', 500)

    def compute_shap_values(
        self,
        model,
        X_background: pd.DataFrame,
        X_explain: pd.DataFrame = None,
        explainer_type: str = 'tree'
    ) -> Dict[str, Any]:
        """Compute SHAP values for model.

        Args:
            model: Trained model
            X_background: Background data for SHAP
            X_explain: Data to explain (if None, use background)
            explainer_type: Type of explainer ('tree', 'kernel', 'linear')

        Returns:
            Dictionary with SHAP analysis
        """
        logger.info(f"\n Computing SHAP values (explainer: {explainer_type})...")

        if X_explain is None:
            X_explain = X_background

        # Limit samples for memory efficiency
        if len(X_background) > self.max_samples:
            logger.info(f"  Limiting background to {self.max_samples} samples (from {len(X_background)})")
            bg_indices = np.random.choice(len(X_background), self.max_samples, replace=False)
            X_background = X_background.iloc[bg_indices]

        if len(X_explain) > self.max_samples:
            logger.info(f"  Limiting explain to {self.max_samples} samples (from {len(X_explain)})")
            ex_indices = np.random.choice(len(X_explain), self.max_samples, replace=False)
            X_explain = X_explain.iloc[ex_indices]

        try:
            # Create explainer based on model type
            if explainer_type == 'tree' and hasattr(model, 'booster'):
                logger.info("  Using TreeExplainer...")
                explainer = shap.TreeExplainer(model)
            else:
                logger.info("  Using KernelExplainer (may be slow)...")
                explainer = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    X_background
                )

            shap_values = explainer.shap_values(X_explain)

            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class

            # Compute global feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_names = X_explain.columns.tolist()

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            logger.info(f"   SHAP values computed")
            logger.info(f"  Top 5 features:")
            for idx, row in importance_df.head(5).iterrows():
                logger.info(f"      {row['feature']}: {row['importance']:.4f}")

            return {
                'shap_values': shap_values,
                'feature_importance': importance_df.to_dict('records'),
                'explainer': explainer,
                'X_explain': X_explain
            }

        except Exception as e:
            logger.error(f"   Error computing SHAP: {e}")
            return None

    def save_shap_results(self, results: Dict[str, Any], output_path: Path):
        """Save SHAP analysis results.

        Args:
            results: SHAP results dictionary
            output_path: Output file path
        """
        if results is None:
            return

        # Save feature importance
        importance_data = {
            'feature_importance': results.get('feature_importance', [])
        }
        save_metadata(importance_data, output_path, "SHAP feature importance")


class FairnessAnalyzer:
    """Fairness analysis across protected attributes."""

    def __init__(self, config=None):
        """Initialize analyzer.

        Args:
            config: ConfigManager instance
        """
        self.config = config or get_config()

    def analyze_fairness(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_attr: pd.Series,
        attr_name: str = 'protected_attribute'
    ) -> Dict[str, Any]:
        """Analyze fairness across protected attribute groups.

        Args:
            y_true: True labels
            y_pred: Binary predictions
            y_pred_proba: Predicted probabilities
            protected_attr: Protected attribute (e.g., gender, race, socioeconomic status)
            attr_name: Name of attribute

        Returns:
            Dictionary with fairness metrics
        """
        logger.info(f"\n  Analyzing fairness across {attr_name}...")

        # Get unique groups
        groups = protected_attr.unique()
        results = {
            'attribute': attr_name,
            'groups': {}
        }

        for group in sorted(groups):
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            y_pred_proba_group = y_pred_proba[mask]

            from sklearn.metrics import (
                roc_auc_score, precision_score, recall_score,
                f1_score, confusion_matrix
            )

            group_metrics = {
                'size': int(mask.sum()),
                'positive_ratio': float(y_true_group.mean()),
            }

            try:
                group_metrics['roc_auc'] = float(roc_auc_score(y_true_group, y_pred_proba_group))
            except:
                group_metrics['roc_auc'] = None

            try:
                group_metrics['precision'] = float(precision_score(y_true_group, y_pred_group, zero_division=0))
            except:
                group_metrics['precision'] = None

            try:
                group_metrics['recall'] = float(recall_score(y_true_group, y_pred_group, zero_division=0))
            except:
                group_metrics['recall'] = None

            try:
                group_metrics['f1'] = float(f1_score(y_true_group, y_pred_group, zero_division=0))
            except:
                group_metrics['f1'] = None

            # Confusion matrix based metrics
            try:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                group_metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
                group_metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
                group_metrics['tpr'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
                group_metrics['selection_rate'] = float(y_pred_group.mean())
            except:
                pass

            results['groups'][str(group)] = group_metrics

            logger.info(f"  Group {group} (n={mask.sum()}):")
            logger.info(f"      Positive ratio: {group_metrics['positive_ratio']:.2%}")
            if group_metrics.get('f1'):
                logger.info(f"      F1: {group_metrics['f1']:.4f}")
            if group_metrics.get('recall'):
                logger.info(f"      Recall (TPR): {group_metrics['recall']:.4f}")
            if group_metrics.get('fpr'):
                logger.info(f"      FPR: {group_metrics['fpr']:.4f}")

        # Compute fairness disparities
        results['disparities'] = self._compute_disparities(results['groups'])

        logger.info(f"\n  Fairness Disparities:")
        for disparity_name, disparity_val in results['disparities'].items():
            if disparity_val is not None:
                logger.info(f"      {disparity_name}: {disparity_val:.4f}")

        return results

    def _compute_disparities(self, groups_data: Dict[str, Dict]) -> Dict[str, float]:
        """Compute fairness disparities between groups.

        Args:
            groups_data: Dictionary with metrics per group

        Returns:
            Dictionary with disparity metrics
        """
        disparities = {}
        group_list = list(groups_data.values())

        if len(group_list) >= 2:
            # Equal Opportunity Difference (EOD)
            tprs = [g.get('tpr') for g in group_list if g.get('tpr') is not None]
            if len(tprs) >= 2:
                disparities['equal_opportunity'] = float(max(tprs) - min(tprs))

            # Demographic Parity Difference (DPD)
            sel_rates = [g.get('selection_rate') for g in group_list if g.get('selection_rate') is not None]
            if len(sel_rates) >= 2:
                disparities['demographic_parity'] = float(max(sel_rates) - min(sel_rates))

            # False Positive Rate Difference
            fprs = [g.get('fpr') for g in group_list if g.get('fpr') is not None]
            if len(fprs) >= 2:
                disparities['fpr_difference'] = float(max(fprs) - min(fprs))

            # False Negative Rate Difference
            fnrs = [g.get('fnr') for g in group_list if g.get('fnr') is not None]
            if len(fnrs) >= 2:
                disparities['fnr_difference'] = float(max(fnrs) - min(fnrs))

        return disparities

    def analyze_all_protected_attributes(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze fairness across multiple protected attributes.

        Args:
            y_true: True labels
            y_pred: Binary predictions
            y_pred_proba: Predicted probabilities
            protected_data: DataFrame with protected attributes

        Returns:
            Dictionary with fairness analysis for each attribute
        """
        logger.info("\n" + "="*80)
        logger.info(" FAIRNESS ANALYSIS - ALL PROTECTED ATTRIBUTES")
        logger.info("="*80)

        all_results = {}

        for attr_name in protected_data.columns:
            try:
                results = self.analyze_fairness(
                    y_true,
                    y_pred,
                    y_pred_proba,
                    protected_data[attr_name],
                    attr_name=attr_name
                )
                all_results[attr_name] = results
            except Exception as e:
                logger.error(f"   Error analyzing {attr_name}: {e}")
                all_results[attr_name] = {'error': str(e)}

        logger.info("\n" + "="*80)
        logger.info(" FAIRNESS ANALYSIS COMPLETE")
        logger.info("="*80)

        return all_results
