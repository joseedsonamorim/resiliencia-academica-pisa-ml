"""Model training with Grid Search and stratified cross-validation."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

from src.config import get_config
from src.utils import setup_logger, save_metadata, Timer

logger = setup_logger(__name__)


class ModelTrainer:
    """Train multiple models with grid search and stratified CV."""

    def __init__(self, config=None, random_state=42):
        """Initialize trainer.

        Args:
            config: ConfigManager instance (uses global if None)
            random_state: Random seed
        """
        self.config = config or get_config()
        self.random_state = random_state
        self.models_dir = self.config.get_path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}

    def get_hyperparameters(self, model_name: str) -> Dict[str, list]:
        """Get hyperparameters for grid search.

        Args:
            model_name: Name of model

        Returns:
            Dictionary with parameters
        """
        params = self.config.get(f'model_training.hyperparameters.{model_name}', {})

        # Compute scale_pos_weight for imbalanced class
        if 'scale_pos_weight' in params and params['scale_pos_weight'] == 'auto':
            # For imbalanced data: 3670 neg vs 164 pos = 22.4:1
            params['scale_pos_weight'] = [22.4]

        return params

    def create_model(self, model_name: str) -> Any:
        """Create model instance.

        Args:
            model_name: Name of model

        Returns:
            Initialized model
        """
        if model_name == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, solver='liblinear')
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train single model with grid search.

        Args:
            model_name: Name of model
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with results
        """
        logger.info(f"\n Training {model_name.upper()}...")

        model = self.create_model(model_name)
        params = self.get_hyperparameters(model_name)

        if not params:
            logger.warning(f"    No hyperparameters found for {model_name}, using defaults")
            model.fit(X, y)
            self.trained_models[model_name] = model
            return {'status': 'trained_default'}

        n_splits = self.config.get('preprocessing.n_splits', 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        with Timer(f"GridSearch {model_name}", logger=logger):
            grid_search = GridSearchCV(
                model,
                params,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X, y)

        self.trained_models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        self.cv_results[model_name] = {
            'best_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'mean_cv_score': float(np.mean(grid_search.cv_results_['mean_test_score'])),
            'std_cv_score': float(np.std(grid_search.cv_results_['mean_test_score']))
        }

        logger.info(f"   Best ROC-AUC: {grid_search.best_score_:.4f}")
        logger.info(f"   Best params: {grid_search.best_params_}")

        return self.cv_results[model_name]

    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            model_name: Name of model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with metrics
        """
        model = self.trained_models[model_name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return metrics

    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Train all configured models.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*80)
        logger.info(" MODEL TRAINING - GRID SEARCH + STRATIFIED CV")
        logger.info("="*80)

        model_names = self.config.get('model_training.models', [])
        results = {}

        for model_name in model_names:
            try:
                result = self.train_model(model_name, X, y)
                results[model_name] = result
            except Exception as e:
                logger.error(f"   Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        logger.info("\n" + "="*80)
        logger.info(" TRAINING COMPLETE")
        logger.info("="*80)

        return results

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models on test set.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("\n EVALUATING MODELS ON TEST SET")

        evaluation = {}
        for model_name in self.trained_models.keys():
            try:
                metrics = self.evaluate_model(model_name, X_test, y_test)
                evaluation[model_name] = metrics

                logger.info(f"\n{model_name.upper()}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

            except Exception as e:
                logger.error(f"   Error evaluating {model_name}: {e}")
                evaluation[model_name] = {'error': str(e)}

        return evaluation

    def save_models(self):
        """Save trained models to disk."""
        logger.info("\n Saving trained models...")

        for model_name, model in self.trained_models.items():
            model_path = self.models_dir / f'{model_name}_model.pkl'
            joblib.dump(model, model_path)
            logger.info(f"   {model_name} saved to {model_path}")

        # Save all models dict
        all_models_path = self.models_dir / 'all_models.pkl'
        joblib.dump(self.trained_models, all_models_path)

        # Save CV results
        cv_results_path = self.models_dir / 'cv_results.json'
        save_metadata(self.cv_results, cv_results_path, "Cross-validation results")

        # Save best params
        params_path = self.models_dir / 'best_params.json'
        save_metadata(self.best_params, params_path, "Best hyperparameters")

    def load_models(self):
        """Load pre-trained models."""
        all_models_path = self.models_dir / 'all_models.pkl'
        if not all_models_path.exists():
            logger.warning(f"Models not found: {all_models_path}")
            return False

        self.trained_models = joblib.load(all_models_path)
        logger.info(f" Loaded {len(self.trained_models)} models")
        return True

    def get_best_model(self, metric='roc_auc') -> Tuple[str, Any]:
        """Get best model by metric.

        Args:
            metric: Metric to use for ranking

        Returns:
            Tuple of (model_name, model)
        """
        best_name = max(
            self.cv_results.keys(),
            key=lambda x: self.cv_results[x].get('best_score', 0)
        )
        return best_name, self.trained_models[best_name]
