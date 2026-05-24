import pandas as pd
import numpy as np
import logging
import gc
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, auc)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import psutil

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ClassificationTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}

    def apply_smote(self):
        """Apply SMOTE only to training data."""
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                self.X_train, self.y_train
            )
            self.logger.info(f"SMOTE applied. Train class distribution: {np.bincount(y_train_balanced)}")
            gc.collect()
            return X_train_balanced, y_train_balanced

        except Exception as e:
            self.logger.error(f"Error applying SMOTE: {str(e)}")
            raise

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression with memory optimization."""
        try:
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                solver='lbfgs',
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            self.models['LogisticRegression'] = model
            self.logger.info("LogisticRegression trained")
            gc.collect()
            return model

        except Exception as e:
            self.logger.error(f"Error training LogisticRegression: {str(e)}")
            raise

    def train_random_forest(self, X_train, y_train):
        """Train RandomForest with memory optimization."""
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                max_features='sqrt'
            )
            model.fit(X_train, y_train)
            self.models['RandomForest'] = model
            self.logger.info("RandomForest trained")
            gc.collect()
            return model

        except Exception as e:
            self.logger.error(f"Error training RandomForest: {str(e)}")
            raise

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with memory optimization."""
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='hist',
                device='cpu',
                eval_metric='logloss',
                scale_pos_weight=((y_train == 0).sum() / (y_train == 1).sum())
            )
            model.fit(X_train, y_train, verbose=False)
            self.models['XGBoost'] = model
            self.logger.info("XGBoost trained")
            gc.collect()
            return model

        except Exception as e:
            self.logger.error(f"Error training XGBoost: {str(e)}")
            raise

    def evaluate_model(self, model_name: str, model, threshold: float = 0.5) -> dict:
        """Evaluate model on test set."""
        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            metrics = {
                'f1': f1_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'train_f1': f1_score(self.y_train, (model.predict_proba(self.X_train)[:, 1] >= threshold).astype(int)),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'y_pred_proba': y_pred_proba
            }

            # Calculate PR-AUC
            precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, y_pred_proba)
            metrics['pr_auc'] = auc(recall_vals, precision_vals)

            self.results[model_name] = metrics
            self.logger.info(f"{model_name} - F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise

    def train_all_models(self, apply_smote: bool = True):
        """Train all models."""
        try:
            X_train = self.X_train
            y_train = self.y_train

            if apply_smote:
                X_train, y_train = self.apply_smote()

            self.train_logistic_regression(X_train, y_train)
            self.train_random_forest(X_train, y_train)
            self.train_xgboost(X_train, y_train)

            for model_name, model in self.models.items():
                self.evaluate_model(model_name, model)

            self.logger.info("All models trained and evaluated")
            return self.results

        except Exception as e:
            self.logger.error(f"Error in train_all_models: {str(e)}")
            raise

    def get_best_model(self) -> tuple:
        """Get best model based on F1 score."""
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        return best_model_name, self.models[best_model_name], self.results[best_model_name]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ClassificationTrainer module loaded successfully")
