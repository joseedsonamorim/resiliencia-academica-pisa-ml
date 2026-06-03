from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.pipeline_data import get_sample_weights, get_target_series, select_feature_columns
from src.utils.seed import set_global_seed


def _build_estimators(seed: int) -> dict[str, object]:
    estimators: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        estimators["catboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=200,
                        depth=4,
                        verbose=0,
                        random_seed=seed,
                        auto_class_weights="Balanced",
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return estimators


def run_modeling(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))
    seed = int(cfg.get("random_seed", 42))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    y, target_key = get_target_series(df, cfg)
    feature_cols = select_feature_columns(df, cfg)
    if len(feature_cols) < 3:
        raise ValueError("Poucas features para modelagem.")

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask = y.notna()
    x, y = x.loc[mask], y.loc[mask].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Target {target_key} não tem duas classes.")

    weights = get_sample_weights(df.loc[mask], cfg)
    test_size = float(cfg.get("modeling", {}).get("test_size", 0.2))
    cv_folds = int(cfg.get("modeling", {}).get("cv_folds", 5))

    split_kw: dict = {"test_size": test_size, "random_state": seed, "stratify": y}
    if weights is not None:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            x, y, weights.loc[mask], **split_kw
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kw)
        w_train = w_test = None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = ["accuracy", "f1", "roc_auc"]
    rows: list[dict] = []
    best_name = None
    best_auc = -1.0
    best_model = None

    for name, est in _build_estimators(seed).items():
        fit_params = {}
        if w_train is not None and hasattr(est, "named_steps"):
            last = est.steps[-1][1]
            if hasattr(last, "fit") and "sample_weight" in last.fit.__code__.co_varnames:
                fit_params = {f"clf__sample_weight": w_train.values}

        try:
            if fit_params:
                est.fit(x_train, y_train, **fit_params)
            else:
                est.fit(x_train, y_train)
        except TypeError:
            est.fit(x_train, y_train)

        proba = est.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "model": name,
            "target": target_key,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        }

        cv_scores = cross_validate(
            est,
            x,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row["cv_roc_auc_mean"] = float(np.nanmean(cv_scores["test_roc_auc"]))
        row["cv_f1_mean"] = float(np.nanmean(cv_scores["test_f1"]))
        rows.append(row)

        if row["roc_auc"] > best_auc:
            best_auc = row["roc_auc"]
            best_name = name
            best_model = est

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(out_tables / "modeling_metrics.csv", index=False)

    meta = {
        "target_key": target_key,
        "best_model": best_name,
        "feature_cols": feature_cols,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "dataset": csv_path.name,
    }
    (models_dir / "modeling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if best_model is not None:
        joblib.dump(best_model, models_dir / "best_model.joblib")

    md = [
        "# Modeling (Fase 8)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target: **{target_key}**\n",
        f"- Features: {len(feature_cols)}\n",
        f"- Melhor modelo (ROC-AUC holdout): **{best_name}** ({best_auc:.4f})\n\n",
        metrics_df.to_markdown(index=False),
        "\n",
    ]
    (out_reports / "modeling_report.md").write_text("".join(md), encoding="utf-8")
