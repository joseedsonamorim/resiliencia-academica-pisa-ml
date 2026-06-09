from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import get_sample_weights, get_target_series, select_feature_columns
from src.utils.seed import set_global_seed


def _scaled_classifier(clf: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _tree_classifier(clf: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    )


def _build_estimators(seed: int) -> dict[str, object]:
    estimators: dict[str, object] = {
        "logistic_regression": _scaled_classifier(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        ),
        "svm_rbf": _scaled_classifier(
            SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
        ),
        "knn_distance": _scaled_classifier(KNeighborsClassifier(n_neighbors=21, weights="distance")),
        "gaussian_nb": _tree_classifier(GaussianNB()),
        "random_forest": _tree_classifier(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "extra_trees": _tree_classifier(
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": _tree_classifier(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=seed)
        ),
        "hist_gradient_boosting": _tree_classifier(
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=seed,
            )
        ),
        "adaboost": _tree_classifier(
            AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=seed)
        ),
    }

    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = _tree_classifier(
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm"] = _tree_classifier(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        estimators["catboost"] = _tree_classifier(
            CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                verbose=0,
                random_seed=seed,
                auto_class_weights="Balanced",
            )
        )
    except ImportError:
        pass

    return estimators


def _ignore_model_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names.*",
        category=UserWarning,
    )


def _search_spaces() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic_regression": {
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "svm_rbf": {
            "clf__C": [0.3, 0.7, 1.0, 2.0, 5.0],
            "clf__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
        },
        "knn_distance": {
            "clf__n_neighbors": [7, 11, 15, 21, 31, 41],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "random_forest": {
            "clf__n_estimators": [250, 400, 600],
            "clf__max_depth": [None, 6, 10, 16, 24],
            "clf__max_features": ["sqrt", "log2", 0.5],
            "clf__min_samples_leaf": [1, 2, 5, 10],
        },
        "extra_trees": {
            "clf__n_estimators": [300, 500, 700],
            "clf__max_depth": [None, 6, 10, 16, 24],
            "clf__max_features": ["sqrt", "log2", 0.5],
            "clf__min_samples_leaf": [1, 2, 5, 10],
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200, 350],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__subsample": [0.75, 0.9, 1.0],
        },
        "hist_gradient_boosting": {
            "clf__max_iter": [150, 250, 400],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        },
        "adaboost": {
            "clf__n_estimators": [100, 200, 350, 500],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
        },
        "xgboost": {
            "clf__n_estimators": [200, 350, 500],
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__subsample": [0.75, 0.9, 1.0],
            "clf__colsample_bytree": [0.75, 0.9, 1.0],
        },
        "lightgbm": {
            "clf__n_estimators": [200, 350, 500],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__num_leaves": [15, 31, 63],
            "clf__min_child_samples": [10, 20, 40],
        },
        "catboost": {
            "clf__iterations": [200, 350, 500],
            "clf__depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
        },
    }


def _final_estimator(estimator: object) -> object:
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("clf", estimator)  # type: ignore[attr-defined]
    return estimator


def _fit_params(estimator: object, weights: pd.Series | None) -> dict[str, np.ndarray]:
    if weights is None:
        return {}
    final = _final_estimator(estimator)
    try:
        signature = inspect.signature(final.fit)
    except (TypeError, ValueError):
        return {}
    if "sample_weight" not in signature.parameters:
        return {}
    key = "clf__sample_weight" if hasattr(estimator, "named_steps") else "sample_weight"
    return {key: weights.to_numpy()}


def _fit_estimator(estimator: object, x: pd.DataFrame, y: pd.Series, weights: pd.Series | None) -> object:
    fitted = clone(estimator)
    params = _fit_params(fitted, weights)
    with warnings.catch_warnings():
        _ignore_model_warnings()
        try:
            fitted.fit(x, y, **params)
        except TypeError:
            fitted.fit(x, y)
    return fitted


def _positive_proba(estimator: object, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        _ignore_model_warnings()
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(x)[:, 1]
        scores = estimator.decision_function(x)  # type: ignore[attr-defined]
    return 1.0 / (1.0 + np.exp(-scores))


def _classification_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
    }
    out["roc_auc"] = roc_auc_score(y_true, proba) if y_true.nunique() > 1 else np.nan
    return {k: float(v) for k, v in out.items()}


def _cv_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    margin = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def _evaluate_cv(
    estimator: object,
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    cv: RepeatedStratifiedKFold,
) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    params = _fit_params(estimator, weights)
    with warnings.catch_warnings():
        _ignore_model_warnings()
        try:
            scores = cross_validate(
                estimator,
                x,
                y,
                cv=cv,
                scoring=scoring,
                fit_params=params or None,
                n_jobs=1,
                error_score="raise",
            )
        except TypeError:
            scores = cross_validate(
                estimator,
                x,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

    row: dict[str, float] = {"n_cv_splits": float(len(scores["test_roc_auc"]))}
    for metric in scoring:
        summary = _cv_summary(scores[f"test_{metric}"])
        row[f"cv_{metric}_mean"] = summary["mean"]
        row[f"cv_{metric}_std"] = summary["std"]
        row[f"cv_{metric}_ci_low"] = summary["ci_low"]
        row[f"cv_{metric}_ci_high"] = summary["ci_high"]
    return row


def _evaluate_candidate(
    *,
    name: str,
    variant: str,
    estimator: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    w_train: pd.Series | None,
    cv: RepeatedStratifiedKFold,
    target_key: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], object, np.ndarray]:
    cv_row = _evaluate_cv(estimator, x_train, y_train, w_train, cv)
    fitted = _fit_estimator(estimator, x_train, y_train, w_train)
    proba = _positive_proba(fitted, x_test)
    holdout = _classification_metrics(y_test, proba, threshold=0.5)
    row: dict[str, Any] = {
        "model": name,
        "variant": variant,
        "target": target_key,
        "selection_metric": "cv_roc_auc_mean",
        "best_params": json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
    }
    row.update(cv_row)
    for metric, value in holdout.items():
        row[f"holdout_{metric}"] = value
    return row, fitted, proba


def _manual_oof_proba(
    estimator: object,
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series | None,
    seed: int,
    cv_folds: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(x, y):
        fold_weights = weights.iloc[train_idx] if weights is not None else None
        fitted = _fit_estimator(estimator, x.iloc[train_idx], y.iloc[train_idx], fold_weights)
        proba[valid_idx] = _positive_proba(fitted, x.iloc[valid_idx])
    return proba


def _threshold_table(y: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = _classification_metrics(y, proba, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows).sort_values(["f1", "balanced_accuracy"], ascending=False)


def _bootstrap_best(
    y: pd.Series,
    proba: np.ndarray,
    threshold: float,
    seed: int,
    n_boot: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_values: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "f1": [],
        "balanced_accuracy": [],
        "brier": [],
    }
    y_arr = y.to_numpy()
    n = len(y_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = pd.Series(y_arr[idx])
        p_b = proba[idx]
        if y_b.nunique() < 2:
            continue
        metrics = _classification_metrics(y_b, p_b, threshold)
        for metric in metric_values:
            metric_values[metric].append(metrics[metric])

    rows = []
    for metric, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(arr)) if len(arr) else np.nan,
                "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else np.nan,
                "ci_low": float(np.nanpercentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.nanpercentile(arr, 97.5)) if len(arr) else np.nan,
                "n_bootstrap": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


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
    if weights is not None:
        weights = weights.loc[x.index]

    modeling_cfg = cfg.get("modeling", {})
    test_size = float(modeling_cfg.get("test_size", 0.2))
    cv_folds = int(modeling_cfg.get("cv_folds", 5))
    cv_repeats = int(modeling_cfg.get("cv_repeats", 3))
    search_enabled = bool(modeling_cfg.get("hyperparameter_search", True))
    search_top_k = int(modeling_cfg.get("search_top_k", 3))
    search_n_iter = int(modeling_cfg.get("search_n_iter", 8))
    n_boot = int(modeling_cfg.get("n_bootstrap_holdout", 500))

    split_kw: dict = {"test_size": test_size, "random_state": seed, "stratify": y}
    if weights is not None:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            x, y, weights, **split_kw
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kw)
        w_train = w_test = None

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=seed,
    )
    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    estimators = _build_estimators(seed)
    search_spaces = _search_spaces()

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, object] = {}
    holdout_probas: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []

    for name, estimator in estimators.items():
        try:
            row, fitted, proba = _evaluate_candidate(
                name=name,
                variant="screening",
                estimator=estimator,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                w_train=w_train,
                cv=cv,
                target_key=target_key,
            )
            rows.append(row)
            fitted_models[name] = fitted
            holdout_probas[name] = proba
        except Exception as exc:
            failures.append({"model": name, "variant": "screening", "error": str(exc)})

    if not rows:
        raise RuntimeError("Nenhum modelo conseguiu ser ajustado.")

    screening_df = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    top_for_search = [
        str(name)
        for name in screening_df["model"].head(search_top_k).tolist()
        if name in search_spaces
    ]

    if search_enabled:
        for name in top_for_search:
            estimator = estimators[name]
            try:
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=search_spaces[name],
                    n_iter=search_n_iter,
                    scoring="roc_auc",
                    cv=inner_cv,
                    n_jobs=1,
                    random_state=seed,
                    error_score=np.nan,
                    refit=True,
                )
                search_params = _fit_params(search.estimator, w_train)
                with warnings.catch_warnings():
                    _ignore_model_warnings()
                    try:
                        search.fit(x_train, y_train, **search_params)
                    except TypeError:
                        search.fit(x_train, y_train)

                tuned_name = f"{name}_tuned"
                row, fitted, proba = _evaluate_candidate(
                    name=tuned_name,
                    variant="tuned",
                    estimator=search.best_estimator_,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    w_train=w_train,
                    cv=cv,
                    target_key=target_key,
                    params=search.best_params_,
                )
                row["search_best_cv_roc_auc"] = float(search.best_score_)
                rows.append(row)
                fitted_models[tuned_name] = fitted
                holdout_probas[tuned_name] = proba
            except Exception as exc:
                failures.append({"model": name, "variant": "tuned", "error": str(exc)})

    metrics_df = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc_mean", "holdout_roc_auc"], ascending=False
    )
    metrics_df.to_csv(out_tables / "modeling_metrics.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(out_tables / "modeling_failures.csv", index=False)

    best_name = str(metrics_df.iloc[0]["model"])
    best_cv_auc = float(metrics_df.iloc[0]["cv_roc_auc_mean"])
    best_model = fitted_models[best_name]
    best_proba = holdout_probas[best_name]

    oof_proba = _manual_oof_proba(
        best_model,
        x_train,
        y_train,
        w_train,
        seed=seed,
        cv_folds=cv_folds,
    )
    threshold_df = _threshold_table(y_train.reset_index(drop=True), oof_proba)
    threshold_df.to_csv(out_tables / "modeling_thresholds.csv", index=False)
    best_threshold = float(threshold_df.iloc[0]["threshold"])
    optimized_holdout = _classification_metrics(y_test, best_proba, best_threshold)
    for metric, value in optimized_holdout.items():
        metrics_df.loc[metrics_df["model"] == best_name, f"holdout_opt_{metric}"] = value
    metrics_df.to_csv(out_tables / "modeling_metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "row_id": x_test.index,
            "y_true": y_test.to_numpy(),
            "sample_weight": w_test.to_numpy() if w_test is not None else np.nan,
            "proba_resilient": best_proba,
            "pred_0_50": (best_proba >= 0.5).astype(int),
            "pred_optimized": (best_proba >= best_threshold).astype(int),
            "threshold_optimized": best_threshold,
        }
    )
    predictions_df.to_csv(out_tables / "modeling_holdout_predictions.csv", index=False)

    bootstrap_df = _bootstrap_best(
        y_test.reset_index(drop=True),
        best_proba,
        threshold=best_threshold,
        seed=seed,
        n_boot=n_boot,
    )
    bootstrap_df.to_csv(out_tables / "modeling_best_bootstrap.csv", index=False)

    feature_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "missing_ratio": x[feature_cols].isna().mean().to_numpy(),
            "variance": x[feature_cols].var(ddof=0).to_numpy(),
        }
    ).sort_values("variance", ascending=False)
    feature_df.to_csv(out_tables / "modeling_feature_set.csv", index=False)

    meta = {
        "target_key": target_key,
        "best_model": best_name,
        "selection_metric": "cv_roc_auc_mean",
        "best_cv_roc_auc_mean": best_cv_auc,
        "best_threshold": best_threshold,
        "feature_cols": feature_cols,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "dataset": csv_path.name,
        "used_sample_weights": weights is not None,
    }
    (models_dir / "modeling_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    joblib.dump(best_model, models_dir / "best_model.joblib")

    best_holdout_auc = float(metrics_df.loc[metrics_df["model"] == best_name, "holdout_roc_auc"].iloc[0])
    best_holdout_f1 = float(metrics_df.loc[metrics_df["model"] == best_name, "holdout_f1"].iloc[0])
    best_holdout_opt_f1 = float(optimized_holdout["f1"])
    top_table = metrics_df[
        [
            "model",
            "variant",
            "cv_roc_auc_mean",
            "cv_roc_auc_ci_low",
            "cv_roc_auc_ci_high",
            "cv_average_precision_mean",
            "cv_f1_mean",
            "holdout_roc_auc",
            "holdout_average_precision",
            "holdout_f1",
            "holdout_precision",
            "holdout_recall",
        ]
    ].head(15)

    md = [
        "# Modelagem preditiva (Fase 8)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target analisado: **{target_key}**\n",
        f"- Amostra modelada: {len(x):,} estudantes; treino={len(x_train):,}; teste={len(x_test):,}\n",
        f"- Features elegíveis após filtros de vazamento/missingness: {len(feature_cols)}\n",
        f"- Validação: holdout estratificado + CV repetida ({cv_folds} folds x {cv_repeats} repetições) no treino\n",
        "- Critério primário de seleção: média de ROC-AUC na validação cruzada, não o desempenho do holdout\n",
        f"- Melhor modelo: **{best_name}** (CV ROC-AUC={best_cv_auc:.4f}; holdout ROC-AUC={best_holdout_auc:.4f})\n",
        f"- F1 no holdout: {best_holdout_f1:.4f} com limiar 0.50; {best_holdout_opt_f1:.4f} com limiar otimizado={best_threshold:.3f}\n",
        f"- Pesos amostrais PISA usados quando suportados pelo estimador: {'sim' if weights is not None else 'não encontrados'}\n\n",
        "## Ranking dos modelos\n\n",
        df_to_markdown(top_table),
        "\n\n## Incerteza do melhor modelo no holdout\n\n",
        df_to_markdown(bootstrap_df),
        "\n\n## Observação metodológica\n\n",
        "Os resultados devem ser lidos como evidência preditiva para a definição operacional de resiliência selecionada. "
        "Para submissão em periódico de alto estrato, recomenda-se reportar a definição do target, o controle de vazamento, "
        "a prevalência da classe positiva, os intervalos de confiança e análises de sensibilidade entre targets A/B/C/D.\n",
    ]
    if failures:
        md.extend(
            [
                "\n## Modelos não concluídos\n\n",
                df_to_markdown(pd.DataFrame(failures)),
                "\n",
            ]
        )
    (out_reports / "modeling_report.md").write_text("".join(md), encoding="utf-8")

    prevalence = float(y.mean())
    summary_md = [
        "# Relatório resumido\n\n",
        "## Objetivo\n",
        "Identificar estudantes resilientes no PISA Brasil e comparar métodos de aprendizado de máquina "
        "para selecionar o modelo com melhor evidência preditiva sob validação rigorosa.\n\n",
        "## Dados e target\n",
        f"Foram analisados {len(x):,} estudantes do arquivo `{csv_path.name}`. "
        f"O target ativo foi **{target_key}**, com {int(y.sum())} casos positivos "
        f"({prevalence:.2%} da amostra). As features passaram por filtros de missingness, "
        "baixa variância, pesos/IDs e variáveis com risco de vazamento.\n\n",
        "## Método\n",
        f"Foram comparados {metrics_df.shape[0]} candidatos/variantes de modelos, incluindo regressão logística, "
        "SVM, KNN, Naive Bayes, Random Forest, Extra Trees, boosting e modelos opcionais instalados "
        "(XGBoost/LightGBM/CatBoost quando disponíveis). A seleção usou ROC-AUC médio em CV repetida "
        f"({cv_folds} folds x {cv_repeats} repetições) no treino; o holdout estratificado foi reservado "
        "para avaliação final. O melhor modelo teve incerteza estimada por bootstrap.\n\n",
        "## Resultado principal\n",
        f"O melhor modelo foi **{best_name}**, com ROC-AUC médio de CV={best_cv_auc:.4f} "
        f"e ROC-AUC no holdout={best_holdout_auc:.4f}. Com limiar padrão 0.50, o F1 no holdout foi "
        f"{best_holdout_f1:.4f}; com limiar otimizado em validação interna ({best_threshold:.3f}), "
        f"o F1 subiu para {best_holdout_opt_f1:.4f}. O bootstrap do holdout estimou ROC-AUC médio "
        f"{bootstrap_df.loc[bootstrap_df['metric'] == 'roc_auc', 'mean'].iloc[0]:.4f} "
        f"(IC95% {bootstrap_df.loc[bootstrap_df['metric'] == 'roc_auc', 'ci_low'].iloc[0]:.4f}-"
        f"{bootstrap_df.loc[bootstrap_df['metric'] == 'roc_auc', 'ci_high'].iloc[0]:.4f}).\n\n",
        "## Leitura científica\n",
        "O desempenho discriminativo é alto, mas a classe resiliente é rara; por isso, precisão, recall, "
        "average precision, calibração, fairness e estabilidade entre targets devem acompanhar o ROC-AUC. "
        "Para submissão em periódico de alto impacto, recomenda-se explicitar a definição teórica de resiliência, "
        "o desenho amostral do PISA, os pesos, o controle de vazamento e análises de sensibilidade entre targets.\n",
    ]
    (out_reports / "relatorio_resumido.md").write_text("".join(summary_md), encoding="utf-8")
