from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import select_feature_columns
from src.utils.seed import set_global_seed


def _prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imp.fit_transform(x))
    return scaled


def _cluster_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(labels) < len(uniq) + 1:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan"), "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
    }


def run_clusterer(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "clusterer"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    n_clusters = int(cfg.get("clustering", {}).get("n_clusters", 4))
    feature_cols = select_feature_columns(df, cfg, max_features=int(cfg.get("clustering", {}).get("max_features", 50)))
    if len(feature_cols) < 3:
        raise ValueError("Poucas features disponíveis para clusterização (mínimo 3).")

    x = _prepare_matrix(df, feature_cols)
    rows: list[dict] = []
    label_frames: dict[str, pd.Series] = {}

    algorithms = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=int(cfg.get("random_seed", 42)), n_init=10),
        "hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "gmm": GaussianMixture(n_components=n_clusters, random_state=int(cfg.get("random_seed", 42))),
    }

    for name, model in algorithms.items():
        if name == "gmm":
            labels = model.fit_predict(x)
        else:
            labels = model.fit_predict(x)
        metrics = _cluster_metrics(x, labels)
        rows.append({"algorithm": name, "n_clusters": n_clusters, **metrics})
        label_frames[f"cluster_{name}"] = pd.Series(labels, index=df.index, name=f"cluster_{name}")

    metrics_df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    metrics_df.to_csv(out_tables / "clusterer_metrics.csv", index=False)

    labels_df = pd.DataFrame(label_frames)
    labels_df.to_csv(out_tables / "clusterer_labels.csv", index=False)

    best = metrics_df.iloc[0]["algorithm"] if not metrics_df.empty else "kmeans"
    md = [
        "# Clusterer (Fase 7)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Features usadas: {len(feature_cols)}\n",
        f"- k / componentes: {n_clusters}\n",
        f"- Melhor algoritmo (silhouette): **{best}**\n\n",
        "## Métricas\n\n",
        df_to_markdown(metrics_df),
        "\n\n## Artefatos\n\n",
        "- `outputs/tables/clusterer_metrics.csv`\n",
        "- `outputs/tables/clusterer_labels.csv`\n",
    ]
    (out_reports / "clusterer_report.md").write_text("".join(md), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.bar(metrics_df["algorithm"], metrics_df["silhouette"])
        plt.title("Silhouette por algoritmo")
        plt.ylabel("silhouette")
        plt.tight_layout()
        plt.savefig(out_figures / "clusterer_silhouette.png", dpi=200)
        plt.close()
    except Exception:
        pass
