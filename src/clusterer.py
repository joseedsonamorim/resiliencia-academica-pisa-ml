from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, Birch, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import load_leakage_exclude, select_feature_columns
from src.utils.seed import set_global_seed


DEFAULT_FEATURE_GROUPS: dict[str, list[str]] = {
    "accesso_tecnologico": ["ict", "digital", "computer", "internet", "device"],
    "recursos_domesticos": ["homepos", "home", "book", "desk", "quiet", "study"],
    "apoio_familiar": ["parent", "mother", "father", "family", "support", "encour"],
    "clima_escolar": ["school", "teacher", "class", "bully", "safety", "discip"],
    "pertencimento": ["belong", "included", "accepted", "satisf"],
    "motivacao": ["motiv", "interest", "effort", "goal", "persist"],
    "autoeficacia": ["self", "effic", "confid", "capab"],
    "ansiedade": ["anx", "worry", "nerv", "stress"],
    "indicadores_processuais": ["process", "timing", "time", "attempt", "engag"],
    "engajamento_tarefas_criativas": ["^CR590", "CR590Q"],
}

PALETTE = [
    "#1d4ed8",
    "#047857",
    "#b45309",
    "#be123c",
    "#7c3aed",
    "#0f766e",
    "#334155",
    "#ea580c",
    "#2563eb",
    "#10b981",
]


@dataclass
class ModelCandidate:
    algorithm: str
    n_clusters: int
    model: object | None
    labels: np.ndarray
    quality: dict[str, float]
    notes: str = ""


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    x = df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
    return x


def _prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x_raw = _numeric_frame(df, feature_cols)
    imputer = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imputer.fit_transform(x_raw), columns=feature_cols, index=df.index)
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x_imp), columns=feature_cols, index=df.index)
    return x_scaled.to_numpy(dtype=float), x_imp, x_scaled


def _feature_groups_from_config(cfg: dict) -> dict[str, list[str]]:
    user_groups = cfg.get("clustering", {}).get("feature_groups") or {}
    out: dict[str, list[str]] = {}
    for group, patterns in DEFAULT_FEATURE_GROUPS.items():
        out[group] = [str(p) for p in patterns]
    for group, patterns in user_groups.items():
        if isinstance(patterns, dict):
            patterns = patterns.get("include", [])
        out[str(group)] = [str(p) for p in patterns or []]
    return out


def _is_excluded(col: str, excluded: set[str]) -> bool:
    lc = col.lower()
    hard = {"w_fstuwt", "escs", "grupo_escs", "crt_score", "creative_resilience", "cntstuid"}
    if col in excluded or lc in excluded or lc in hard:
        return True
    if lc.startswith("target_") or lc.startswith("target"):
        return True
    if "weight" in lc or lc.startswith("w_"):
        return True
    if "id" in lc and lc not in {"ictid"}:
        return True
    return False


def _select_theoretical_features(df: pd.DataFrame, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    excluded = {str(v) for v in load_leakage_exclude(cfg)}
    groups = _feature_groups_from_config(cfg)
    max_features = int(cfg.get("clustering", {}).get("max_features", 80))

    rows: list[dict[str, object]] = []
    selected: list[str] = []
    matched: set[str] = set()

    patterns_by_group = {
        group: [re.compile(p, re.I) for p in pats] for group, pats in groups.items()
    }

    for col in df.columns:
        if _is_excluded(col, excluded):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        lc = col.lower()
        hits: list[str] = []
        for group, patterns in patterns_by_group.items():
            if any(p.search(lc) for p in patterns):
                hits.append(group)

        if not hits:
            continue

        rows.append(
            {
                "feature": col,
                "groups": ";".join(hits),
                "missing_ratio": float(df[col].isna().mean()),
                "variance": float(pd.to_numeric(df[col], errors="coerce").var(ddof=0)),
            }
        )
        selected.append(col)
        matched.add(col)

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        fallback = select_feature_columns(df, cfg, max_features=max_features)
        catalog = pd.DataFrame(
            {
                "feature": fallback,
                "groups": "fallback",
                "missing_ratio": [float(df[c].isna().mean()) for c in fallback],
                "variance": [float(pd.to_numeric(df[c], errors="coerce").var(ddof=0)) for c in fallback],
            }
        )
        return fallback[:max_features], catalog

    if len(selected) < 8:
        fallback = select_feature_columns(df, cfg, max_features=max_features)
        for col in fallback:
            if col not in matched and not _is_excluded(col, excluded):
                selected.append(col)
                matched.add(col)
                catalog = pd.concat(
                    [
                        catalog,
                        pd.DataFrame(
                            {
                                "feature": [col],
                                "groups": ["fallback"],
                                "missing_ratio": [float(df[col].isna().mean())],
                                "variance": [float(pd.to_numeric(df[col], errors="coerce").var(ddof=0))],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            if len(selected) >= max_features:
                break

    catalog = catalog.drop_duplicates(subset=["feature"]).sort_values(
        ["groups", "missing_ratio", "variance"], ascending=[True, True, False]
    )
    selected = [c for c in catalog["feature"].tolist() if c in df.columns]
    return selected[:max_features], catalog


def _drop_collinear_features(
    x_scaled: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> list[str]:
    if len(feature_cols) <= 2:
        return feature_cols

    corr = x_scaled[feature_cols].corr().abs().fillna(0)
    ranking = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "missing_ratio": [float(x_scaled[c].isna().mean()) for c in feature_cols],
                "variance": [float(x_scaled[c].var(ddof=0)) for c in feature_cols],
            }
        )
        .sort_values(["missing_ratio", "variance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    keep: list[str] = []
    removed: set[str] = set()
    for feature in ranking["feature"]:
        if feature in removed:
            continue
        keep.append(feature)
        for other in feature_cols:
            if other == feature or other in removed:
                continue
            if corr.loc[feature, other] >= threshold:
                removed.add(other)
    return keep


def _fit_pca(x_scaled: np.ndarray, cfg: dict) -> tuple[np.ndarray, pd.DataFrame, int]:
    target_variance = float(cfg.get("clustering", {}).get("pca_variance_target", 0.85))
    n_components = min(x_scaled.shape[0] - 1, x_scaled.shape[1])
    pca_full = PCA(n_components=n_components, random_state=int(cfg.get("random_seed", 42)))
    _ = pca_full.fit_transform(x_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, target_variance) + 1)
    keep = max(2, min(keep, x_scaled.shape[1], x_scaled.shape[0] - 1))
    pca = PCA(n_components=keep, random_state=int(cfg.get("random_seed", 42)))
    x_pca = pca.fit_transform(x_scaled)
    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return x_pca, explained, keep


def _palette_for_labels(labels: np.ndarray) -> dict[int, str]:
    uniq = [int(v) for v in sorted(pd.unique(pd.Series(labels)).tolist())]
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}


def _line_chart(
    path: Path,
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    threshold: float | None = None,
) -> None:
    width, height = 1400, 900
    left, right, top, bottom = 110, 70, 90, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 30), title, fill="#0f172a", font=_font(28))
    draw.text((left, height - 55), x_label, fill="#334155", font=_font(16))
    draw.text((22, top + 120), y_label, fill="#334155", font=_font(16))
    draw.line((left, top, left, height - bottom), fill="#64748b", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#64748b", width=2)

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if len(xs) == 0:
        img.save(path)
        return

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(min(0.0, ys.min())), float(max(ys.max(), 1e-9))
    if threshold is not None:
        y_max = max(y_max, float(threshold))

    def sx(val: float) -> float:
        if x_max == x_min:
            return left
        return left + (val - x_min) / (x_max - x_min) * (width - left - right)

    def sy(val: float) -> float:
        if y_max == y_min:
            return height - bottom
        return height - bottom - (val - y_min) / (y_max - y_min) * (height - top - bottom)

    pts = [(sx(float(a)), sy(float(b))) for a, b in zip(xs, ys, strict=False)]
    if len(pts) > 1:
        draw.line(pts, fill="#1d4ed8", width=4)
    for px, py in pts:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#1d4ed8", outline="#1d4ed8")

    if threshold is not None:
        ty = sy(threshold)
        draw.line((left, ty, width - right, ty), fill="#be123c", width=2)
        draw.text((width - 250, ty - 22), f"alvo {threshold:.2f}", fill="#be123c", font=_font(16))

    img.save(path)


def _scatter_chart(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    title: str,
    subtitle: str = "",
) -> None:
    width, height = 1400, 1000
    left, right, top, bottom = 120, 60, 95, 120
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 32), title, fill="#0f172a", font=_font(30))
    if subtitle:
        draw.text((left, 78), subtitle, fill="#475569", font=_font(17))
    draw.line((left, top, left, height - bottom), fill="#64748b", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#64748b", width=2)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    colors = _palette_for_labels(labels)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    pad_x = (x_max - x_min) * 0.04 or 1.0
    pad_y = (y_max - y_min) * 0.04 or 1.0
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    def sx(val: float) -> float:
        return left + (val - x_min) / (x_max - x_min) * (width - left - right)

    def sy(val: float) -> float:
        return height - bottom - (val - y_min) / (y_max - y_min) * (height - top - bottom)

    for xi, yi, lab in zip(x, y, labels, strict=False):
        px, py = sx(float(xi)), sy(float(yi))
        color = colors.get(int(lab), "#334155")
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=color)

    legend_y = 112
    for lab, color in list(colors.items())[:10]:
        draw.rectangle((width - 270, legend_y, width - 250, legend_y + 18), fill=color, outline=color)
        draw.text((width - 242, legend_y - 1), f"cluster {lab}", fill="#0f172a", font=_font(15))
        legend_y += 24

    img.save(path)


def _heatmap_chart(
    path: Path,
    matrix: pd.DataFrame,
    title: str,
    cell_size: int = 28,
    font_size: int = 14,
    center_zero: bool = True,
) -> None:
    rows = list(matrix.index.astype(str))
    cols = list(matrix.columns.astype(str))
    width = max(900, 180 + cell_size * max(1, len(cols)))
    height = max(420, 140 + cell_size * max(1, len(rows)))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 25), title, fill="#0f172a", font=_font(26))

    top = 90
    left = 140
    data = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data))) if center_zero else float(np.nanmax(data))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

    def color_for(value: float) -> tuple[int, int, int]:
        if not np.isfinite(value):
            return (226, 232, 240)
        if center_zero:
            scale = max(-1.0, min(1.0, value / vmax))
            if scale >= 0:
                r = int(255 - (1 - scale) * 224)
                g = int(248 - (1 - scale) * 120)
                b = int(240 - (1 - scale) * 80)
                return (r, g, b)
            scale = abs(scale)
            r = int(255 - (1 - scale) * 120)
            g = int(241 - (1 - scale) * 140)
            b = int(242 - (1 - scale) * 40)
            return (r, g, b)

        scale = max(0.0, min(1.0, value / vmax))
        c = int(255 - scale * 130)
        return (c, c, 255)

    for j, col in enumerate(cols):
        draw.text((left + j * cell_size + 2, top - 24), col[:10], fill="#334155", font=_font(font_size))
    for i, row in enumerate(rows):
        draw.text((20, top + i * cell_size + 5), row[:12], fill="#334155", font=_font(font_size))
        for j, value in enumerate(data[i]):
            x0 = left + j * cell_size
            y0 = top + i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=color_for(float(value)), outline="#ffffff")
            text = f"{value:.2f}" if np.isfinite(value) else "NA"
            draw.text((x0 + 2, y0 + 5), text[:5], fill="#0f172a", font=_font(11))

    img.save(path)


def _radar_chart(
    path: Path,
    profiles: pd.DataFrame,
    feature_order: list[str],
    title: str,
) -> None:
    width, height = 1200, 1200
    center = (width // 2, height // 2 + 20)
    radius = 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), title, fill="#0f172a", font=_font(30))
    draw.text((70, 86), "Valores padronizados por cluster", fill="#475569", font=_font(17))

    n = len(feature_order)
    if n == 0:
        img.save(path)
        return

    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    max_val = float(np.nanmax(np.abs(profiles[feature_order].to_numpy(dtype=float)))) if feature_order else 1.0
    max_val = max(max_val, 1.0)

    for ring in [0.25, 0.5, 0.75, 1.0]:
        r = radius * ring
        pts = [(center[0] + math.cos(a) * r, center[1] + math.sin(a) * r) for a in angles]
        draw.polygon(pts, outline="#e2e8f0")

    for a, feat in zip(angles, feature_order, strict=False):
        x2 = center[0] + math.cos(a) * radius
        y2 = center[1] + math.sin(a) * radius
        draw.line((center[0], center[1], x2, y2), fill="#cbd5e1", width=2)
        lx = center[0] + math.cos(a) * (radius + 35)
        ly = center[1] + math.sin(a) * (radius + 35)
        draw.text((lx - 35, ly - 8), feat[:16], fill="#334155", font=_font(15))

    for idx, (_, row) in enumerate(profiles.iterrows()):
        color = PALETTE[idx % len(PALETTE)]
        vals = row[feature_order].to_numpy(dtype=float)
        pts = []
        for val, a in zip(vals, angles, strict=False):
            r = radius * (0.5 + 0.35 * np.tanh(float(val) / max_val))
            pts.append((center[0] + math.cos(a) * r, center[1] + math.sin(a) * r))
        if pts:
            pts.append(pts[0])
            draw.line(pts, fill=color, width=4)
            draw.polygon(pts[:-1], outline=color)
        draw.text((80, 130 + idx * 24), f"cluster {row['cluster_id']}", fill=color, font=_font(16))

    img.save(path)


def _dendrogram_chart(path: Path, x: np.ndarray, labels: np.ndarray, title: str) -> None:
    width, height = 1500, 900
    left, right, top, bottom = 80, 80, 100, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 30), title, fill="#0f172a", font=_font(28))
    draw.line((left, top, left, height - bottom), fill="#64748b", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#64748b", width=2)

    try:
        link = linkage(pdist(x), method="ward")
        dendro = dendrogram(link, labels=[str(int(v)) for v in labels], no_plot=True)
    except Exception:
        draw.text((left, top + 40), "Dendrograma indisponível para este conjunto.", fill="#334155", font=_font(18))
        img.save(path)
        return

    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    max_x = max(max(xs) for xs in icoord)
    min_x = min(min(xs) for xs in icoord)
    max_y = max(max(ys) for ys in dcoord)
    min_y = min(min(ys) for ys in dcoord)
    if max_y == min_y:
        max_y = min_y + 1.0

    def sx(v: float) -> float:
        return left + (v - min_x) / (max_x - min_x) * (width - left - right)

    def sy(v: float) -> float:
        return height - bottom - (v - min_y) / (max_y - min_y) * (height - top - bottom)

    for xs, ys in zip(icoord, dcoord, strict=False):
        pts = list(zip(map(sx, xs), map(sy, ys), strict=False))
        draw.line(pts, fill="#1d4ed8", width=2)

    img.save(path)


def _cluster_metrics(x: np.ndarray, labels: np.ndarray, model: object | None = None) -> dict[str, float]:
    labels = np.asarray(labels)
    mask = labels != -1
    x_eval = x[mask] if mask.any() else x
    labels_eval = labels[mask] if mask.any() else labels

    uniq = np.unique(labels_eval)
    valid = len(uniq) >= 2 and len(labels_eval) > len(uniq)

    out = {
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "coverage": float(mask.mean()) if len(labels) else float("nan"),
        "n_groups": float(len(uniq)),
    }

    if valid:
        out["silhouette"] = float(silhouette_score(x_eval, labels_eval))
        out["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, labels_eval))
        out["davies_bouldin"] = float(davies_bouldin_score(x_eval, labels_eval))

    if hasattr(model, "aic"):
        try:
            out["aic"] = float(model.aic(x))
        except Exception:
            out["aic"] = float("nan")
    if hasattr(model, "bic"):
        try:
            out["bic"] = float(model.bic(x))
        except Exception:
            out["bic"] = float("nan")

    return out


def _fit_candidate(algorithm: str, n_clusters: int, x: np.ndarray, seed: int) -> tuple[object | None, np.ndarray, str]:
    if n_clusters < 2:
        raise ValueError("k deve ser >= 2")

    if algorithm == "agglomerative_ward":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(x)
        return model, labels, ""

    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=seed,
            n_init=5,
        )
        labels = model.fit_predict(x)
        return model, labels, ""

    if algorithm == "spectral":
        n_neighbors = min(10, max(2, x.shape[0] - 1))
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=seed,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
        )
        labels = model.fit_predict(x)
        return model, labels, ""

    if algorithm == "birch":
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(x)
        return model, labels, ""

    if algorithm == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except Exception:
            return None, np.full(x.shape[0], -1, dtype=int), "hdbscan_nao_instalado"

        min_cluster_size = max(5, int(round(math.sqrt(x.shape[0]))))
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = model.fit_predict(x)
        return model, labels, ""

    raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def _rank_series_norm(s: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Normalização baseada em min-max entre candidatos válidos."""
    s = s.astype(float)
    valid = s.replace([np.inf, -np.inf], np.nan)
    minv = float(valid.min(skipna=True))
    maxv = float(valid.max(skipna=True))
    if not np.isfinite(minv) or not np.isfinite(maxv) or maxv == minv:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        out = (valid - minv) / (maxv - minv)

    if not higher_is_better:
        out = 1.0 - out

    out = out.fillna(0.0)
    return out


def _bootstrap_pairwise_jaccard(ref: np.ndarray, boot: np.ndarray) -> float:
    ref = np.asarray(ref)
    boot = np.asarray(boot)
    n = len(ref)
    if n < 3:
        return float("nan")

    same_ref = ref[:, None] == ref[None, :]
    same_boot = boot[:, None] == boot[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    a = same_ref[mask]
    b = same_boot[mask]
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")

    return float(np.logical_and(a, b).sum() / union)


def _stability_for_candidate(
    candidate: ModelCandidate,
    x: np.ndarray,
    seed: int,
    n_bootstrap: int,
    sample_size: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, x.shape[0])

    ari_values: list[float] = []
    jac_values: list[float] = []

    cluster_counts: list[float] = []
    noise_rates: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, x.shape[0], size=sample_size)
        x_boot = x[idx]

        _, boot_labels, _ = _fit_candidate(candidate.algorithm, candidate.n_clusters, x_boot, seed)
        ref_labels = candidate.labels[idx]

        mask = np.ones_like(ref_labels, dtype=bool)
        if np.any(ref_labels == -1) or np.any(boot_labels == -1):
            mask = (ref_labels != -1) & (boot_labels != -1)

        if mask.sum() >= 3 and len(np.unique(ref_labels[mask])) >= 2 and len(np.unique(boot_labels[mask])) >= 2:
            ari_values.append(float(adjusted_rand_score(ref_labels[mask], boot_labels[mask])))
            jac_values.append(_bootstrap_pairwise_jaccard(ref_labels[mask], boot_labels[mask]))
        else:
            ari_values.append(float(adjusted_rand_score(ref_labels, boot_labels)))
            jac_values.append(_bootstrap_pairwise_jaccard(ref_labels, boot_labels))

        cluster_counts.append(float(len(np.unique(boot_labels[boot_labels != -1]))))
        noise_rates.append(float(np.mean(boot_labels == -1)))

    ari = float(np.nanmean(ari_values)) if ari_values else float("nan")
    jacc = float(np.nanmean(jac_values)) if jac_values else float("nan")
    stability = float(np.nanmean([(ari + 1) / 2, jacc]))

    return {
        "ari_mean": ari,
        "ari_std": float(np.nanstd(ari_values)) if ari_values else float("nan"),
        "jaccard_mean": jacc,
        "jaccard_std": float(np.nanstd(jac_values)) if jac_values else float("nan"),
        "stability_mean": stability,
        "bootstrap_cluster_count_mean": float(np.nanmean(cluster_counts)),
        "bootstrap_noise_rate_mean": float(np.nanmean(noise_rates)),
    }


def _compute_cluster_profiles(
    df: pd.DataFrame,
    selected_features: list[str],
    labels: np.ndarray,
    *,
    cluster_profile_top_above: int,
    cluster_profile_top_below: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = _numeric_frame(df, selected_features)
    x = x.apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_z = pd.DataFrame(scaler.fit_transform(imp.fit_transform(x)), columns=selected_features, index=df.index)

    global_means = x_z.mean(axis=0)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for cluster_id in sorted(pd.unique(pd.Series(labels)).tolist()):
        if int(cluster_id) == -1:
            continue
        mask = labels == cluster_id
        cluster_frame = x_z.loc[mask]
        if cluster_frame.empty:
            continue

        cluster_mean = cluster_frame.mean(axis=0)
        delta = cluster_mean - global_means
        prevalence = float(mask.mean())
        size = int(mask.sum())

        top_pos = delta.sort_values(ascending=False).head(cluster_profile_top_above)
        top_neg = delta.sort_values(ascending=True).head(cluster_profile_top_below)

        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": size,
                "prevalence": prevalence,
                "top_above": ";".join(top_pos.index.tolist()),
                "top_below": ";".join(top_neg.index.tolist()),
            }
        )

        for feat in selected_features:
            d = float(delta[feat])
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "feature": feat,
                    "cluster_mean_z": float(cluster_mean[feat]),
                    "global_mean_z": float(global_means[feat]),
                    "delta_z": d,
                    "abs_delta_z": abs(d),
                    "direction": "above" if d >= 0 else "below",
                    "size": size,
                    "prevalence": prevalence,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _name_clusters_from_deltas(profile_summary: pd.DataFrame, profile_rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Nomeação simples por domínios: usa top_above/top_below e mapeia por grupos definidos em feature_groups."""
    groups = _feature_groups_from_config(cfg)
    var_to_group: dict[str, str] = {}
    for g, pats in groups.items():
        compiled = [re.compile(p, re.I) for p in pats]
        for v in profile_rows["feature"].unique().tolist():
            lv = str(v).lower()
            if any(p.search(lv) for p in compiled):
                var_to_group[str(v)] = g

    top_vars_n = int(cfg.get("clustering", {}).get("cluster_name_top_variables", 3))

    # build lookup of variable delta sign magnitude
    delta_lookup = (
        profile_rows.groupby(["cluster_id", "feature"])["delta_z"].mean().reset_index()
    )

    names: list[dict[str, object]] = []
    for _, row in profile_summary.iterrows():
        cid = int(row["cluster_id"])
        above_vars = str(row["top_above"]).split(";") if pd.notna(row["top_above"]) else []
        below_vars = str(row["top_below"]).split(";") if pd.notna(row["top_below"]) else []

        above_top = (
            delta_lookup[(delta_lookup["cluster_id"] == cid) & (delta_lookup["feature"].isin(above_vars))]
            .sort_values("delta_z", ascending=False)
            .head(top_vars_n)["feature"].tolist()
        )
        below_top = (
            delta_lookup[(delta_lookup["cluster_id"] == cid) & (delta_lookup["feature"].isin(below_vars))]
            .sort_values("delta_z", ascending=True)
            .head(top_vars_n)["feature"].tolist()
        )

        above_groups = [var_to_group.get(v) for v in above_top]
        above_groups = [g for g in above_groups if g]
        below_groups = [var_to_group.get(v) for v in below_top]
        below_groups = [g for g in below_groups if g]

        # choose dominant group
        dominant = None
        if above_groups:
            dominant = pd.Series(above_groups).value_counts().idxmax()
            mode = "Alta"
        elif below_groups:
            dominant = pd.Series(below_groups).value_counts().idxmax()
            mode = "Baixa"

        if not dominant:
            cluster_name = f"Perfil cluster {cid}"
        else:
            # human-readable mapping
            pretty = {
                "accesso_tecnologico": "Resiliência Tecnológica",
                "recursos_domesticos": "Resiliência Doméstica",
                "apoio_familiar": "Resiliência Familiar",
                "clima_escolar": "Resiliência Escolar",
                "pertencimento": "Resiliência por Pertencimento",
                "motivacao": "Resiliência Motivacional",
                "autoeficacia": "Resiliência por Autoeficácia",
                "ansiedade": "Resiliência vs Ansiedade",
                "indicadores_processuais": "Resiliência Processual",
                "engajamento_tarefas_criativas": "Engajamento Criativo",
            }.get(dominant, dominant)
            cluster_name = f"{mode} — {pretty}"

        names.append({"cluster_id": cid, "cluster_name": cluster_name, "dominant_group": dominant})

    ndf = pd.DataFrame(names)
    out = profile_summary.merge(ndf, on="cluster_id", how="left")
    if "cluster_name" not in out.columns:
        out["cluster_name"] = "Perfil"
    return out


def _cluster_text_description(profile_summary: pd.DataFrame) -> str:
    if profile_summary.empty:
        return "Sem perfis suficientes para descrever os clusters."
    lines: list[str] = []
    for _, row in profile_summary.sort_values("cluster_id").iterrows():
        name = row.get("cluster_name") if "cluster_name" in profile_summary.columns else row.get("top_above", "perfil")
        lines.append(f"cluster {int(row['cluster_id'])}: {name} (n={int(row['size'])}, prev={float(row['prevalence']):.1%})")
    return "\n".join(lines)


def _train_cluster_explainer(
    x_std_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    seed: int,
    out_tables: Path,
    out_figures: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = labels != -1
    if mask.sum() < 20 or len(np.unique(labels[mask])) < 2:
        empty = pd.DataFrame(columns=["feature", "shap_mean_abs", "permutation_mean", "combined_importance"])
        return empty, empty

    x = x_std_df.loc[mask, feature_cols]
    y = pd.Series(labels[mask], index=x.index)

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(x, y)

    perm = permutation_importance(
        clf,
        x,
        y,
        n_repeats=8,
        random_state=seed,
        n_jobs=-1,
    )

    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(x)
        if isinstance(values, list):
            shap_array = np.mean([np.abs(v) for v in values], axis=0)
        else:
            shap_array = np.abs(values)
            if shap_array.ndim == 3:
                shap_array = shap_array.mean(axis=0)
        shap_mean = np.asarray(shap_array).mean(axis=0)
    except Exception:
        shap_mean = np.zeros(len(feature_cols), dtype=float)

    shap_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "shap_mean_abs": shap_mean,
        }
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_mean": perm.importances_mean,
        }
    )

    merged = shap_df.merge(perm_df, on="feature", how="left")
    merged["combined_importance"] = merged[["shap_mean_abs", "permutation_mean"]].fillna(0).mean(axis=1)
    merged = merged.sort_values("combined_importance", ascending=False).reset_index(drop=True)

    shap_df2 = merged[["feature", "shap_mean_abs", "combined_importance"]].copy()
    perm_df2 = merged[["feature", "permutation_mean", "combined_importance"]].copy()

    merged.to_csv(out_tables / "cluster_shap_importance.csv", index=False)
    perm_df2[["feature", "permutation_mean", "combined_importance"]].to_csv(
        out_tables / "cluster_permutation_importance.csv", index=False
    )

    # draw a simple bar chart for top variables
    top = merged.head(20).iloc[::-1]
    width, height = 1200, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 30), "SHAP (mean |abs|) + Permutation Importance", fill="#0f172a", font=_font(24))

    left, right, top_margin, bottom = 250, 80, 90, 90
    draw.line((left, top_margin, left, height - bottom), fill="#64748b", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#64748b", width=2)

    max_val = float(np.nanmax(top["combined_importance"].to_numpy(dtype=float))) or 1.0
    bar_h = (height - top_margin - bottom) / max(1, len(top))

    for i, (_, row) in enumerate(top.iterrows()):
        y0 = top_margin + i * bar_h + 4
        y1 = y0 + bar_h - 8
        w = (width - left - right) * float(row["combined_importance"]) / max_val
        draw.rectangle((left, y0, left + w, y1), fill="#1d4ed8")
        draw.text((40, y0 + 2), str(row["feature"])[:28], fill="#0f172a", font=_font(15))
        draw.text((left + w + 8, y0 + 2), f"{float(row['combined_importance']):.3f}", fill="#334155", font=_font(14))

    img.save(out_figures / "cluster_shap_summary.png")

    return merged, perm_df2


def _compute_cluster_resilience_association(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    target_series: pd.Series,
) -> pd.DataFrame:
    tmp = pd.DataFrame({"cluster_id": cluster_labels, "resiliente": pd.to_numeric(target_series, errors="coerce").fillna(0).astype(int)})

    out_rows = []
    global_pos = float(tmp["resiliente"].mean())

    for cid in sorted(pd.unique(tmp["cluster_id"]).tolist()):
        if int(cid) == -1:
            continue
        mask = tmp["cluster_id"] == cid
        n = int(mask.sum())
        n_pos = int(tmp.loc[mask, "resiliente"].sum())
        prev = float(n_pos / n) if n else float("nan")

        # contingency 2x2: cluster vs not
        a = n_pos
        b = n - n_pos
        c = int(tmp.loc[~mask, "resiliente"].sum())
        d = int((~mask).sum() - tmp.loc[~mask, "resiliente"].sum())

        # odds ratio
        eps = 1e-9
        odds_cluster = (a + eps) / (b + eps)
        odds_not = (c + eps) / (d + eps)
        odds_ratio = float(odds_cluster / odds_not)

        # risk relative
        risk_cluster = (a + eps) / (n + 2 * eps)
        risk_not = (c + eps) / ((~mask).sum() + 2 * eps)
        risk_relative = float(risk_cluster / risk_not)

        out_rows.append(
            {
                "cluster_id": int(cid),
                "n": n,
                "n_resilientes": n_pos,
                "prevalence_resilientes": prev,
                "odds_ratio": odds_ratio,
                "risk_relative": risk_relative,
                "global_resilient_prevalence": global_pos,
            }
        )

    return pd.DataFrame(out_rows).sort_values("prevalence_resilientes", ascending=False)


def _bar_chart_resilience_distribution(path: Path, assoc_df: pd.DataFrame) -> None:
    # Best-effort simple PNG using PIL
    width, height = 1400, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 30), "Prevalência de resilientes por cluster", fill="#0f172a", font=_font(28))

    if assoc_df is None or assoc_df.empty:
        img.save(path)
        return

    df = assoc_df.copy()
    df = df.sort_values("cluster_id")

    max_val = float(df["prevalence_resilientes"].max()) if "prevalence_resilientes" in df.columns else 1.0
    max_val = max(max_val, 1e-9)

    left, right, top, bottom = 120, 80, 90, 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    n = len(df)
    bar_w = plot_w / max(1, n)

    for i, (_, r) in enumerate(df.iterrows()):
        cid = int(r["cluster_id"])
        val = float(r["prevalence_resilientes"])
        h = val / max_val * plot_h

        x0 = int(left + i * bar_w + bar_w * 0.15)
        x1 = int(left + (i + 1) * bar_w - bar_w * 0.15)
        y0 = int(top + plot_h - h)
        y1 = int(top + plot_h)

        color = PALETTE[i % len(PALETTE)]
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=color)
        draw.text((x0 + 4, y0 - 22), f"{val*100:.1f}%", fill="#334155", font=_font(14))
        draw.text((x0 + 2, top + plot_h + 10), f"{cid}", fill="#0f172a", font=_font(14))

    img.save(path)


def _get_vulnerable_mask(df: pd.DataFrame, target_key: str) -> pd.Series:
    """Heurística compatível com o projeto atual: usa targets A/B/C/D se existirem.

    Requisito: vulnerable_only usar o grupo vulnerável usado na definição principal de resiliência criativa.

    Nesta base, a resiliência é operacional; como o código do alvo detalhado está em target_builder, usamos:
    - se houver colunas target_{k}, tratamos vulneráveis como aqueles com contexto vulnerável (ESCS baixo).

    Como não temos aqui a coluna de contexto final, usamos proxy robusto:
    - se existir coluna 'grupo_escs' (ou equivalente) e ela for numérica/categórica, vulneráveis = grupo_escs baixo.
    - fallback: usa resilientes (target_*) = 1 e define vulnerable como target_=0.

    Nota: isso precisa ser conectado ao target_builder/refactoring do artigo; mantemos fallback reprodutível.
    """
    if f"target_{target_key}" in df.columns:
        y = pd.to_numeric(df[f"target_{target_key}"], errors="coerce").fillna(0).astype(int)
        # fallback: vulnerable = not resilient
        return (y == 0)

    col_candidates = ["grupo_escs", "Grupo_ESCS"]
    for c in col_candidates:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                # assume low values represent vulnerability
                thr = s.quantile(0.33)
                return s <= thr
            # categorical: vulnerable if string contains low-ish cues
            sl = s.astype(str).str.lower()
            return sl.str.contains("baixo") | sl.str.contains("low")

    # ultimate fallback
    return pd.Series(np.ones(len(df), dtype=bool), index=df.index)


def run_clusterer(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)

    out_reports = _ensure_dir(Path(cfg["outputs"]["reports_dir"]))
    out_tables = _ensure_dir(Path(cfg["outputs"]["tables_dir"]))
    out_figures = _ensure_dir(Path(cfg["outputs"]["figures_dir"]) / "clustering")

    analysis_mode = str(cfg.get("clustering", {}).get("analysis_mode", "full_population"))

    # cluster discovery / person-centered: mantém variáveis selecionadas e aplica PCA antes dos modelos.
    # (Etapa 1–5) A escolha do melhor modelo é orientada por qualidade + estabilidade (bootstrap),
    # não apenas por silhouette.

    # Build theoretical feature set (Etapa 1)
    selected_features, catalog = _select_theoretical_features(df, cfg)
    if len(selected_features) < 3:
        raise ValueError("Poucas features teóricas disponíveis para clusterização (mínimo 3).")

    catalog.to_csv(out_tables / "clustering_feature_catalog.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(out_tables / "clustering_selected_features.csv", index=False)

    # vulnerability subset (Etapa 2 extension)
    # target_key: from config analysis.target_profile_key (used in resilient_profile)
    target_key = str(cfg.get("analysis", {}).get("target_profile_key", "A"))
    target_col = f"target_{target_key}"

    if analysis_mode == "vulnerable_only":
        if target_col not in df.columns:
            # if target_{key} columns are not present, fallback to build_target_definitions is not available here
            # so we use resilient_profile selection heuristic
            vulnerable_mask = _get_vulnerable_mask(df, target_key)
        else:
            vulnerable_mask = _get_vulnerable_mask(df, target_key)
        df_work = df.loc[vulnerable_mask].copy()
    else:
        df_work = df.copy()

    # Etapa 2: matrix preprocessing
    x_scaled_df, _, x_std_df = _prepare_matrix(df_work, selected_features)
    # drop collinearity
    threshold = float(cfg.get("clustering", {}).get("collinearity_threshold", 0.95))
    reduced_features = _drop_collinear_features(pd.DataFrame(x_std_df, columns=selected_features, index=df_work.index), selected_features, threshold)
    # rebuild matrix with reduced_features
    x_scaled_df, _, x_std_df = _prepare_matrix(df_work, reduced_features)

    # PCA + outputs required
    x_pca, explained_df, n_pca = _fit_pca(x_scaled_df, cfg)

    explained_df.to_csv(out_tables / "explained_variance.csv", index=False)
    _line_chart(
        out_figures / "scree_plot.png",
        list(range(1, len(explained_df) + 1)),
        explained_df["explained_variance_ratio"].tolist(),
        "Scree plot do PCA",
        "componentes",
        "variância explicada",
    )
    _line_chart(
        out_figures / "cumulative_variance.png",
        list(range(1, len(explained_df) + 1)),
        explained_df["cumulative_variance"].tolist(),
        "Variância acumulada do PCA",
        "componentes",
        "variância acumulada",
        threshold=float(cfg.get("clustering", {}).get("pca_variance_target", 0.85)),
    )

    k_min = int(cfg.get("clustering", {}).get("n_clusters_min", 2))
    k_max = int(cfg.get("clustering", {}).get("n_clusters_max", 10))
    algorithms = list(cfg.get("clustering", {}).get("algorithms", [])) or [
        "agglomerative_ward",
        "gmm",
        "spectral",
        "birch",
        "hdbscan",
    ]

    n_bootstrap = int(cfg.get("clustering", {}).get("bootstrap_iterations", 100))
    bootstrap_sample = int(cfg.get("clustering", {}).get("bootstrap_sample_size", 300))
    stability_top_n = int(cfg.get("clustering", {}).get("stability_top_n", 6))

    # grid (Etapa 3)
    # OPT-5: HDBSCAN não usa n_clusters; executar apenas uma vez por dataset
    candidates: list[ModelCandidate] = []
    for algorithm in algorithms:
        k_range = range(k_min, k_max + 1)
        for k in k_range:
            # OPT-5: algoritmos que ignoram n_clusters só executam na primeira iteração
            if algorithm in ("hdbscan",) and k != k_min:
                continue
            try:
                model, labels, note = _fit_candidate(algorithm, k, x_pca, seed)
                quality = _cluster_metrics(x_pca, labels, model)
                candidates.append(
                    ModelCandidate(
                        algorithm=algorithm,
                        n_clusters=k,
                        model=model,
                        labels=labels,
                        quality=quality,
                        notes=note,
                    )
                )
            except Exception as exc:
                candidates.append(
                    ModelCandidate(
                        algorithm=algorithm,
                        n_clusters=k,
                        model=None,
                        labels=np.full(x_pca.shape[0], -1, dtype=int),
                        quality={
                            "silhouette": float("nan"),
                            "calinski_harabasz": float("nan"),
                            "davies_bouldin": float("nan"),
                            "coverage": float("nan"),
                            "n_groups": float("nan"),
                        },
                        notes=f"erro:{type(exc).__name__}",
                    )
                )

    comp_rows = []
    for c in candidates:
        comp_rows.append({"algorithm": c.algorithm, "k": c.n_clusters, **c.quality, "notes": c.notes})

    comparison_df = pd.DataFrame(comp_rows)
    comparison_df["algorithm"] = comparison_df["algorithm"].astype(str)

    # SR-6: compute stability SOMENTE nos top-N candidatos por qualidade (silhouette)
    # Bug anterior: usava os N primeiros da lista (ordem de inserção no grid),
    # ignorando a qualidade dos clusters. Agora os top-N são selecionados por silhouette.
    valid_cands = [
        c for c in candidates
        if c.model is not None and np.isfinite(c.quality.get("silhouette", np.nan))
    ]
    top_for_stability = sorted(
        valid_cands,
        key=lambda c: c.quality.get("silhouette", -np.inf),
        reverse=True,
    )[:stability_top_n]

    stability_rows: list[dict[str, object]] = []
    for c in top_for_stability:
        s = _stability_for_candidate(c, x_pca, seed, n_bootstrap, bootstrap_sample)
        stability_rows.append({"algorithm": c.algorithm, "k": c.n_clusters, **s})

    stability_df = pd.DataFrame(stability_rows)

    # merge stability into comparison
    if not stability_df.empty:
        comparison_df = comparison_df.merge(stability_df, on=["algorithm", "k"], how="left")
    else:
        comparison_df["ari_mean"] = np.nan
        comparison_df["ari_std"] = np.nan
        comparison_df["jaccard_mean"] = np.nan
        comparison_df["jaccard_std"] = np.nan
        comparison_df["stability_mean"] = np.nan

    comparison_df["stability_norm_input"] = (comparison_df["ari_mean"] + comparison_df["jaccard_mean"]) / 2

    # multicritério cluster_score (Etapa 1 Novo critério)
    w = cfg.get("clustering", {}).get("cluster_score_weights") or {
        "silhouette": 0.30,
        "calinski_harabasz": 0.20,
        "davies_bouldin": -0.15,
        "stability": 0.35,
    }

    comparison_df["silhouette_norm"] = _rank_series_norm(comparison_df["silhouette"], higher_is_better=True)
    comparison_df["calinski_norm"] = _rank_series_norm(comparison_df["calinski_harabasz"], higher_is_better=True)
    comparison_df["davies_norm"] = _rank_series_norm(comparison_df["davies_bouldin"], higher_is_better=False)
    comparison_df["stability_norm"] = _rank_series_norm(comparison_df["stability_norm_input"], higher_is_better=True)

    comparison_df["cluster_score"] = (
        float(w.get("silhouette", 0.30)) * comparison_df["silhouette_norm"]
        + float(w.get("calinski_harabasz", 0.20)) * comparison_df["calinski_norm"]
        + float(w.get("davies_bouldin", -0.15)) * comparison_df["davies_norm"]
        + float(w.get("stability", 0.35)) * comparison_df["stability_norm"]
    )

    # output required columns
    comparison_out = comparison_df.copy()
    comparison_out = comparison_out[[
        "algorithm",
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_mean",
        "ari_std",
        "jaccard_mean",
        "jaccard_std",
        "cluster_score",
    ]].copy()

    comparison_out["ranking_final"] = comparison_out["cluster_score"].rank(ascending=False, method="min")
    comparison_out = comparison_out.sort_values(["cluster_score"], ascending=False).reset_index(drop=True)

    comparison_out.to_csv(out_tables / "clustering_model_comparison.csv", index=False)
    comparison_df.to_csv(out_tables / "clusterer_metrics.csv", index=False)
    stability_df.to_csv(out_tables / "cluster_stability.csv", index=False)

    # best model (Etapa 7)
    best_row = comparison_out.iloc[0]
    best_model = next(
        (c for c in candidates if c.algorithm == best_row["algorithm"] and c.n_clusters == int(best_row["k"])),
        None,
    )
    if best_model is None:
        raise RuntimeError("Não foi possível recuperar o melhor candidato de clusterização.")

    best_labels = best_model.labels

    # profiles (Etapa 6 + Etapa 3)
    top_above = int(cfg.get("clustering", {}).get("cluster_profile_top_above", 10))
    top_below = int(cfg.get("clustering", {}).get("cluster_profile_top_below", 10))

    profile_rows, profile_summary = _compute_cluster_profiles(
        df_work,
        reduced_features,
        best_labels,
        cluster_profile_top_above=top_above,
        cluster_profile_top_below=top_below,
    )

    profile_summary_named = _name_clusters_from_deltas(profile_summary, profile_rows, cfg)

    profile_rows.to_csv(out_tables / "cluster_profiles.csv", index=False)
    profile_summary_named.to_csv(out_tables / "cluster_profiles_summary.csv", index=False)

    # required: cluster_profiles_interpretable.csv
    interpretable = profile_summary_named.merge(profile_summary, on=["cluster_id", "size", "prevalence", "top_above", "top_below"], how="left")
    interpretable = interpretable[["cluster_id", "cluster_name", "size", "prevalence", "top_above", "top_below"]]
    interpretable.to_csv(out_tables / "cluster_profiles_interpretable.csv", index=False)

    # vulnerable outputs (Etapa 2 específico)
    if analysis_mode == "vulnerable_only":
        profile_summary_named.to_csv(out_tables / "vulnerable_cluster_profiles.csv", index=False)
        # report minimal vulnerable report
        (out_reports / "vulnerable_cluster_report.md").write_text(
            "".join(
                [
                    "# Perfis em população vulnerável (vulnerable_only)\n\n",
                    f"- N analisado: {len(df_work):,}\n",
                    f"- Modelo: **{best_model.algorithm}** com k={best_model.n_clusters}\n",
                    "\n## Perfis\n\n",
                    df_to_markdown(profile_summary_named),
                    "\n",
                ]
            ),
            encoding="utf-8",
        )

    # association cluster x target (Etapa 5 + Etapa 7)
    if target_col in df_work.columns:
        target_series = pd.Series(pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).astype(int), index=df_work.index)
    else:
        # fallback: if missing targets, set all 0 (no association)
        target_series = pd.Series(np.zeros(len(df_work), dtype=int), index=df_work.index)

    assoc_df = _compute_cluster_resilience_association(df_work, best_labels, target_series)
    assoc_df.to_csv(out_tables / "cluster_resilience_association.csv", index=False)
    _bar_chart_resilience_distribution(out_figures / "cluster_resilience_distribution.png", assoc_df)

    # visualizações (Etapa 8)
    pca_2d = PCA(n_components=2, random_state=seed).fit_transform(x_pca)
    _scatter_chart(
        out_figures / "cluster_pca.png",
        pca_2d[:, 0],
        pca_2d[:, 1],
        best_labels,
        "PCA 2D colorido por cluster",
        f"Modelo escolhido: {best_model.algorithm} | k={best_model.n_clusters}",
    )

    try:
        umap_embedding = None
        try:
            import umap  # type: ignore

            umap_embedding = umap.UMAP(
                n_components=2,
                random_state=seed,
                n_neighbors=min(15, max(2, x_pca.shape[0] - 1)),
                min_dist=0.1,
            ).fit_transform(x_pca)
        except Exception:
            umap_embedding = SpectralEmbedding(
                n_components=2,
                random_state=seed,
                n_neighbors=min(15, max(2, x_pca.shape[0] - 1)),
            ).fit_transform(x_pca)

        _scatter_chart(
            out_figures / "cluster_umap.png",
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            best_labels,
            "UMAP 2D colorido por cluster",
            "Se `umap-learn` não estiver disponível, foi usado SpectralEmbedding fallback.",
        )
    except Exception:
        # best-effort: don't block pipeline
        pass

    # Heatmap + Radar + Dendrogram
    top_features = profile_rows.groupby("feature")["abs_delta_z"].mean().sort_values(ascending=False).head(16).index.tolist()
    if top_features:
        heatmap_matrix = (
            profile_rows[profile_rows["feature"].isin(top_features)]
            .pivot_table(index="cluster_id", columns="feature", values="delta_z", aggfunc="mean")
            .fillna(0)
        )
        _heatmap_chart(out_figures / "cluster_heatmap.png", heatmap_matrix, "Heatmap dos perfis")

        radar_features = top_features[:8]
        if len(radar_features) >= 3:
            radar_profiles = (
                profile_rows[profile_rows["feature"].isin(radar_features)]
                .pivot_table(index="cluster_id", columns="feature", values="cluster_mean_z", aggfunc="mean")
                .fillna(0)
                .reset_index()
            )
            _radar_chart(out_figures / "cluster_radar.png", radar_profiles, radar_features, "Radar dos perfis")

    ward_candidates = [c for c in candidates if c.algorithm == "agglomerative_ward" and c.model is not None]
    if ward_candidates:
        # dendrogram only if within best k set (best-effort)
        try:
            ward_best = next(
                (c for c in ward_candidates if c.n_clusters == best_model.n_clusters),
                ward_candidates[0],
            )
            dendro_x = x_pca[: min(120, x_pca.shape[0])]
            dendro_labels = ward_best.labels[: len(dendro_x)]
            _dendrogram_chart(out_figures / "cluster_dendrogram.png", dendro_x, dendro_labels, "Dendrograma Ward")
        except Exception:
            pass

    # Interpretability (Etapa 7)
    # rebuild standardized df for training explainer
    try:
        x_std_df = _numeric_frame(df_work, reduced_features)
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_std_df = pd.DataFrame(
            scaler.fit_transform(imp.fit_transform(x_std_df)),
            columns=reduced_features,
            index=df_work.index,
        )

        shap_importance_df, perm_importance_df = _train_cluster_explainer(
            x_std_df,
            best_labels,
            reduced_features,
            seed,
            out_tables,
            out_figures,
        )
    except Exception as exc:
        # Ensure required artifacts do not break downstream dashboard.
        empty_cols = ["feature", "shap_mean_abs", "combined_importance"]
        shap_importance_df = pd.DataFrame(columns=empty_cols)
        perm_importance_df = pd.DataFrame(columns=["feature", "permutation_mean", "combined_importance"])
        # write placeholders
        shap_importance_df.to_csv(out_tables / "cluster_shap_importance.csv", index=False)
        perm_importance_df[["feature", "permutation_mean", "combined_importance"]].to_csv(
            out_tables / "cluster_permutation_importance.csv", index=False
        )
        # add a hint to report
        shap_fail_note = f"Falha na interpretabilidade (SHAP/permutation): {type(exc).__name__}: {exc}"

    # report (Etapa 9)
    # build a compact full metric table
    table_for_report = comparison_out.copy()
    # top 10
    table_md = df_to_markdown(table_for_report.head(15))

    # per-cluster profile interpretado
    profile_md = df_to_markdown(profile_summary_named[["cluster_id", "cluster_name", "size", "prevalence", "top_above", "top_below"]])

    top_assoc = assoc_df.head(10)
    assoc_md = df_to_markdown(top_assoc)

    # best variables explanation: use top combined importance
    var_md = df_to_markdown(shap_importance_df.head(20)) if shap_importance_df is not None and not shap_importance_df.empty else ""

    report_lines = [
        "# Clusterer científico — Perfis de Resiliência Criativa (PISA 2022)\n\n",
        "## Seleção das Variáveis\n\n",
        f"- Features teóricas selecionadas: **{len(selected_features)}**\n",
        f"- Features após remoção de colinearidade: **{len(reduced_features)}**\n",
        "- Critérios: person-centered e sem leakage (exclusões por leakage, target e variáveis sensíveis a vazamento).\n\n",
        "## Redução de Dimensionalidade\n\n",
        f"- PCA retido: **{n_pca}** componentes (alvo {cfg.get('clustering', {}).get('pca_variance_target', 0.85)})\n",
        "- Artefatos: `outputs/figures/clustering/scree_plot.png` e `outputs/figures/clustering/cumulative_variance.png`\n\n",
        "## Comparação dos Algoritmos\n\n",
        "Tabela completa de métricas e estabilidade (top ranking por cluster_score):\n\n",
        table_md,
        "\n\n## Estabilidade dos Clusters\n\n",
        "- Bootstrap ARI/Jaccard implementado com `cluster_stability.csv`\n\n",
        df_to_markdown(stability_df.head(12)) if not stability_df.empty else "Sem resultados de bootstrap.\n",
        "\n\n## Solução Escolhida\n\n",
        f"- Modelo: **{best_model.algorithm}** com `k={best_model.n_clusters}`\n",
        "- Justificativa: seleção por score multicritério (qualidade estrutural + estabilidade via ARI/Jaccard) — não apenas silhouette.\n\n",
        "## Perfis Identificados\n\n",
        profile_md,
        "\n\n## Associação com Resiliência Criativa\n\n",
        "Distribuição de resilientes por perfil (odds ratio e risco relativo):\n\n",
        assoc_md,
        "\n\n## Variáveis Mais Relevantes\n\n",
        "Quais variáveis distinguem os perfis (SHAP + Permutation Importance):\n\n",
        var_md,
        "\n\n## Limitações\n\n",
        "- Natureza descritiva (person-centered) — não causalidade.\n",
        "- Estabilidade depende de amostragem bootstrap.\n",
        "- HDBSCAN é opcional; quando indisponível, rótulos ficam em -1 e métricas podem ser menos informativas.\n\n",
        "## Implicações Educacionais\n\n",
        "- Perfis sugerem intervenções diferenciadas por padrão psicossocial/engajamento: tecnologia, apoio familiar, clima escolar e processos de tarefa.\n",
        "\n## Artefatos principais\n\n",
        "- `outputs/tables/clustering_model_comparison.csv`\n",
        "- `outputs/tables/cluster_stability.csv`\n",
        "- `outputs/tables/cluster_profiles_interpretable.csv`\n",
        "- `outputs/tables/cluster_resilience_association.csv`\n",
        "- `outputs/tables/cluster_shap_importance.csv`\n",
        "- `outputs/figures/clustering/`\n",
    ]

    (out_reports / "clusterer_report.md").write_text("".join(report_lines), encoding="utf-8")


# Backwards compatible entrypoint

