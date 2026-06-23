from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.utils.config import load_config
from src.utils.dataset_loader import detect_primary_csv
from src.utils.paths import METADATA_PATH, PROJECT_ROOT
from src.utils.pipeline_data import load_dataframe, merge_saved_targets


def ensure_dirs(cfg: dict) -> None:
    for p in [
        cfg["paths"]["data_dir"],
        cfg["paths"]["outputs_dir"],
        cfg["paths"]["models_dir"],
        cfg["paths"]["dashboard_dir"],
        cfg["paths"]["src_dir"],
        cfg["paths"]["tests_dir"],
        cfg["paths"]["docs_dir"],
        cfg["outputs"]["figures_dir"],
        cfg["outputs"]["tables_dir"],
        cfg["outputs"]["reports_dir"],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def update_metadata(csv_path: Path) -> None:
    meta_path = METADATA_PATH
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("dataset", {})
    try:
        meta["dataset"]["csv_detected"] = str(csv_path.relative_to(PROJECT_ROOT))
    except ValueError:
        meta["dataset"]["csv_detected"] = str(csv_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_df_for_modeling(cfg: dict, csv_path: Path) -> pd.DataFrame:
    df = load_dataframe(csv_path)
    return merge_saved_targets(df, cfg)


StageFn = Callable[[dict, pd.DataFrame, Path], None]


def run_stage(stage: str, cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    csv_path = detect_primary_csv(cfg)
    update_metadata(csv_path)

    if stage == "data_audit":
        from src.data_audit import run_data_audit

        run_data_audit(cfg, csv_path)
        return

    if stage == "dashboard":
        run_dashboard_stage(cfg)
        return

    df = load_df_for_modeling(cfg, csv_path) if stage != "data_dictionary" else load_dataframe(csv_path)

    handlers: dict[str, StageFn | Callable[..., None]] = {
        "data_dictionary": lambda c, d, p: __import__(
            "src.data_dictionary", fromlist=["make_data_dictionary"]
        ).make_data_dictionary(d, Path(c["outputs"]["reports_dir"]) / "data_dictionary.md"),
        "variable_discovery": lambda c, d, p: __import__(
            "src.variable_discovery", fromlist=["run_variable_discovery"]
        ).run_variable_discovery(c, d),
        "leakage_audit": lambda c, d, p: __import__(
            "src.leakage_detector", fromlist=["run_leakage_audit"]
        ).run_leakage_audit(c, d, p),
        "target_comparison": lambda c, d, p: __import__(
            "src.target_builder", fromlist=["run_target_comparison"]
        ).run_target_comparison(c, d, p),
        "eda": lambda c, d, p: __import__("src.eda", fromlist=["run_eda"]).run_eda(c, d, p),
        "resilient_profile": lambda c, d, p: __import__(
            "src.resilient_profile", fromlist=["run_resilient_profile"]
        ).run_resilient_profile(c, d, p),
        "clusterer": lambda c, d, p: __import__(
            "src.clusterer", fromlist=["run_clusterer"]
        ).run_clusterer(c, d, p),
        "modeling": lambda c, d, p: __import__(
            "src.modeling", fromlist=["run_modeling"]
        ).run_modeling(c, d, p),
        "shap": lambda c, d, p: __import__(
            "src.shap_analysis", fromlist=["run_shap_analysis"]
        ).run_shap_analysis(c, d, p),
        "fairness": lambda c, d, p: __import__(
            "src.fairness", fromlist=["run_fairness"]
        ).run_fairness(c, d, p),
        "robustness": lambda c, d, p: __import__(
            "src.robustness", fromlist=["run_robustness"]
        ).run_robustness(c, d, p),
        "sensitivity": lambda c, d, p: __import__(
            "src.sensitivity", fromlist=["run_sensitivity_analysis"]
        ).run_sensitivity_analysis(c, d, p),
    }

    if stage not in handlers:
        raise ValueError(f"Stage desconhecido: {stage}")

    if stage == "data_dictionary":
        handlers[stage](cfg, df, csv_path)  # type: ignore[misc]
    else:
        handlers[stage](cfg, df, csv_path)  # type: ignore[misc]


def run_dashboard_stage(cfg: dict) -> None:
    """Valida artefatos e imprime comando Streamlit (não abre servidor automaticamente)."""
    reports = Path(cfg["outputs"]["reports_dir"])
    n_reports = len(list(reports.glob("*.md"))) if reports.exists() else 0
    manifest = {
        "reports_count": n_reports,
        "streamlit_cmd": "python3 -m streamlit run dashboard/app.py",
    }
    out = Path(cfg["paths"]["dashboard_dir"]) / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dashboard: {n_reports} relatórios encontrados.")
    print(f"Para abrir: {manifest['streamlit_cmd']}")
