from __future__ import annotations

import glob
from pathlib import Path


def detect_primary_csv(cfg: dict) -> Path:
    patterns = cfg.get("data", {}).get("dataset_glob_patterns", [])
    candidates: list[Path] = []

    for pat in patterns:
        for p in glob.glob(pat):
            pp = Path(p)
            if pp.is_file():
                candidates.append(pp)

    # dedupe preserving order
    seen = set()
    unique: list[Path] = []
    for c in candidates:
        if str(c) not in seen:
            unique.append(c)
            seen.add(str(c))

    if not unique:
        raise FileNotFoundError(
            "Nenhum CSV correspondente encontrado em data/ pelos padrões: "
            + ", ".join(patterns)
        )

    # heuristic: prefer arquivo com 'limpo' no nome
    def score(p: Path) -> int:
        s = p.name.lower()
        return (10 if "limpo" in s else 0) + (5 if "estudo" in s else 0) + (1 if "pisa" in s else 0)

    unique.sort(key=score, reverse=True)
    return unique[0]

