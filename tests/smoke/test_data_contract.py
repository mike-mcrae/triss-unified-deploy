from __future__ import annotations

import csv
import json
import os
import unittest
from pathlib import Path


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip("\"").strip("'")
    return out


def _data_dir(root: Path) -> Path:
    defaults = {"TRISS_DATA_DIR": str(root / "data")}
    settings = defaults | _parse_simple_yaml(root / "config" / "settings.local.yml")
    env = os.environ.get("TRISS_DATA_DIR")
    if env:
        settings["TRISS_DATA_DIR"] = env
    p = Path(settings["TRISS_DATA_DIR"]).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p.resolve()


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


class DataContractSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.final = _data_dir(cls.root) / "final"

    def test_required_artifacts_exist(self) -> None:
        required = [
            self.final / "profiles" / "1. profiles_summary.csv",
            self.final / "profiles" / "4. triss_researcher_summary.csv",
            self.final / "network" / "7.researcher_similarity_matrix.csv",
            self.final / "publications" / "4.All measured publications.csv",
            self.final / "analysis" / "global" / "global_cluster_descriptions.json",
            self.final / "analysis" / "global" / "global_umap_coordinates.csv",
            self.final / "analysis" / "global" / "stage3" / "policy_domains_metadata.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        self.assertFalse(missing, f"Missing required files: {missing}")

    def test_schema_columns(self) -> None:
        checks = [
            (self.final / "profiles" / "1. profiles_summary.csv", {"n_id", "firstname", "lastname", "school", "department"}),
            (self.final / "profiles" / "4. triss_researcher_summary.csv", {"n_id", "one_line_summary", "overall_research_area"}),
            (self.final / "publications" / "4.All measured publications.csv", {"n_id", "article_id", "Title", "abstract"}),
            (self.final / "analysis" / "global" / "global_umap_coordinates.csv", {"n_id", "cluster", "umap_x", "umap_y"}),
        ]
        for path, required_cols in checks:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                columns = set(reader.fieldnames or [])
            self.assertTrue(required_cols.issubset(columns), f"{path} missing columns {required_cols - columns}")

    def test_minimum_row_counts(self) -> None:
        counts = [
            _row_count(self.final / "profiles" / "1. profiles_summary.csv"),
            _row_count(self.final / "publications" / "4.All measured publications.csv"),
            _row_count(self.final / "analysis" / "global" / "global_umap_coordinates.csv"),
        ]
        for count in counts:
            self.assertGreater(count, 0)

    def test_app_import(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("fastapi not installed in current interpreter")
        os.environ.setdefault("TRISS_DATA_DIR", str(self.final.parent))
        import app.backend.main  # noqa: F401


if __name__ == "__main__":
    unittest.main()
