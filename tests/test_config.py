import pytest
from src.utils.config import load_config, get_data_paths

def test_data_paths_roundtrip(tmp_path, monkeypatch):
    # 1) Create a temp YAML
    cfg = tmp_path / "config.yaml"
    cfg.write_text("data:\n  raw: foo.csv\n")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))

    # 2) Load and assert
    conf = load_config()
    paths = get_data_paths(conf)
    assert paths["raw"] == "foo.csv"
