"""Configuration loader"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve data split strategy (3 supported):
    # - pre_split: use pre-split train/test files on disk
    # - timecut: split the original source file by date cutoff
    # - ratio: random split the original source file by test_size (optionally stratified)
    #
    # Also supports legacy configs (split_strategy/active_split/etc.) and maps them to these 3.
    data_cfg = config.get("data", {}) or {}

    # Legacy mapping (best-effort)
    if "split_strategy" in data_cfg or "active_split" in data_cfg or "use_separate_files" in data_cfg:
        legacy = str(data_cfg.get("split_strategy", "auto")).lower().strip()
        if legacy in {"profile", "active_split"}:
            active = data_cfg.get("active_split")
            splits = data_cfg.get("splits") or {}
            profile = (splits.get(active) or {}) if isinstance(splits, dict) else {}
            config.setdefault("data", {})
            config["data"]["strategy"] = "pre_split"
            config["data"].setdefault("pre_split", {})
            config["data"]["pre_split"]["train_path"] = profile.get("train_path", data_cfg.get("train_path"))
            config["data"]["pre_split"]["test_path"] = profile.get("test_path", data_cfg.get("test_path"))
        elif legacy in {"files", "pre_split", "separate_files"} or bool(data_cfg.get("use_separate_files", False)):
            config.setdefault("data", {})
            config["data"]["strategy"] = "pre_split"
            config["data"].setdefault("pre_split", {})
            config["data"]["pre_split"]["train_path"] = data_cfg.get("train_path")
            config["data"]["pre_split"]["test_path"] = data_cfg.get("test_path")
        elif legacy in {"single_file", "split"}:
            # Old single_file behavior was a random split
            config.setdefault("data", {})
            config["data"]["strategy"] = "ratio"
            config["data"].setdefault("ratio", {})
            config["data"]["source_path"] = data_cfg.get("source_path") or data_cfg.get("train_path")
            if data_cfg.get("test_size") is not None:
                config["data"]["ratio"]["test_size"] = data_cfg.get("test_size")
            if data_cfg.get("stratify") is not None:
                config["data"]["ratio"]["stratify"] = data_cfg.get("stratify")

        # refresh local view
        data_cfg = config.get("data", {}) or {}

    strategy = str(data_cfg.get("strategy", "pre_split")).lower().strip()
    config.setdefault("data", {})

    if strategy == "pre_split":
        ps = data_cfg.get("pre_split", {}) or {}
        train_path = ps.get("train_path")
        test_path = ps.get("test_path")
        if not train_path or not test_path:
            raise ValueError("data.strategy='pre_split' requires data.pre_split.train_path and data.pre_split.test_path")
        config["data"]["resolved_split"] = {
            "strategy": "pre_split",
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        # Convenience aliases (some scripts historically read these)
        config["data"]["train_path"] = train_path
        config["data"]["test_path"] = test_path

    elif strategy == "timecut":
        source_path = data_cfg.get("source_path")
        tc = data_cfg.get("timecut", {}) or {}
        date_col = tc.get("date_col", "Effective To Date")
        cutoff = tc.get("cutoff")
        if not source_path:
            raise ValueError("data.strategy='timecut' requires data.source_path")
        if not cutoff:
            raise ValueError("data.strategy='timecut' requires data.timecut.cutoff")
        config["data"]["resolved_split"] = {
            "strategy": "timecut",
            "source_path": str(source_path),
            "date_col": str(date_col),
            "cutoff": str(cutoff),
        }

    elif strategy == "ratio":
        source_path = data_cfg.get("source_path")
        ra = data_cfg.get("ratio", {}) or {}
        test_size = ra.get("test_size", 0.2)
        stratify = ra.get("stratify", True)
        random_state = ra.get("random_state", data_cfg.get("random_state", 42))
        if not source_path:
            raise ValueError("data.strategy='ratio' requires data.source_path")
        config["data"]["resolved_split"] = {
            "strategy": "ratio",
            "source_path": str(source_path),
            "test_size": float(test_size),
            "stratify": bool(stratify),
            "random_state": int(random_state),
        }

    else:
        raise ValueError("Unknown data.strategy. Supported: 'pre_split' | 'timecut' | 'ratio'")

    return config


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get and create necessary directories"""
    paths = config.get('paths', {})
    base_dir = Path.cwd()
    
    dirs = {
        'models': base_dir / paths.get('models_dir', 'models'),
        'outputs': base_dir / paths.get('outputs_dir', 'outputs'),
        'logs': base_dir / paths.get('logs_dir', 'logs'),
    }
    
    # Create directories if they don't exist
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs
