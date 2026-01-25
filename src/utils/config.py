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

    # Optional: resolve dataset split profile to train/test paths
    data_cfg = config.get("data", {}) or {}
    active = data_cfg.get("active_split")
    splits = data_cfg.get("splits") or {}
    if active and isinstance(splits, dict) and active in splits:
        profile = splits.get(active) or {}
        train_path = profile.get("train_path")
        test_path = profile.get("test_path")
        if train_path and test_path:
            config.setdefault("data", {})
            config["data"]["train_path"] = train_path
            config["data"]["test_path"] = test_path
            # Ensure downstream pipeline uses the resolved separate files.
            config["data"]["use_separate_files"] = True

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
