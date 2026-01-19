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
