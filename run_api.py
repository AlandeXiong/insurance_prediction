"""Run API server"""
import uvicorn
from src.utils.config import load_config

if __name__ == "__main__":
    config = load_config()
    api_config = config.get('api', {})
    
    uvicorn.run(
        "src.api.app:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', False)
    )
