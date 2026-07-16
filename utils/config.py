"""
Configuration management module for LightGCN Recommender System.
Handles loading and accessing configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the LightGCN Recommender System."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if not self._config:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        # Load environment variables
        load_dotenv()
        
        # Find config file
        config_paths = [
            Path("config/settings.yaml"),
            Path("../config/settings.yaml"),
            Path(__file__).parent.parent / "config" / "settings.yaml",
        ]
        
        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break
        
        if config_file is None:
            raise FileNotFoundError("Configuration file 'settings.yaml' not found")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables if present
        self._override_from_env()
    
    def _override_from_env(self) -> None:
        """Override configuration with environment variables."""
        env_mappings = {
            'DATA_RAW_PATH': ('data', 'raw_data_path'),
            'DATA_PROCESSED_PATH': ('data', 'processed_data_path'),
            'MODEL_EMBEDDING_DIM': ('model', 'embedding_dim'),
            'MODEL_EPOCHS': ('model', 'epochs'),
            'TRAINING_DEVICE': ('training', 'device'),
            'API_PORT': ('api', 'port'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(config_path, value)
    
    def _set_nested_config(self, path: tuple, value: str) -> None:
        """Set nested configuration value."""
        current = self._config
        for key in path[:-1]:
            current = current.setdefault(key, {})
        
        # Try to convert to appropriate type
        try:
            if '.' in value:
                current[path[-1]] = float(value)
            else:
                current[path[-1]] = int(value)
        except ValueError:
            if value.lower() in ('true', 'false'):
                current[path[-1]] = value.lower() == 'true'
            else:
                current[path[-1]] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value by nested keys."""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_section('data')
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get_section('preprocessing')
    
    @property
    def graph(self) -> Dict[str, Any]:
        """Get graph configuration."""
        return self.get_section('graph')
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_section('model')
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get_section('training')
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get_section('evaluation')
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get_section('api')
    
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get_section('database')
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get_section('logging')
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get_section('paths')


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config