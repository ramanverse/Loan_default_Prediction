"""
Configuration management utility.
Loads and manages application configuration.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, looks for config.yaml in project root.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Default configuration
            self._config = {
                'data': {'train_file': 'application_train.csv'},
                'preprocessing': {'missing_threshold': 0.40},
                'model': {'test_size': 0.2, 'random_state': 42}
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.train_file')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None


# Global config instance
config = Config()
