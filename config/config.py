import os
import yaml
from pathlib import Path

class Config:
    """Configuration class for the crack detection project."""
    
    def __init__(self, config_path=None):
        # Default paths
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.annotations_dir = self.data_dir / "annotations"
        self.models_dir = self.project_root / "saved_models"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.raw_data_dir, self.processed_data_dir, 
                          self.annotations_dir, self.models_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Default parameters
        self.image_size = (512, 512)
        self.batch_size = 8
        self.num_workers = 4
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.device = "cuda"  # or "cpu"
        
        # Load from YAML if provided
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def save_to_yaml(self, config_path):
        """Save current configuration to a YAML file."""
        config_dict = {key: value for key, value in self.__dict__.items()
                       if not key.startswith('_') and not callable(value)}
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)