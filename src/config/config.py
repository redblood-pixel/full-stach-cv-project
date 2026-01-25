import os
import yaml
from typing import Dict, Any

class Config:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _create_directories(self):
        directories = self.get('directories', [])
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_app_host(self) -> str:
        return self.get('app.host', '0.0.0.0')
    
    def get_app_port(self) -> int:
        return self.get('app.port', 5001)
    
    def get_app_debug(self) -> bool:
        return self.get('app.debug', False)
    
    def get_model_path(self) -> str:
        return self.get('model.path', 'yolov8n.pt')
    
    def get_yandex_disk_url(self) -> str:
        return self.get('model.yandex_disk_url', '')
    
    
    
    def get_uploads_dir(self) -> str:
        return self.get('paths.uploads', 'static/uploads')
    
    def get_results_dir(self) -> str:
        return self.get('paths.results', 'static/results')
    
    def get_reports_dir(self) -> str:
        return self.get('paths.reports', 'reports')
    
    def get_history_file(self) -> str:
        return self.get('paths.history_file', 'request_history.json')
    
    def get_video_frame_interval(self) -> int:
        return self.get('video.frame_interval_seconds', 2)
    
    def get_video_base_frame_interval(self) -> float:
        return self.get('video.base_frame_interval', 2.0)
    
    def get_video_min_frame_interval(self) -> float:
        return self.get('video.min_frame_interval', 0.5)
    
    def get_max_short_video_duration(self) -> int:
        return self.get('video.max_short_video_duration', 30)
    
    def get_min_long_video_duration(self) -> int:
        return self.get('video.min_long_video_duration', 300)
    
    def get_cors_origins(self) -> str:
        return self.get('cors.origins', '*')