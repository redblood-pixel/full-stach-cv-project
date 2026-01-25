import os
import datetime
import json
import requests
from typing import Optional, Any, Dict

from ultralytics import YOLO

from src.config.config import Config

class ApplicationContext:
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self._history = None
    
    def _download_model(self):
        """
        Download the model from Yandex Disk if it doesn't exist locally.
        """
        model_path = self.config.get_model_path()
        yandex_disk_url = self.config.get_yandex_disk_url()
        
        if not yandex_disk_url:
            raise ValueError("Yandex Disk URL is not configured")
        
        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}")
            return
        
        print(f"Downloading model from {yandex_disk_url} to {model_path}")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            # Download the file
            response = requests.get(yandex_disk_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Model downloaded successfully to {model_path}")
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download model from Yandex Disk: {e}")

    def initialize(self):
        """
        Initialize the application context, including downloading the model if needed.
        """
        # Download model if it doesn't exist
        self._download_model()
        self.model = YOLO(self.config.get_model_path())
    
    def get_model(self) -> YOLO:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self.model
    
    def get_history_file(self) -> str:
        return self.config.get_history_file()
    
    def get_history(self) -> Dict[str, Any]:
        if self._history is None:
            history_file = self.get_history_file()
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        self._history = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading history file {history_file}: {e}")
                    self._history = {"requests": []}
            else:
                self._history = {"requests": []}
        return self._history
    
    def save_history(self, history: Dict[str, Any]) -> None:
        history_file = self.get_history_file()
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            self._history = history
        except IOError as e:
            print(f"Error saving history file {history_file}: {e}")
            raise
    
    def get_uploads_dir(self) -> str:
        return self.config.get_uploads_dir()
    
    def get_results_dir(self) -> str:
        return self.config.get_results_dir()
    
    def get_reports_dir(self) -> str:
        return self.config.get_reports_dir()
    
    
    def get_video_frame_interval(self) -> int:
        return self.config.get_video_frame_interval()
    
    def get_timestamp(self) -> str:
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')