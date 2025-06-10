import yaml
from pathlib import Path
from pydantic import BaseModel
from importlib import resources

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    format: str = 'paInt16'

class VADConfig(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 1000

class WhisperConfig(BaseModel):
    model: str = 'base.en'

class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    whisper: WhisperConfig = WhisperConfig()
    paste_delay: float = 0.2
    stopword: str = 'stop'

def load_config(config_file: Path = None) -> AppConfig:
    with resources.files('voice2cursor').joinpath('default_config.yaml').open('r') as f:
        default_config_data = yaml.safe_load(f)

    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            user_config_data = yaml.safe_load(f)
        
        # Simple merge - user config overrides default
        merged_config = default_config_data.copy()
        for key, value in user_config_data.items():
            if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        return AppConfig.parse_obj(merged_config)
    
    return AppConfig.parse_obj(default_config_data)

# A global config instance for convenience, to be initialized by the CLI entry point.
config: AppConfig = None

def set_config(new_config: AppConfig):
    global config
    config = new_config 