import tomllib
from pathlib import Path

def load_config(path=None):
    if path is None:
        for p in [Path("config.toml"), Path.home() / ".config.toml"]:
            if p.exists():
                path = p
                break
    
    if path is None or not Path(path).exists():
        raise FileNotFoundError("Config file not found. Create config.toml")
    
    with open(path, "rb") as f:
        return tomllib.load(f)

CONFIG = None

def get_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config()
    return CONFIG