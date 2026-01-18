import tomllib
from pathlib import Path
from typing import Optional


def load_config(path: Optional[Path] = None) -> dict:
    """
    Load configuration from TOML file.
    
    Searches in order:
    1. Provided path
    2. ./config.toml
    3. ~/.config/idris/config.toml
    """
    if path is None:
        search_paths = [
            Path("config.toml"),
            Path.home() / ".config" / "idris" / "config.toml",
        ]
        for p in search_paths:
            if p.exists():
                path = p
                break
    
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "Config file not found. Create config.toml with required settings:\n"
            "[paths]\n"
            "llvm_test_dir = '/path/to/llvm/test'\n"
            "output_dir = './output'\n"
            "opt = '/path/to/opt'\n"
            "alive_tv = '/path/to/alive-tv'\n\n"
            "[fuzzer]\n"
            "model = 'codellama/CodeLlama-7b-hf'\n"
            "batch_size = 32\n"
            "num_iterations = 100\n"
            "temperature = 0.8\n"
            "max_tokens = 512\n"
            "num_workers = 4\n"
        )
    
    with open(path, "rb") as f:
        return tomllib.load(f)


# Global config singleton
_CONFIG: Optional[dict] = None


def get_config() -> dict:
    """Get the global configuration (lazy loaded)"""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def set_config(config: dict) -> None:
    """Set the global configuration (for testing)"""
    global _CONFIG
    _CONFIG = config