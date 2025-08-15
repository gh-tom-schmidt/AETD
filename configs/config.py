import importlib
from pathlib import Path


class Config:
    data = {}

    def __init__(self, file_name="debug_default.pyi"):
        # Remove extension so Python can import it
        module_name = Path(file_name).stem
        config_module = importlib.import_module(module_name)

        # Get all variables that do not start with "__"
        for key, value in vars(config_module).items():
            if not key.startswith("__"):
                Config.data[key.upper()] = value

        # Inject into globals
        globals().update(Config.data)
