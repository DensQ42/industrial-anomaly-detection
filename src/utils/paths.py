from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

def get_model_path(filename):
    return MODELS_DIR / filename

def get_data_path(filename, subdir='processed'):
    return DATA_DIR / subdir / filename