# config.py
import json
from pathlib import Path

def load_hyperparams(path: Path = Path(__file__).parent / "hyperparameters.json") -> dict:
    """
    Load hyperparameter configuration from JSON file.
    Returns a dict with keys: 'FedSGD', 'FedAvg', 'simulation'.
    """
    with open(path, "r") as f:
        return json.load(f)