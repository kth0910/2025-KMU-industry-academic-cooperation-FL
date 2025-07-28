import argparse
import json

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/fedavg_config.json", help="Path to config JSON file")
    parser.add_argument('--cid', type=int, default=0, help="Client ID (for client.py only)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    
    config["cid"] = args.cid  # client ID 추가
    return config
