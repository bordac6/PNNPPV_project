import json

def load_path(dataset, path = '../../config.json'):
    with open(path) as f:
        config = json.load(f)
        return config[dataset]

def load_val_path(path = '../../config.json'):
    with open(path) as f:
        config = json.load(f)
        return config['val_path']