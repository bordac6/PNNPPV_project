import json

def load_path(path = '../../config.json'):
    with open(path) as file:
        config = json.load(file)
        return config['dataset_path']