import json

def load_path(path = '../tools/config.json'):
    with open(path) as file:
        config = json.load(file)
        return config['dataset_path']