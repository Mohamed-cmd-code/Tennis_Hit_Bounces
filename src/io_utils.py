import json


def load_ball_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = sorted(data.keys(), key=lambda x: int(x))
    return data, frames

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)