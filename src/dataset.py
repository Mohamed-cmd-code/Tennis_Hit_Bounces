import os
import numpy as np
from .io_utils import load_ball_data
from .feature_extraction import extract_features




def build_dataset(data_folder):
    X_all, y_all = [], []


    for file in os.listdir(data_folder):
        if not file.endswith('.json'):
            continue


        data, frames = load_ball_data(os.path.join(data_folder, file))
        X, valid_frames = extract_features(data, frames)


        if X is None:
            continue


        y = [data[f]['action'] for f in valid_frames]
        X_all.append(X)
        y_all.append(y)


    return np.vstack(X_all), np.concatenate(y_all)