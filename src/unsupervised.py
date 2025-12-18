import numpy as np
from .feature_extraction import extract_features

def unsupervised_hit_bounce_detection(data, frames):
    X, valid_frames = extract_features(data, frames)

    if X is None:
        return data

    vx, vy = X[:,2], X[:,3]
    ax, ay = X[:,4], X[:,5]
    speed = X[:,6]

    speed_change = np.abs(np.gradient(speed))
    ay_abs = np.abs(ay)

    bounce_idx = (
        (vy[:-2] > 0) &
        (vy[2:] < 0) &
        (ay_abs[1:-1] > np.percentile(ay_abs, 95))
    )
    bounce_idx = np.pad(bounce_idx, (1,1))

    hit_idx = (
        speed_change > np.percentile(speed_change, 95)
    ) & (~bounce_idx)

    for f in frames:
        data[f]['pred_action'] = 'air'

    for i, f in enumerate(valid_frames):
        if bounce_idx[i]:
            data[f]['pred_action'] = 'bounce'
        elif hit_idx[i]:
            data[f]['pred_action'] = 'hit'

    return data
