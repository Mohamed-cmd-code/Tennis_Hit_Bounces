import numpy as np




def filter_visible_frames(data, frames):
    return [
        f for f in frames
        if data[f].get('visible', False)
        and data[f].get('x') is not None
        and data[f].get('y') is not None
    ]




def extract_features(data, frames):
    frames = filter_visible_frames(data, frames)
    if len(frames) < 5:
        return None, frames


    x = np.array([data[f]['x'] for f in frames], dtype=float)
    y = np.array([data[f]['y'] for f in frames], dtype=float)


    vx = np.gradient(x)
    vy = np.gradient(y)
    ax = np.gradient(vx)
    ay = np.gradient(vy)


    speed = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)
    angle_change = np.gradient(angle)


    X = np.column_stack([
        x, y, vx, vy, ax, ay, speed, angle_change
    ])


    return X, frames