import joblib
from .feature_extraction import extract_features
from .io_utils import load_ball_data, save_json




def predict_file(json_path, model_path, output_path):
    model = joblib.load(model_path)
    data, frames = load_ball_data(json_path)


    for f in frames:
        data[f]['pred_action'] = 'air'


    X, valid_frames = extract_features(data, frames)
    if X is None:
        save_json(data, output_path)
        return


    preds = model.predict(X)
    for f, p in zip(valid_frames, preds):
        data[f]['pred_action'] = p


    save_json(data, output_path)