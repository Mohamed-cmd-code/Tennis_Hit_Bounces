import sys
from src.io_utils import load_ball_data, save_json
from src.inference import predict_file
from src.unsupervised import unsupervised_hit_bounce_detection

from sklearn.metrics import classification_report, accuracy_score


def evaluate_predictions(data):
    """
    Compare ground truth (action) vs predicted (pred_action)
    """
    y_true = []
    y_pred = []

    for frame in data.values():
        if frame.get("visible", False):
            y_true.append(frame["action"])
            y_pred.append(frame["pred_action"])

    print("\n=== Evaluation Metrics ===")
    print(classification_report(y_true, y_pred, digits=2))
    print("Accuracy:", accuracy_score(y_true, y_pred))


def run_unsupervised(json_path, output_path, evaluate=True):
    data, frames = load_ball_data(json_path)
    data = unsupervised_hit_bounce_detection(data, frames)
    save_json(data, output_path)

    if evaluate:
        evaluate_predictions(data)


def run_supervised(json_path, model_path, output_path, evaluate=True):
    predict_file(json_path, model_path, output_path)

    if evaluate:
        data, _ = load_ball_data(output_path)
        evaluate_predictions(data)


if __name__ == "__main__":
    """
    Usage:
    Supervised:
    python main.py supervised data/point.json output/pred.json models/hit_bounce_rf.pkl

    Unsupervised:
    python main.py unsupervised data/point.json output/pred.json
    """

    method = sys.argv[1]       # supervised / unsupervised
    json_path = sys.argv[2]
    output_path = sys.argv[3]

    if method == "unsupervised":
        run_unsupervised(json_path, output_path)

    elif method == "supervised":
        model_path = sys.argv[4]
        run_supervised(json_path, model_path, output_path)

    else:
        raise ValueError("Unknown method: choose supervised or unsupervised")
