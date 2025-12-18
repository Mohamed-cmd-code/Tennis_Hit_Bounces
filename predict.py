from src.inference import predict_file
from src.io_utils import load_ball_data
from sklearn.metrics import classification_report

JSON_PATH = 'ball_data_19.json'
MODEL_PATH = 'models/hit_bounce_rf.pkl'
OUTPUT_PATH = 'output/point_001_pred.json'

predict_file(JSON_PATH, MODEL_PATH, OUTPUT_PATH)

# testing
data, frames = load_ball_data(OUTPUT_PATH)

y_true, y_pred = [], []

for f in frames:
    if 'action' in data[f]:
        y_true.append(data[f]['action'])
        y_pred.append(data[f]['pred_action'])

if len(y_true) > 0:
    print("\n=== Prediction Performance ===")
    print(classification_report(y_true, y_pred))
