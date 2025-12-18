# 🎾 Tennis Hit & Bounce Detection – Roland-Garros 2025

This repository contains a complete solution for detecting **tennis ball hits and bounces** from ball-tracking data extracted from the Roland-Garros 2025 Final.
The project was developed as part of a **Sport Scientist / Computer Vision internship technical test**.

The solution implements **two distinct approaches**:

* **Unsupervised (Physics-Based) Detection**
* **Supervised (Machine Learning) Detection**

Both methods start **only from (x, y) ball trajectories** and produce frame-level predictions.

---

## 📁 Project Structure

```
project_root/
│
├── data/                       # Raw ball-tracking JSON files
├── output/                     # Prediction results (JSON)
├── models/                     # Trained supervised model
│   └── hit_bounce_rf.pkl
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py             # Load / save JSON files
│   ├── feature_extraction.py   # Shared feature engineering
│   ├── dataset.py              # Dataset builder (training)
│   ├── model.py                # ML model definition
│   ├── inference.py            # Supervised inference logic
│   └── unsupervised.py         # Physics-based detection
│
├── train.py                    # Train supervised model
├── predict.py                  # Supervised prediction + metrics
├── main.py                     # Run & compare both methods
├── requirements.txt
└── README.md
```

---

## 🧠 Problem Definition

Each JSON file contains frame-level ball data:

```json
"56100": {
  "x": 894,
  "y": 395,
  "visible": true,
  "action": "air"
}
```

### Goal

Add a new key to every frame:

```json
"pred_action": "hit" | "bounce" | "air"
```

---

## 🔹 Method 1 – Unsupervised (Physics-Based)

### Concept

A tennis ball follows a **smooth parabolic trajectory** in the air.
Any **contact event** (racket hit or ground bounce) introduces:

* Sudden velocity changes
* Acceleration spikes
* Direction inversions (especially vertical for bounces)

### Detection Logic

* Compute velocity and acceleration from (x, y)
* **Bounce** detected when:

  * Vertical velocity changes sign
  * High vertical acceleration
* **Hit** detected when:

  * Large speed or direction change
  * No vertical inversion
* Remaining frames → **air**

### Advantages

* No labels required
* Fully interpretable
* Physics-consistent

### Limitations

* Sensitive to thresholds
* Lower performance on rare events (hit, bounce)

---

## 🔹 Method 2 – Supervised (Machine Learning)

### Concept

Use the provided **ground-truth labels (`action`)** to train a classifier that learns temporal ball dynamics.

### Features Used

* Position: `x`, `y`
* Velocity: `vx`, `vy`
* Acceleration: `ax`, `ay`
* Speed
* Direction change

### Model

**Random Forest Classifier**

Reasons:

* Handles non-linear dynamics
* Robust to noise
* No scaling required
* Interpretable feature importance

Class imbalance handled using:

```python
class_weight='balanced'
```

---

## 📊 Model Evaluation

### Supervised Method – Example Results

```
accuracy: 0.99

Class     Precision  Recall  F1
Air       0.99       1.00    0.99
Bounce    0.75       0.60    0.67
Hit       0.80       0.67    0.73
```

**Interpretation**:

* Excellent overall accuracy due to dominance of `air` frames
* Good detection of rare events given strong class imbalance
* Minor recall loss on bounce/hit due to limited samples

---

### Unsupervised Method – Example Results

```
accuracy: 0.94

Class     Precision  Recall  F1
Air       0.99       0.96    0.97
Bounce    0.25       0.40    0.31
Hit       0.18       0.50    0.26
```

**Interpretation**:

* Strong detection of `air`
* Lower precision on `hit` and `bounce`
* Expected behavior for rule-based detection

---

## 🔍 Supervised vs Unsupervised Comparison

| Aspect               | Supervised | Unsupervised        |
| -------------------- | ---------- | ------------------- |
| Labels needed        | Yes        | No                  |
| Interpretability     | Medium     | High                |
| Accuracy             | Very High  | Medium              |
| Rare event detection | Good       | Limited             |
| Robustness           | High       | Threshold-dependent |

---

## ▶️ How to Run

### Train supervised model

```bash
python train.py
```

### Predict with supervised model

```bash
python main.py supervised data/point_001.json output/pred_sup.json models/hit_bounce_rf.pkl
```

### Predict with unsupervised method

```bash
python main.py unsupervised data/point_001.json output/pred_unsup.json
```

---

## 🏁 Final Notes

This project demonstrates:

* Physics-based reasoning
* Time-series feature engineering
* Supervised ML modeling
* Clean, modular, production-ready code structure
* Objective comparison between approaches

The framework is easily extensible to:

* LSTM / Temporal CNNs
* Event-level consolidation
* Video overlay visualization

---

**Author**: Mohamed ALOUI
**Context**: Sport Scientist Internship – Computer Vision & Data Science
