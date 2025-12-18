from sklearn.ensemble import RandomForestClassifier




def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    -)