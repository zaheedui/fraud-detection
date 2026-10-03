from sklearn.ensemble import RandomForestClassifier

from experiment import tune


def make_model():
    return RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)


if __name__ == "__main__":
    tune("random_forest", make_model)
