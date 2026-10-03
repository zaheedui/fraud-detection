from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import tune


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


if __name__ == "__main__":
    tune("logistic", make_model)
