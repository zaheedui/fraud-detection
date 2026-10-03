from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from experiment import tune


def make_model():
    # all squared and interaction terms (30 features -> 495), then L1 (LASSO) penalty
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=200),
    )


if __name__ == "__main__":
    tune("quadratic_lasso", make_model)
