from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from experiment import tune


def make_model():
    # one hidden layer, 3 units, logistic activation (same as the paper)
    return make_pipeline(
        MinMaxScaler(),
        MLPClassifier(hidden_layer_sizes=(3,), activation="logistic",
                      max_iter=200, random_state=42),
    )


if __name__ == "__main__":
    tune("neural_net", make_model)
