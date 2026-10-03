import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)


def get_splits(path="data/creditcard.csv"):
    """Return X_train, y_train, X_val, y_val, X_test, y_test (2/3, 1/6, 1/6)."""
    df = load_data(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2/3 train, 1/3 left over
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1/3, stratify=y, random_state=SEED
    )
    # split the leftover half and half -> 1/6 validation, 1/6 test
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits()
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(name, "rows:", len(y), "fraud:", int(y.sum()))
