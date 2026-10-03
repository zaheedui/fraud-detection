from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

SEED = 42
METHODS = ["none", "under", "over", "both", "smote"]
P_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def resample(X, y, method, p):
    """Return a resampled training set where fraud is about a fraction p of rows."""
    r = p / (1 - p)  # fraud-to-normal ratio the library expects

    if method == "none":
        return X, y
    if method == "under":
        return RandomUnderSampler(sampling_strategy=r, random_state=SEED).fit_resample(X, y)
    if method == "over":
        return RandomOverSampler(sampling_strategy=r, random_state=SEED).fit_resample(X, y)
    if method == "smote":
        return SMOTE(sampling_strategy=r, random_state=SEED).fit_resample(X, y)
    if method == "both":
        X1, y1 = RandomOverSampler(sampling_strategy=r / 2, random_state=SEED).fit_resample(X, y)
        return RandomUnderSampler(sampling_strategy=r, random_state=SEED).fit_resample(X1, y1)
    raise ValueError("unknown method: " + method)


if __name__ == "__main__":
    from data_utils import get_splits

    X_train, y_train, _, _, _, _ = get_splits()
    for m in METHODS:
        Xs, ys = resample(X_train, y_train, m, 0.02)
        print(m, "rows:", len(ys), "fraud:", int(ys.sum()),
              "fraud share: %.2f%%" % (100 * ys.mean()))
