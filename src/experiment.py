import os
import time

import pandas as pd
from sklearn.metrics import average_precision_score

from data_utils import get_splits
from sampling import resample, METHODS, P_GRID


def tune(model_name, make_model):
    """Try every sampling method and p, score on the validation set (AUPRC)."""
    X_train, y_train, X_val, y_val, _, _ = get_splits()
    rows = []
    for method in METHODS:
        # "none" does not use p, so run it only once
        p_list = [None] if method == "none" else P_GRID
        for p in p_list:
            start = time.time()
            p_arg = 0.5 if p is None else p  # p is ignored for "none"
            Xs, ys = resample(X_train, y_train, method, p_arg)
            model = make_model()
            model.fit(Xs, ys)
            scores = model.predict_proba(X_val)[:, 1]
            auprc = average_precision_score(y_val, scores)
            rows.append({"method": method, "p": p, "val_auprc": round(auprc, 4)})
            print(method, p, "AUPRC:", round(auprc, 4),
                  "(%.0fs)" % (time.time() - start), flush=True)

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/" + model_name + "_val.csv", index=False)

    print("\nTop 5 combinations for", model_name)
    print(results.sort_values("val_auprc", ascending=False).head(5).to_string(index=False))
    return results
