import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

import run_logistic
import run_nn
import run_quadratic_lasso
import run_rf
from data_utils import get_splits
from sampling import resample

MODELS = {
    "Logistic": ("logistic", run_logistic.make_model),
    "Quadratic+LASSO": ("quadratic_lasso", run_quadratic_lasso.make_model),
    "Random Forest": ("random_forest", run_rf.make_model),
    "Neural Net": ("neural_net", run_nn.make_model),
}


def best_setup(csv_name):
    df = pd.read_csv("results/" + csv_name + "_val.csv")
    row = df.sort_values("val_auprc", ascending=False).iloc[0]
    p = 0.5 if pd.isna(row["p"]) else float(row["p"])
    return row["method"], p


X_train, y_train, X_val, y_val, X_test, y_test = get_splits()
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

rows = []
cm_text = ""
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

for label, (csv_name, make_model) in MODELS.items():
    method, p = best_setup(csv_name)
    print("Training", label, "with", method, "p =", p, flush=True)

    Xs, ys = resample(X_full, y_full, method, p)
    model = make_model()
    model.fit(Xs, ys)

    scores = model.predict_proba(X_test)[:, 1]
    pred = (scores > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    rows.append({
        "model": label,
        "setup": method if method == "none" else method + " p=" + str(p),
        "AUROC": round(roc_auc_score(y_test, scores), 4),
        "AUPRC": round(average_precision_score(y_test, scores), 4),
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "sensitivity": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "F1": round(f1_score(y_test, pred), 4),
    })
    cm_text += ("%s\n  true normal: predicted normal %d, predicted fraud %d\n"
                "  true fraud:  predicted normal %d, predicted fraud %d\n\n"
                % (label, tn, fp, fn, tp))

    fpr, tpr, _ = roc_curve(y_test, scores)
    prec, rec, _ = precision_recall_curve(y_test, scores)
    ax_roc.plot(fpr, tpr, label=label)
    ax_pr.plot(rec, prec, label=label)

ax_roc.set_title("ROC curve (test set)")
ax_roc.set_xlabel("1 - Specificity")
ax_roc.set_ylabel("Sensitivity")
ax_roc.legend()
ax_pr.set_title("Precision-Recall curve (test set)")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend()
plt.tight_layout()
plt.savefig("results/roc_pr_curves.png", dpi=150)

table = pd.DataFrame(rows)
table.to_csv("results/final_test.csv", index=False)
with open("results/confusion_matrices.txt", "w") as f:
    f.write(cm_text)

print("\nFINAL TEST RESULTS")
print(table.to_string(index=False))
print("\n" + cm_text)
