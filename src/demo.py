import pandas as pd

from data_utils import get_splits

MODELS = ["logistic", "quadratic_lasso", "random_forest", "neural_net"]

print("=" * 60)
print("1. DATA SPLIT (train / validation / test)")
print("=" * 60)
X_train, y_train, X_val, y_val, X_test, y_test = get_splits()
for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(name, "rows:", len(y), "fraud:", int(y.sum()),
          "fraud share: %.2f%%" % (100 * y.mean()))

print("\n" + "=" * 60)
print("2. BEST VALIDATION SETUP PER MODEL (stage 1 tournament)")
print("=" * 60)
for m in MODELS:
    df = pd.read_csv("results/" + m + "_val.csv")
    print("\n" + m)
    print(df.sort_values("val_auprc", ascending=False).head(3).to_string(index=False))

print("\n" + "=" * 60)
print("3. FINAL TEST RESULTS (stage 2)")
print("=" * 60)
print(pd.read_csv("results/final_test.csv").to_string(index=False))

print("\n" + "=" * 60)
print("4. CONFUSION MATRICES")
print("=" * 60)
print(open("results/confusion_matrices.txt").read())

print("ROC / precision-recall plot saved at: results/roc_pr_curves.png")
