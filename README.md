# Credit Card Fraud Detection

UE24CS352A Machine Learning mini project. We reimplement "Detecting Credit Card Fraud with Machine Learning" (Rosenbaum, Stanford CS229, 2019) in Python and extend it by tuning the sampling method and minority share p separately for every model (the paper used one p for all models).

Models: Logistic Regression, Logistic Regression with quadratic terms + LASSO, Random Forest, Neural Network (1 hidden layer).
Sampling: none, undersampling, oversampling, both, SMOTE (our Python substitute for ROSE).
Main metric: AUPRC.

## Dataset
Download `creditcard.csv` from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), unzip it, and place it at `data/creditcard.csv`. The file is not in this repository because it is larger than GitHub's file limit.

## Setup (Git Bash on Windows)
    git clone https://github.com/zaheedui/fraud-detection.git
    cd fraud-detection
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt

## How to run
Quick demo (uses saved results, takes seconds):

    python src/demo.py

Full pipeline (slow: the sweeps take from minutes to about an hour each):

    python src/run_logistic.py
    python src/run_quadratic_lasso.py
    python src/run_rf.py
    python src/run_nn.py
    python src/final_test.py

Check the train/validation/test split and the sampling methods:

    python src/data_utils.py
    python src/sampling.py

## Project structure
- `src/data_utils.py`: loads data, stratified 2/3 train, 1/6 validation, 1/6 test split (seed 42)
- `src/sampling.py`: the five sampling methods for any minority share p
- `src/experiment.py`: tries every method and p for a model, saves validation AUPRC
- `src/run_*.py`: one script per model
- `src/final_test.py`: picks the best setup per model, retrains on train + validation, tests once
- `src/demo.py`: prints saved results quickly
- `results/`: validation sweeps, final test table, confusion matrices, ROC/PR plot

## Final test results
| Model | Best setup | AUROC | AUPRC | Sensitivity | F1 |
|---|---|---|---|---|---|
| Logistic | SMOTE, p=0.5 | 0.9689 | 0.7082 | 0.8780 | 0.2466 |
| Quadratic + LASSO | none | 0.9831 | 0.8395 | 0.7927 | 0.8442 |
| Random Forest | SMOTE, p=0.1 | 0.9595 | 0.8623 | 0.8659 | 0.8820 |
| Neural Net | SMOTE, p=0.5 | 0.9813 | 0.6830 | 0.8780 | 0.2034 |

Random Forest performed best. The best sampling method and p differ for every model.

## Team
Syed Syed Zaheed and Laxmikanth .
