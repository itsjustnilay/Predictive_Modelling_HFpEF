# Import necessary libraries for model specification and evaluation
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
from scipy.stats import sem
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Dictionary defining the machine learning models and their hyperparameters.
models = {
    "SVC": SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Lasso": LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42),
    "Elastic Net": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Function to calculate confidence intervals for a given data set.
def calculate_confidence_intervals(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * 1.96  
    return mean, mean - h, mean + h

# Function to evaluate a model, generate performance metrics, and calculate confidence intervals.
def evaluate_and_plot_with_ci(model, model_name, impute_strategy, resample_method, X_train_scaled, y_train_resampled, X_test_scaled, y_test, threshold=0.50, n_bootstrap=1000):
    # Evaluates the model, calculates performance metrics, and performs bootstrapping for confidence intervals.
    metrics_dict = {
        "Model Name": model_name,
        "Imputation Strategy": impute_strategy,
        "Resampling Method": resample_method
    }
    # Cross-validation and model fitting.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) if resample_method is None else 5
    cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv)
    metrics_dict["Average CV Accuracy"] = np.mean(cv_scores)

    # More code for model fitting and performance metrics.
    model.fit(X_train_scaled, y_train_resampled)
    y_probs_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_probs_test >= threshold).astype(int)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    metrics_dict.update({"Test Accuracy": accuracy_test, "Test MCC": mcc_test, "Test F1 Score": f1_test})

    sensitivity_test = recall_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])
    metrics_dict.update({"Test Sensitivity": sensitivity_test, "Test Specificity": specificity_test})

    roc_auc_test = auc(*roc_curve(y_test, y_probs_test)[:2])
    metrics_dict["Test AUC Score"] = roc_auc_test

    metrics_bootstrap = {metric: [] for metric in ["Test Accuracy", "Test MCC", "Test F1 Score", "Test Sensitivity", "Test Specificity", "Test AUC Score"]}

    for _ in range(n_bootstrap):
        X_test_boot, y_test_boot = resample(X_test_scaled, y_test)
        y_probs_test_boot = model.predict_proba(X_test_boot)[:, 1]
        y_pred_test_boot = (y_probs_test_boot >= threshold).astype(int)

        metrics_bootstrap["Test Accuracy"].append(accuracy_score(y_test_boot, y_pred_test_boot))
        metrics_bootstrap["Test MCC"].append(matthews_corrcoef(y_test_boot, y_pred_test_boot))
        metrics_bootstrap["Test F1 Score"].append(f1_score(y_test_boot, y_pred_test_boot))
        metrics_bootstrap["Test Sensitivity"].append(recall_score(y_test_boot, y_pred_test_boot))
        cm_test_boot = confusion_matrix(y_test_boot, y_pred_test_boot)
        metrics_bootstrap["Test Specificity"].append(cm_test_boot[0, 0] / (cm_test_boot[0, 0] + cm_test_boot[0, 1]))
        metrics_bootstrap["Test AUC Score"].append(auc(*roc_curve(y_test_boot, y_probs_test_boot)[:2]))

    for metric, values in metrics_bootstrap.items():
        mean_val, ci_lower, ci_upper = calculate_confidence_intervals(values)
        metrics_dict[f"{metric} CI Lower"] = ci_lower
        metrics_dict[f"{metric} CI Upper"] = ci_upper

    return pd.DataFrame([metrics_dict]), y_probs_test

# Function to perform randomized hyperparameter search using RandomizedSearchCV.
def randomized_search(model, param_distributions, X, y, cv):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=10, cv=cv, random_state=42)
    random_search.fit(X, y)
    return random_search.best_estimator_

imputation_strategies = ['mean', 'median', 'multiple']
resampling_methods = [None, 'random_oversample', 'smote', 'random_undersample', 'balanced']

metrics_list = []
probs_dict = {}

total_iterations = len(models) * len(imputation_strategies) * len(resampling_methods)

param_grid = {
    "SVC": {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10, 100]},
    "Lasso": {'C': [0.01, 0.1, 1, 10, 100]},
    "Elastic Net": {'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]},
    "Random Forest": {'n_estimators': [50, 100, 200, 500], 'max_depth': [None, 10, 20, 30]},
    "HistGradientBoostingClassifier": {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_iter': [100, 200, 300]},
    "XGBoost": {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'n_estimators': [50, 100, 200, 500], 'max_depth': [3, 6, 9]}
}