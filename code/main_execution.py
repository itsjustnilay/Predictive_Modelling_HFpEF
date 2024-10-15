# Import necessary libraries for main execution and iteration over models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from preprocessing import impute_data, oversample_data, undersample_data, balanced_oversample_undersample_data, apply_transformations
from model_specification_evaluation import models, randomized_search, evaluate_and_plot_with_ci, imputation_strategies, resampling_methods, param_grid

# Setting a random seed for reproducibility.
np.random.seed(42)

# Data encoding and splitting for training and testing.
encoded_df = pd.get_dummies(norace, columns=['gender'], prefix='cat')
X = encoded_df.drop(['death_30_days', 'death_1_year'], axis=1)
y = encoded_df['death_1_year']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Main pipeline for iterating over models, imputing strategies, resampling methods, and evaluating performance.
with tqdm(total=total_iterations) as pbar:
    for model_name, model in models.items():
        for impute_strategy in imputation_strategies:
            for resample_method in resampling_methods:

                X_train_prep = impute_data(apply_transformations(X_train), strategy=impute_strategy)
                X_test_prep = impute_data(apply_transformations(X_test), strategy=impute_strategy)
                
                # Resampling the data based on the selected method.
                X_train_resampled, y_train_resampled = X_train_prep, y_train
                if resample_method == 'random_oversample':
                    X_train_resampled, y_train_resampled = oversample_data(X_train_prep, y_train, method='random')
                elif resample_method == 'smote':
                    X_train_resampled, y_train_resampled = oversample_data(X_train_prep, y_train, method='smote')
                elif resample_method == 'random_undersample':
                    X_train_resampled, y_train_resampled = undersample_data(X_train_prep, y_train, method='random')
                elif resample_method == 'balanced':
                    X_train_resampled, y_train_resampled = balanced_oversample_undersample_data(X_train_prep, y_train)
                
                # Scaling data for linear models and logistic regression.
                if model_name in ["Logistic Regression", "SVC", "Lasso", "Elastic Net"]:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_resampled)
                    X_test_scaled = scaler.transform(X_test_prep)
                else:
                    X_train_scaled = X_train_resampled
                    X_test_scaled = X_test_prep

                # Apply RandomizedSearchCV and evaluate the model.
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) if resample_method is None else 5
                best_model = randomized_search(model, param_grid[model_name], X_train_scaled, y_train_resampled, cv=cv)

                metrics_df, y_probs_test = evaluate_and_plot_with_ci(best_model, model_name, impute_strategy, resample_method, X_train_scaled, y_train_resampled, X_test_scaled, y_test)
                metrics_list.append(metrics_df)
                key = f"{model_name}_{impute_strategy}_{resample_method}"
                probs_dict[key] = y_probs_test

                pbar.update(1)

# Combining all metrics into a single DataFrame.
metrics_combined_df = pd.concat(metrics_list, ignore_index=True)