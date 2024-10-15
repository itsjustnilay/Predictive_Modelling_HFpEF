# Import necessary libraries for preprocessing tasks
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Function to handle data imputation based on a specified strategy (mean, median, multiple).
def impute_data(X, strategy='mean'):
    if strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == 'multiple':
        imputer = IterativeImputer(max_iter=10, random_state=42)
    else:
        raise ValueError("Unsupported imputation strategy.")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Function to oversample the minority class using either random oversampling or SMOTE.
def oversample_data(X, y, method='random'):
    if method == 'random':
        oversampler = RandomOverSampler(random_state=42)
    elif method == 'smote':
        oversampler = SMOTE(random_state=42)
    return oversampler.fit_resample(X, y)

# Function to undersample the majority class using random undersampling.
def undersample_data(X, y, method='random'):
    undersampler = RandomUnderSampler(random_state=42)
    return undersampler.fit_resample(X, y)

# Function that balances data by both oversampling and undersampling using SMOTE-ENN.
def balanced_oversample_undersample_data(X, y):
    smote_enn = SMOTEENN(random_state=42)
    return smote_enn.fit_resample(X, y)

# Function to apply data transformations (power, quantile, and standard scaling) to specific features.
def apply_transformations(X):
    pt = PowerTransformer(method='yeo-johnson')
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X), 1000))
    transformations = ['creatinine', 'platelet_count', 'wbc_count']
    for feature in transformations:
        X[feature] = pt.fit_transform(X[feature].values.reshape(-1, 1))
    X['inr'] = qt.fit_transform(X['inr'].values.reshape(-1, 1))
    X['bmi'] = (X['bmi'] - np.mean(X['bmi'])) / np.std(X['bmi'])
    X['oxygen_saturation'] = qt.fit_transform(X['oxygen_saturation'].values.reshape(-1, 1))
    return X