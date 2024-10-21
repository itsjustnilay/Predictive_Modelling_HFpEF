# Predicting 30-Day and 1-Year Mortality in Heart Failure with Preserved Ejection Fraction (HFpEF)

This repository contains Python scripts used for the paper *"Predicting 30-Day and 1-Year Mortality in Heart Failure with Preserved Ejection Fraction (HFpEF)."* The data supporting this study's findings are openly available in [PhysioNet](https://doi.org/10.13026/hxp0-hg59) (Johnson et al., 2024). Follow our preprint [here](https://www.medrxiv.org/content/10.1101/2024.10.15.24315524v1) for further details.

Below is a concise description of each file in the repository:

preprocessing.py: Handles data preprocessing tasks such as imputation of missing values, oversampling/undersampling, and feature transformations. These transformations are essential for model readiness.

Below is a concise description of each file in the repository:

- **preprocessing.py**: 
  Handles data preprocessing tasks such as imputation of missing values, oversampling/undersampling, and feature transformations. These transformations are essential for model readiness.

- **model_specification_evaluation.py**: 
  Defines the machine learning models used in the pipeline and functions for evaluating them with cross-validation. Also includes functionality for calculating confidence intervals for model metrics.

- **main_execution.py**: 
  The main execution file that brings everything together. It runs multiple combinations of models, imputation strategies, and resampling methods, evaluates the performance, and stores results.

### Required Libraries
To install the necessary libraries, use the `requirements.txt` file included in the repository:

```bash
pip install -r requirements.txt
```

### Reference
Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.0) [Data set]. PhysioNet. https://doi.org/10.13026/hxp0-hg59


