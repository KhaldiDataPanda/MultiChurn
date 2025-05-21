# Customer Churn Analysis

This repository contains analyses of customer churn in two different sectors: Banking and Telecommunications. My main aim was to explore the Churn problem 

## Projects

The repository is divided into two main projects:

1.  **Bank Customer Churn (`BanckChurn.ipynb`)**:
    *   Analyzes customer data from `Banking.csv` to predict bank customer churn.
    *   Likely involves data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model training and evaluation.

2.  **Telecom Customer Churn (`TelecomChurn.ipynb`)**:
    *   Analyzes customer data from `Telecom.csv` to predict telecom customer churn.
    *   Involves steps such as:
        *   Data loading and initial exploration.
        *   Data cleaning and preprocessing (e.g., dropping irrelevant columns like `customerID`).
        *   Visualization of churn distribution and relationships between features (e.g., tenure, monthly charges, contract type, internet service) and churn.
        *   Encoding categorical features (e.g., using TargetEncoder).
        *   Correlation analysis.
        *   Data scaling and splitting for model training.
        *   Training and evaluating multiple classification models (SVM, Random Forest, XGBoost).
        *   Analyzing model performance using classification reports and confusion matrices.
        *   Visualizing feature importances.

## Datasets

*   `Banking.csv`: Contains data related to bank customers.
*   `Telecom.csv`: Contains data related to telecom customers, including demographics, account information, services, and churn status.

## Key Libraries Used (across both projects, inferred from `TelecomChurn.ipynb`)

*   **Data Manipulation**: pandas, numpy
*   **Data Visualization**: seaborn, matplotlib
*   **Data Analysis/Utility**: klib
*   **Machine Learning**:
    *   scikit-learn (for preprocessing, model selection, metrics, SVM, RandomForestClassifier)
    *   xgboost (for XGBClassifier)
    *   category_encoders (for TargetEncoder)

## How to Use

1.  Ensure you have Python and the necessary libraries installed. You can typically install them using pip:
    ```bash
    pip install pandas numpy seaborn matplotlib klib scikit-learn xgboost category_encoders jupyter
    ```
2.  Clone this repository or download the files.
3.  Open the Jupyter Notebook files (`BanckChurn.ipynb`, `TelecomChurn.ipynb`) in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension).
4.  Run the cells in the notebooks to execute the analysis. Make sure the corresponding CSV files (`Banking.csv`, `Telecom.csv`) are in the same directory as the notebooks.

## Repository Structure

```
.
├── BanckChurn.ipynb
├── Banking.csv
├── readme.md
├── Telecom.csv
└── TelecomChurn.ipynb
```
