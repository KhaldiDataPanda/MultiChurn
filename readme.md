# Multi-Churn Prediction Platform

A comprehensive machine learning platform for predicting customer churn in Banking and Telecom industries using multiple ML models with a user-friendly Streamlit interface and FastAPI backend.

## 🎯 Features

- **Dual Prediction Models**: Bank Customer Churn & Telecom Customer Churn
- **Interactive Web Interface**: Built with Streamlit for easy data upload and visualization
- **RESTful API**: FastAPI backend for scalable predictions
- **Data Validation**: Automatic validation of uploaded data against model requirements
- **Paginated Results**: Handle large datasets with paginated table views (max 20 rows per page)
- **Export Functionality**: Download prediction results as CSV
- **Visual Analytics**: Interactive pie charts for prediction distribution
- **Real-time Feedback**: Instant validation and prediction results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

First, train and save the machine learning models:

```bash
python train_models.py
```

This will create a `models/` directory with:
- `bank_churn_model.pkl` - Bank churn prediction model
- `telecom_churn_model.pkl` - Telecom churn prediction model
- `model_info.json` - Model metadata and requirements

### 3. Start FastAPI Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 4. Launch Streamlit Interface

In a new terminal:

```bash
streamlit run streamlit_app.py
```

The web interface will open in your browser at `http://localhost:8501`

## 📊 Model Requirements

### Bank Churn Model
Required columns:
- `Age` - Customer age (numeric)
- `IsActiveMember` - Active membership status (0/1)
- `Gender` - Customer gender (categorical)
- `NumOfProducts` - Number of products (numeric)
- `CreditScore` - Credit score (numeric)
- `Balance` - Account balance (numeric)
- `Geography` - Customer location (categorical)
- `Tenure` - Years with bank (numeric)
- `EstimatedSalary` - Estimated salary (numeric)

### Telecom Churn Model
Required columns:
- `tenure` - Months with company (numeric)
- `MonthlyCharges` - Monthly charges (numeric)
- `TotalCharges` - Total charges (numeric)
- `gender` - Customer gender (categorical)
- `SeniorCitizen` - Senior citizen status (0/1)
- `Partner` - Has partner (categorical)
- `Dependents` - Has dependents (categorical)
- `PhoneService` - Has phone service (categorical)
- `InternetService` - Internet service type (categorical)
- `OnlineSecurity` - Online security service (categorical)
- `OnlineBackup` - Online backup service (categorical)
- `DeviceProtection` - Device protection service (categorical)
- `TechSupport` - Tech support service (categorical)
- `StreamingTV` - Streaming TV service (categorical)
- `StreamingMovies` - Streaming movies service (categorical)
- `Contract` - Contract type (categorical)
- `PaperlessBilling` - Paperless billing (categorical)
- `PaymentMethod` - Payment method (categorical)

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


## How to Use

1.  Ensure you have Python and the necessary libraries installed. You can typically install them using pip:
    ```bash
    pip install pandas numpy seaborn matplotlib klib scikit-learn xgboost category_encoders jupyter
    ```
2.  Clone this repository or download the files.
3.  Open the Jupyter Notebook files (`BanckChurn.ipynb`, `TelecomChurn.ipynb`) in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension).
4.  Run the cells in the notebooks to execute the analysis. Make sure the corresponding CSV files (`Banking.csv`, `Telecom.csv`) are in the same directory as the notebooks.


## 🎨 Web Interface Features

### Data Upload & Validation
- Drag-and-drop CSV file upload
- Real-time data validation
- Clear error messages for missing columns, data types, and missing values
- Display of required columns for each model

### Prediction Results
- Comprehensive summary statistics
- Paginated table view (20 rows per page)
- Probability scores for each prediction
- CSV download functionality

### Visualizations
- Interactive pie chart showing churn distribution
- Responsive design for different screen sizes



## 📈 Performance

- **Bank Model**: ~85% accuracy on test set
- **Telecom Model**: ~80% accuracy on test set
- **API Response Time**: < 2 seconds for typical datasets
- **Frontend**: Real-time updates and responsive interface

## Usage Examples

### Data Format Examples

#### Bank Churn CSV Format:
```csv
Age,IsActiveMember,Gender,NumOfProducts,CreditScore,Balance,Geography,Tenure,EstimatedSalary
42,1,Female,2,619,0.00,France,2,101348.88
41,1,Female,1,608,83807.86,Spain,1,112542.58
```

#### Telecom Churn CSV Format:
```csv
gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
```

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the API documentation at `http://localhost:8000/docs`
3. Run `python test_setup.py` to verify your setup
4. Open an issue on GitHub
