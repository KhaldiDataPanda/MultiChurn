import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
import joblib
import os
from typing import Dict, List
import numpy as np

# Import model classes
from train_models import BankChurnModel, TelecomChurnModel

# Page configuration
st.set_page_config(
    page_title="Multi-Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        background-color: #d4edda;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        background-color: #f8d7da;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    try:
        # Train models if they don't exist
        if not os.path.exists('models/bank_churn_model.pkl') or not os.path.exists('models/telecom_churn_model.pkl'):
            st.info("Training models for the first time... This may take a minute.")
            from train_models import train_and_save_models
            train_and_save_models()
        
        bank_model = joblib.load('models/bank_churn_model.pkl')
        telecom_model = joblib.load('models/telecom_churn_model.pkl')
        
        model_info = {
            'bank_churn': {
                'required_columns': bank_model.required_columns,
                'target_column': 'Exited',
                'model_type': 'RandomForestClassifier'
            },
            'telecom_churn': {
                'required_columns': telecom_model.required_columns,
                'target_column': 'Churn',
                'model_type': 'RandomForestClassifier'
            }
        }
        
        return bank_model, telecom_model, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def validate_data(df: pd.DataFrame, required_cols: List[str]) -> Dict:
    """Validate uploaded data"""
    uploaded_cols = df.columns.tolist()
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in uploaded_cols]
    
    # Check for extra columns
    extra_cols = [col for col in uploaded_cols if col not in required_cols]
    
    # Check for missing values in required columns
    available_required_cols = [col for col in required_cols if col in uploaded_cols]
    missing_values = {}
    
    for col in available_required_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_values[col] = int(missing_count)
    
    # Check data types
    type_issues = {}
    for col in available_required_cols:
        if col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'Age', 'CreditScore', 
                   'Balance', 'EstimatedSalary', 'NumOfProducts', 'SeniorCitizen', 'IsActiveMember']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    type_issues[col] = f"Expected numeric, got {df[col].dtype}"
    
    validation_result = {
        "is_valid": len(missing_cols) == 0 and len(missing_values) == 0 and len(type_issues) == 0,
        "data_shape": df.shape,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "missing_values": missing_values,
        "type_issues": type_issues,
        "required_columns": required_cols,
        "uploaded_columns": uploaded_cols
    }
    
    return validation_result

def display_validation_results(validation_result: Dict):
    """Display validation results"""
    if validation_result.get("is_valid", False):
        st.markdown("""
        <div class="success-box">
            ✅ <strong>Data Validation Successful!</strong><br>
            Your data is compatible with the selected model.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", validation_result["data_shape"][0])
        with col2:
            st.metric("Total Columns", validation_result["data_shape"][1])
            
    else:
        st.markdown("""
        <div class="error-box">
            ❌ <strong>Data Validation Failed!</strong><br>
            Please fix the following issues:
        </div>
        """, unsafe_allow_html=True)
        
        if validation_result.get("missing_columns"):
            st.error(f"Missing required columns: {', '.join(validation_result['missing_columns'])}")
        
        if validation_result.get("missing_values"):
            st.error("Missing values found in:")
            for col, count in validation_result["missing_values"].items():
                st.write(f"  - {col}: {count} missing values")
        
        if validation_result.get("type_issues"):
            st.error("Data type issues:")
            for col, issue in validation_result["type_issues"].items():
                st.write(f"  - {col}: {issue}")

def display_required_columns(model_info: Dict):
    """Display required columns for the selected model"""
    if model_info:
        st.markdown("""
        <div class="info-box">
            📋 <strong>Required Columns</strong><br>
            Make sure your CSV file contains the following columns:
        </div>
        """, unsafe_allow_html=True)
        
        required_cols = model_info.get("required_columns", [])
        cols_per_row = 3
        
        for i in range(0, len(required_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(required_cols[i:i+cols_per_row]):
                with cols[j]:
                    st.code(col, language=None)

def display_predictions_table(predictions: List[Dict], page_size: int = 20):
    """Display predictions in a paginated table"""
    total_rows = len(predictions)
    total_pages = (total_rows - 1) // page_size + 1
    
    if total_pages > 1:
        page = st.selectbox(
            "Select Page", 
            range(1, total_pages + 1),
            format_func=lambda x: f"Page {x} of {total_pages}"
        )
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        st.info(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows}")
    else:
        start_idx = 0
        end_idx = total_rows
    
    df_display = pd.DataFrame(predictions[start_idx:end_idx])
    
    if 'Churn_Probability' in df_display.columns:
        df_display['Churn_Probability'] = df_display['Churn_Probability'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(df_display, use_container_width=True)

def create_pie_chart(summary: Dict):
    """Create pie chart for prediction results"""
    labels = ['No Churn', 'Churn']
    values = [summary['predicted_no_churn'], summary['predicted_churn']]
    colors = ['#2E8B57', '#DC143C']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        title={
            'text': 'Churn Prediction Results',
            'x': 0.5,
            'font': {'size': 20}
        },
        font={'size': 14},
        showlegend=True,
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Multi-Churn Prediction Platform</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        bank_model, telecom_model, model_info = load_models()
    
    if bank_model is None or telecom_model is None:
        st.error("Failed to load models. Please check the console for errors.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")

    # Sidebar
    st.sidebar.markdown("### 🎯 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Prediction Task",
        options=["bank_churn", "telecom_churn"],
        format_func=lambda x: "🏦 Bank Customer Churn" if x == "bank_churn" else "📱 Telecom Customer Churn"
    )
    
    # Select appropriate model
    selected_model = bank_model if model_type == "bank_churn" else telecom_model
    selected_info = model_info[model_type]
    
    # Display model information
    model_name = "Bank Customer Churn" if model_type == "bank_churn" else "Telecom Customer Churn"
    st.markdown(f'<h2 class="sub-header">🎯 {model_name} Prediction</h2>', unsafe_allow_html=True)
    
    # Display required columns
    display_required_columns(selected_info)
    
    # File upload
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer data for churn prediction"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Validate data
            with st.spinner("Validating data..."):
                validation_result = validate_data(df, selected_info['required_columns'])
            
            # Display validation results
            display_validation_results(validation_result)
            
            # If data is valid, proceed with prediction
            if validation_result.get("is_valid", False):
                if st.button("🚀 Make Predictions", type="primary", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            predictions = selected_model.predict(df)
                            probabilities = selected_model.predict_proba(df)
                            
                            # Create result dataframe
                            result_df = df.copy()
                            result_df['Predicted_Churn'] = predictions
                            result_df['Churn_Probability'] = probabilities[:, 1]
                            
                            # Calculate summary
                            total_customers = len(predictions)
                            churn_count = int(np.sum(predictions))
                            no_churn_count = total_customers - churn_count
                            churn_percentage = (churn_count / total_customers * 100) if total_customers > 0 else 0
                            
                            summary = {
                                "total_customers": total_customers,
                                "predicted_churn": churn_count,
                                "predicted_no_churn": no_churn_count,
                                "churn_percentage": round(churn_percentage, 2)
                            }
                            
                            # Display results
                            st.markdown("### 📊 Prediction Results")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Customers", summary['total_customers'])
                            with col2:
                                st.metric("Predicted Churn", summary['predicted_churn'])
                            with col3:
                                st.metric("Predicted No Churn", summary['predicted_no_churn'])
                            with col4:
                                st.metric("Churn Rate", f"{summary['churn_percentage']:.1f}%")
                            
                            # Predictions table
                            st.markdown("### 📋 Detailed Predictions")
                            predictions_list = result_df.to_dict('records')
                            display_predictions_table(predictions_list)
                            
                            # Download button
                            csv_buffer = io.StringIO()
                            result_df.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"{model_type}_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Pie chart
                            st.markdown("### 📈 Prediction Distribution")
                            fig = create_pie_chart(summary)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
            else:
                st.warning("Please fix the data validation issues before making predictions.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit | Multi-Churn Prediction Platform
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
