import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

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

def check_api_health():
    """Check if API is running and ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            health_status = response.json()
            return health_status.get("status") == "healthy" and \
                   health_status.get("bank_model_loaded") and \
                   health_status.get("telecom_model_loaded")
        return False
    except requests.exceptions.RequestException:
        return False

def get_model_info(model_type: str) -> Dict:
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info/{model_type}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def validate_data(model_type: str, file) -> Dict:
    """Validate uploaded data"""
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        response = requests.post(f"{API_BASE_URL}/validate-data/{model_type}", files=files)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Validation failed")}
    except Exception as e:
        return {"error": str(e)}

def make_predictions(model_type: str, file) -> Dict:
    """Make predictions using the API"""
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        response = requests.post(f"{API_BASE_URL}/predict/{model_type}", files=files)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Prediction failed")}
    except Exception as e:
        return {"error": str(e)}

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
            📋 <strong>Required Columns for Training</strong><br>
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
    
    # Convert to DataFrame for better display
    df_display = pd.DataFrame(predictions[start_idx:end_idx])
    
    # Format the probability column
    if 'Churn_Probability' in df_display.columns:
        df_display['Churn_Probability'] = df_display['Churn_Probability'].apply(lambda x: f"{x:.3f}")
    
    # Display the table
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
    
    # Wait for API to be ready
    with st.spinner("Waiting for API server and models to load... This may take a moment."):
        api_ready = False
        for _ in range(60):  # Wait for up to 60 seconds
            if check_api_health():
                api_ready = True
                break
            time.sleep(1)

    # Check API health
    if not api_ready:
        st.error("🚨 API server is not ready. Please check the FastAPI server logs.")
        st.code("python api.py", language="bash")
        st.stop()
    
    st.success("✅ API server is running and models are loaded!")

    # Sidebar
    st.sidebar.markdown("### 🎯 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Prediction Task",
        options=["bank_churn", "telecom_churn"],
        format_func=lambda x: "🏦 Bank Customer Churn" if x == "bank_churn" else "📱 Telecom Customer Churn"
    )
    
    # Get model information
    model_info = get_model_info(model_type)
    
    if model_info is None:
        st.error("Failed to load model information. Please check the API server.")
        st.stop()
    
    # Display model information
    model_name = "Bank Customer Churn" if model_type == "bank_churn" else "Telecom Customer Churn"
    st.markdown(f'<h2 class="sub-header">🎯 {model_name} Prediction</h2>', unsafe_allow_html=True)
    
    # Display required columns
    display_required_columns(model_info)
    
    # File upload
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer data for churn prediction"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Validate data
        with st.spinner("Validating data..."):
            validation_result = validate_data(model_type, uploaded_file)
        
        if "error" in validation_result:
            st.error(f"Validation error: {validation_result['error']}")
            st.stop()
        
        # Display validation results
        display_validation_results(validation_result)
        
        # If data is valid, proceed with prediction
        if validation_result.get("is_valid", False):
            if st.button("🚀 Make Predictions", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    prediction_result = make_predictions(model_type, uploaded_file)
                
                if "error" in prediction_result:
                    st.error(f"Prediction error: {prediction_result['error']}")
                else:
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    
                    # Summary metrics
                    summary = prediction_result['summary']
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
                    predictions = prediction_result['predictions']
                    display_predictions_table(predictions)
                    
                    # Download button for results
                    df_results = pd.DataFrame(predictions)
                    csv_buffer = io.StringIO()
                    df_results.to_csv(csv_buffer, index=False)
                    
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
        
        else:
            st.warning("Please fix the data validation issues before making predictions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit and FastAPI | Multi-Churn Prediction Platform
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()