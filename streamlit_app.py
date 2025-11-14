import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt

# Set page configuration with favicon
st.set_page_config(
    page_title='Data Visualization Dashboard',
    page_icon=':chart_with_upwards_trend:',
    layout='wide'
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None
if 'value_column' not in st.session_state:
    st.session_state.value_column = None
if 'data_warnings' not in st.session_state:
    st.session_state.data_warnings = []

def validate_and_clean_data(df):
    """Validate dataset and clean inappropriate data elements"""
    warnings = []
    
    # Check for null values
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        warnings.append(f"Dataset contains null values in columns: {', '.join(columns_with_nulls.index.tolist())}. These rows will be skipped in visualization.")
    
    # Check for inappropriate data types
    for col in df.columns:
        # Try to detect if column can be parsed as date
        try:
            pd.to_datetime(df[col], errors='raise')
            # If successful, this is a potential date column
        except (ValueError, TypeError):
            pass
    
    return warnings

def detect_date_columns(df):
    """Detect columns that can be parsed as dates"""
    date_columns = []
    
    for col in df.columns:
        # Skip columns with too many nulls
        if df[col].isnull().sum() / len(df) > 0.5:
            continue
            
        # Try to parse as datetime with different approaches
        try:
            # First try with infer_datetime_format
            parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if parsed.notnull().sum() > len(df) * 0.7:  # At least 70% valid dates
                date_columns.append(col)
        except:
            continue
            
        try:
            # Try with dayfirst assumption
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            if parsed.notnull().sum() > len(df) * 0.7:
                if col not in date_columns:
                    date_columns.append(col)
        except:
            continue
    
    return date_columns

def detect_numerical_columns(df):
    """Detect numerical columns, excluding dates"""
    numerical_columns = []
    
    for col in df.columns:
        # Skip if column is already identified as date
        try:
            pd.to_datetime(df[col], errors='coerce')
            continue
        except:
            pass
        
        # Check if column can be converted to numeric
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.notnull().sum() > len(df) * 0.7:  # At least 70% numeric values
                numerical_columns.append(col)
        except:
            continue
    
    return numerical_columns

def load_data(file):
    """Load data from uploaded file with validation"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files only.")
            return None
        
        if df.empty:
            st.error("Uploaded file is empty. Please upload a file with data.")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Create sidebar menu (accessible via upper-right toggle button)
with st.sidebar:
    st.header("📋 Data Configuration")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload Dataset", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.session_state.df = df
            
            # Validate data and get warnings
            warnings = validate_and_clean_data(df)
            st.session_state.data_warnings = warnings
            
            # Show warnings if any
            for warning in warnings:
                st.warning(warning)
            
            # Detect date columns
            date_columns = detect_date_columns(df)
            
            if not date_columns:
                st.error("❌ No valid date columns found in the dataset. Please ensure your data contains a column with date/time values.")
            else:
                st.session_state.date_column = st.selectbox(
                    "📅 Select Date Column (X-axis)",
                    options=date_columns,
                    help="Choose the column containing date/time values for the X-axis"
                )
            
            # Detect numerical columns
            numerical_columns = detect_numerical_columns(df)
            
            if not numerical_columns:
                st.error("❌ No valid numerical columns found. Please ensure your data contains at least one column with numeric values.")
            else:
                st.session_state.value_column = st.selectbox(
                    "📈 Select Value Column (Y-axis)",
                    options=numerical_columns,
                    help="Choose the column containing numeric values for the Y-axis"
                )
            
            if st.button("🚀 Generate Chart", type="primary", use_container_width=True):
                if st.session_state.date_column and st.session_state.value_column:
                    st.session_state.data_loaded = True
                    st.success("✅ Chart generated successfully!")
                else:
                    st.error("❌ Please select both date and value columns.")
    
    st.markdown("---")
    st.markdown("### ℹ️ Requirements")
    st.markdown("""
    - **Date Column**: Must contain date/time values
    - **Value Column**: Must contain numerical data
    - **Data Quality**: Null values will be skipped automatically
    - **File Types**: CSV or Excel (.xlsx, .xls)
    """)

# Main content area
st.title('📊 Data Visualization Dashboard')
st.markdown('''
<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
    <h3>👋 Welcome to the Dashboard!</h3>
    <p>Click the <strong>≣ menu button</strong> in the upper-right corner to upload your dataset and start visualizing.</p>
</div>
''', unsafe_allow_html=True)

# Create placeholder for chart
chart_placeholder = st.empty()

# Create placeholder for data info
info_placeholder = st.empty()

if not st.session_state.data_loaded:
    # Show blank chart with hint
    with chart_placeholder:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Click the menu button (≣) in the upper-right corner\nto upload your dataset and start visualizing!',
                ha='center', va='center', fontsize=14, color='gray', alpha=0.7)
        ax.set_title('📊 Blank Chart - Ready for Your Data', fontsize=16, color='gray', alpha=0.5)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
    
    with info_placeholder:
        st.info("💡 **Tip**: Your dataset should contain at least one date column and one numerical column for optimal visualization.")
else:
    # Display the actual chart with data
    df = st.session_state.df
    
    if df is not None and st.session_state.date_column and st.session_state.value_column:
        try:
            # Create copy to avoid modifying original data
            plot_df = df.copy()
            
            # Convert date column to datetime
            plot_df[st.session_state.date_column] = pd.to_datetime(
                plot_df[st.session_state.date_column], 
                errors='coerce',
                infer_datetime_format=True
            )
            
            # Convert value column to numeric
            plot_df[st.session_state.value_column] = pd.to_numeric(
                plot_df[st.session_state.value_column], 
                errors='coerce'
            )
            
            # Drop rows with null values in the key columns
            plot_df = plot_df.dropna(subset=[st.session_state.date_column, st.session_state.value_column])
            
            # Sort by date
            plot_df = plot_df.sort_values(by=st.session_state.date_column)
            
            with chart_placeholder:
                st.subheader(f"📈 {st.session_state.value_column} over Time")
                
                # Create interactive chart using Streamlit's native chart
                st.line_chart(
                    plot_df.set_index(st.session_state.date_column)[st.session_state.value_column],
                    use_container_width=True,
                    height=500
                )
            
            # Show data summary
            with info_placeholder:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Total Rows", len(df))
                
                with col2:
                    st.metric("✅ Valid Rows", len(plot_df))
                
                with col3:
                    st.metric("EmptyEntries Skipped", len(df) - len(plot_df))
                
                # Show data preview in expander
                with st.expander("🔍 View Data Preview"):
                    st.dataframe(plot_df.head(10))
                    
                    # Show data types
                    st.subheader("Column Data Types")
                    dtype_df = pd.DataFrame({
                        'Column': plot_df.columns,
                        'Data Type': plot_df.dtypes.astype(str)
                    })
                    st.dataframe(dtype_df)
        
        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")
            st.session_state.data_loaded = False

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px; color: #666;">
    <small>✅ Data Validation • 🔒 Secure File Handling • 📊 Interactive Visualization</small><br>
    <small>Built with Streamlit • For educational and analytical purposes only</small>
</div>
""", unsafe_allow_html=True)
