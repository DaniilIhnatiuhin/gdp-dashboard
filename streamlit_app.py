import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration with favicon
st.set_page_config(
    page_title='Data Visualization Dashboard',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'x_column' not in st.session_state:
    st.session_state.x_column = None
if 'y_column' not in st.session_state:
    st.session_state.y_column = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default theme

# Theme toggle function
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Add custom CSS for theme support with proper contrast
st.markdown(f"""
<style>
    /* Theme-aware CSS variables */
    :root {{
        --light-bg-primary: #ffffff;
        --light-bg-secondary: #f8f9fa;
        --light-bg-tertiary: #e9ecef;
        --light-text-primary: #212529;
        --light-text-secondary: #6c757d;
        --light-border: #dee2e6;
        --light-accent: #0d6efd;
        
        --dark-bg-primary: #1e1e1e;
        --dark-bg-secondary: #2d2d2d;
        --dark-bg-tertiary: #3d3d3d;
        --dark-text-primary: #f8f9fa;
        --dark-text-secondary: #adb5bd;
        --dark-border: #495057;
        --dark-accent: #4da6ff;
    }}
    
    /* Apply theme based on session state */
    {'body { background-color: var(--light-bg-primary); color: var(--light-text-primary); }' if st.session_state.theme == 'light' else 'body { background-color: var(--dark-bg-primary); color: var(--dark-text-primary); }'}
    
    .welcome-container {{
        text-align: center;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        transition: all 0.3s ease;
        {'background-color: var(--light-bg-secondary); border: 1px solid var(--light-border);' if st.session_state.theme == 'light' else 'background-color: var(--dark-bg-secondary); border: 1px solid var(--dark-border);'}
    }}
    
    .welcome-title {{
        {'color: var(--light-text-primary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-primary);'}
        font-size: 1.5em;
        margin-bottom: 10px;
        font-weight: 600;
    }}
    
    .welcome-text {{
        {'color: var(--light-text-secondary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-secondary);'}
        font-size: 1.1em;
        line-height: 1.5;
    }}
    
    .hint-icon {{
        font-size: 1.8em;
        margin-bottom: 15px;
        {'color: var(--light-accent);' if st.session_state.theme == 'light' else 'color: var(--dark-accent);'}
    }}
    
    .theme-toggle-container {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .theme-button {{
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.2s ease;
        {'background-color: var(--light-bg-secondary); color: var(--light-text-primary);' if st.session_state.theme == 'light' else 'background-color: var(--dark-bg-secondary); color: var(--dark-text-primary);'}
    }}
    
    .theme-button:hover {{
        {'background-color: var(--light-bg-tertiary);' if st.session_state.theme == 'light' else 'background-color: var(--dark-bg-tertiary);'}
    }}
    
    .sidebar-header {{
        {'color: var(--light-text-primary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-primary);'}
        font-weight: 600;
    }}
    
    .footer {{
        text-align: center;
        padding: 15px;
        margin-top: 30px;
        border-radius: 8px;
        {'background-color: var(--light-bg-secondary);' if st.session_state.theme == 'light' else 'background-color: var(--dark-bg-secondary);'}
    }}
    
    .footer-text {{
        {'color: var(--light-text-secondary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-secondary);'}
        font-size: 0.9em;
    }}
    
    .chart-placeholder {{
        {'background-color: var(--light-bg-secondary);' if st.session_state.theme == 'light' else 'background-color: var(--dark-bg-secondary);'}
        border-radius: 8px;
        padding: 20px;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }}
    
    .chart-placeholder-text {{
        {'color: var(--light-text-secondary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-secondary);'}
        font-size: 1.2em;
        opacity: 0.8;
        margin: 10px 0;
    }}
    
    .chart-placeholder-title {{
        {'color: var(--light-text-primary);' if st.session_state.theme == 'light' else 'color: var(--dark-text-primary);'}
        font-size: 1.4em;
        font-weight: 500;
        margin: 5px 0;
    }}
    
    .column-selector {{
        margin: 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

# Add theme toggle button in the upper right corner
st.markdown('<div class="theme-toggle-container">', unsafe_allow_html=True)
if st.button("🌓 Toggle Theme", key="theme_toggle", help="Switch between light and dark themes"):
    toggle_theme()
st.markdown('</div>', unsafe_allow_html=True)

def load_data(file):
    """Load data from uploaded file with minimal validation"""
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
    st.markdown(f'<h3 class="sidebar-header">📋 Data Configuration</h3>', unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload Dataset", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.session_state.df = df
            
            # Get all column names for selection
            all_columns = list(df.columns)
            
            if not all_columns:
                st.warning("No columns found in the dataset.")
            else:
                # X-axis column selection (any column)
                st.session_state.x_column = st.selectbox(
                    "AxisSize X-axis Column",
                    options=all_columns,
                    help="Choose any column for the X-axis"
                )
                
                # Y-axis column selection (any column)
                st.session_state.y_column = st.selectbox(
                    "AxisSize Y-axis Column", 
                    options=all_columns,
                    help="Choose any column for the Y-axis"
                )
                
                # Allow same column selection for X and Y
                if st.session_state.x_column == st.session_state.y_column:
                    st.info("💡 You've selected the same column for both X and Y axes. This is allowed.")
                
                if st.button("🚀 Generate Chart", type="primary", use_container_width=True):
                    st.session_state.data_loaded = True
    
    st.markdown("---")
    st.markdown(f'<h4 class="sidebar-header">ℹ️ Flexible Data Handling</h4>', unsafe_allow_html=True)
    info_text = """
    <div style="color: var(--light-text-secondary);" class="requirement-item">
        • <strong>Any Data Format:</strong> No restrictions on column types<br>
        • <strong>Any Columns:</strong> Select any columns for X and Y axes<br>
        • <strong>No Validation:</strong> No warnings or errors for data quality<br>
        • <strong>Permissive:</strong> The app will attempt to plot any data you provide
    </div>
    """
    st.markdown(info_text, unsafe_allow_html=True)

# Main content area
st.title('📊 Data Visualization Dashboard')

# Show welcome message with proper theming
if not st.session_state.data_loaded:
    st.markdown(f'''
    <div class="welcome-container">
        <div class="hint-icon">👋</div>
        <div class="welcome-title">Welcome to the Dashboard!</div>
        <div class="welcome-text">Click the <strong>≣ menu button</strong> in the upper-right corner to upload your dataset and start visualizing.</div>
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="welcome-container">
        <div class="welcome-title">Dataset Loaded Successfully!</div>
        <div class="welcome-text">Your data is now visualized below. Use the sidebar to adjust settings or upload a new dataset.</div>
    </div>
    ''', unsafe_allow_html=True)

# Create placeholder for chart
chart_placeholder = st.empty()

# Create placeholder for data info
info_placeholder = st.empty()

if not st.session_state.data_loaded:
    # Show blank chart with hint using theme-aware styling
    with chart_placeholder:
        st.markdown(f'''
        <div class="chart-placeholder">
            <div class="chart-placeholder-text">Click the menu button (≣) in the upper-right corner</div>
            <div class="chart-placeholder-text">to upload your dataset and start visualizing!</div>
            <div class="chart-placeholder-title">📊 Blank Chart - Ready for Your Data</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with info_placeholder:
        st.info("💡 **Tip**: This app accepts any dataset format. No restrictions on column types or data quality.")
else:
    # Display the actual chart with data
    df = st.session_state.df
    
    if df is not None and st.session_state.x_column and st.session_state.y_column:
        try:
            # Create copy to avoid modifying original data
            plot_df = df.copy()
            
            # Do not attempt to convert data types - use as-is
            x_data = plot_df[st.session_state.x_column]
            y_data = plot_df[st.session_state.y_column]
            
            with chart_placeholder:
                st.subheader(f"📈 {st.session_state.y_column} vs {st.session_state.x_column}")
                
                # Try to create chart with minimal error handling
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Attempt to plot - this may fail with non-numeric data but we catch the error
                    ax.plot(x_data, y_data, marker='o', linestyle='-')
                    
                    ax.set_xlabel(st.session_state.x_column)
                    ax.set_ylabel(st.session_state.y_column)
                    ax.set_title(f'{st.session_state.y_column} vs {st.session_state.x_column}')
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels if they're text
                    if x_data.dtype == 'object' or x_data.dtype.name == 'category':
                        plt.xticks(rotation=45, ha='right')
                    
                    st.pyplot(fig)
                    
                except Exception as plot_error:
                    # Fallback to simple display if plotting fails
                    st.warning(f"Could not create chart: {str(plot_error)}")
                    st.write("Raw data preview:")
                    st.dataframe(plot_df[[st.session_state.x_column, st.session_state.y_column]].head(10))
            
            # Show basic data info without warnings
            with info_placeholder:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Total Rows", len(df))
                
                with col2:
                    st.metric("📏 X Column", st.session_state.x_column)
                
                with col3:
                    st.metric("📏 Y Column", st.session_state.y_column)
                
                # Show data preview in expander
                with st.expander("🔍 View Data Preview"):
                    st.write(f"Showing first 10 rows of {len(df)} total rows")
                    preview_df = df.head(10)
                    st.dataframe(preview_df)
                    
                    # Show basic column info
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count()
                    })
                    st.dataframe(col_info)
        
        except Exception as e:
            # Generic error handling without specific validation messages
            st.error(f"An error occurred while processing your data: {str(e)}")
            if st.button("🔄 Try Again"):
                st.session_state.data_loaded = False

# Footer with theme-aware styling
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <div class="footer-text">
        🔄 Flexible Data Handling • 🔒 Secure File Uploads • 📊 Interactive Visualization<br>
        <small>Built with Streamlit • No data restrictions • All column types accepted</small>
    </div>
</div>
""", unsafe_allow_html=True)
