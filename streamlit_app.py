import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import sys
import traceback
import json
import os
from datetime import datetime

# Set page config with fallback
try:
    st.set_page_config(
        page_title="Housing Data Explorer",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.warning(f"Could not set page config: {str(e)}")

# Custom CSS for better UI and error visibility
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #f5c6cb;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #ffeeba;
    }
    .diagnostic-box {
        background-color: #e9ecef;
        color: #495057;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9em;
    }
    .stButton button {
        background-color: #4e79a7;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with better error handling
def init_session_state():
    """Initialize session state variables with safe defaults"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.filtered_df = None
        st.session_state.error_message = None
        st.session_state.diagnostic_info = {}
        st.session_state.show_diagnostic = False
        st.session_state.default_lon = None
        st.session_state.default_lat = None
        st.session_state.default_value = None
        st.session_state.memory_usage = 0
        st.session_state.last_error_time = None

init_session_state()

# Title and description
st.title("🏠 Housing Data Explorer")
st.markdown("Upload your housing dataset and explore prices with interactive maps and filters")

# Display error if exists
if st.session_state.error_message:
    st.markdown(f'<div class="error-box"><strong>Error:</strong> {st.session_state.error_message}</div>', unsafe_allow_html=True)
    
    # Show diagnostic info button
    if st.button("Show Diagnostic Information"):
        st.session_state.show_diagnostic = True
        
    if st.session_state.show_diagnostic and st.session_state.diagnostic_info:
        st.markdown('<div class="diagnostic-box">', unsafe_allow_html=True)
        st.subheader(" диагностическая информация")
        for key, value in st.session_state.diagnostic_info.items():
            if key == 'traceback':
                st.text(value)
            else:
                st.write(f"**{key}:** {value}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add copy button for diagnostics
        diag_str = json.dumps(st.session_state.diagnostic_info, indent=2)
        st.download_button(
            label="Download Diagnostic Report",
            data=diag_str,
            file_name=f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Add diagnostic mode toggle
with st.expander("🔧 Advanced Diagnostics (for troubleshooting)"):
    if st.checkbox("Enable Diagnostic Mode"):
        st.markdown("### System Information")
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        st.write(f"**Pandas Version:** {pd.__version__}")
        st.write(f"**NumPy Version:** {np.__version__}")
        st.write(f"**Plotly Version:** {px.__version__}")
        
        st.markdown("### Memory Usage")
        st.write(f"**Current Memory Usage:** {st.session_state.memory_usage:.2f} MB")
        
        # Show environment variables (sanitized)
        st.markdown("### Environment Variables (sanitized)")
        env_vars = {k: v for k, v in os.environ.items() if 'PATH' not in k and 'HOME' not in k}
        st.json(env_vars)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Data Controls")
    
    # File uploader with better error handling
    uploaded_file = st.file_uploader("Upload Housing CSV File", type=['csv'], accept_multiple_files=False)
    
    if uploaded_file is not None:
        try:
            st.session_state.error_message = None  # Clear previous errors
            
            # Get file info for diagnostics
            file_size = uploaded_file.size
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            
            st.session_state.diagnostic_info['file_info'] = {
                'name': file_name,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024),
                'type': file_type
            }
            
            # Read CSV with flexible separator detection and error handling
            content = uploaded_file.getvalue().decode('utf-8', errors='replace')
            
            # Detect separator
            separator = ','
            if ';' in content[:1000]:
                separator = ';'
            elif '\t' in content[:1000]:
                separator = '\t'
            
            st.session_state.diagnostic_info['detected_separator'] = separator
            
            # Try to read with detected separator
            try:
                df = pd.read_csv(io.StringIO(content), sep=separator)
            except Exception as sep_error:
                # Try with different separators if first attempt fails
                separators = [',', ';', '\t', ' ']
                for sep in separators:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=sep)
                        separator = sep
                        break
                    except:
                        continue
                else:
                    raise sep_error
            
            # Check if data was loaded successfully
            if df.empty:
                raise ValueError("The uploaded file is empty or could not be parsed properly.")
            
            # Store original data
            st.session_state.df = df
            st.session_state.filtered_df = df.copy()
            st.session_state.data_loaded = True
            
            # Update memory usage
            st.session_state.memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Auto-detect coordinate columns
            cols = df.columns.tolist()
            lon_col_options = [col for col in cols if 'lon' in col.lower() or 'long' in col.lower() or 'x' in col.lower()]
            lat_col_options = [col for col in cols if 'lat' in col.lower() or 'y' in col.lower()]
            value_col_options = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.session_state.diagnostic_info.update({
                'row_count': len(df),
                'column_count': len(cols),
                'columns': cols,
                'detected_lon_cols': lon_col_options,
                'detected_lat_cols': lat_col_options,
                'detected_value_cols': value_col_options,
                'memory_usage_mb': st.session_state.memory_usage
            })
            
            st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
            st.write(f"📊 **Dataset Info:** {len(df)} rows, {len(cols)} columns")
            st.write(f"💾 **Memory Usage:** {st.session_state.memory_usage:.2f} MB")
            
        except Exception as e:
            error_msg = f"❌ Error loading file: {str(e)}"
            st.session_state.error_message = error_msg
            st.session_state.last_error_time = datetime.now()
            
            # Capture detailed traceback
            tb = traceback.format_exc()
            st.session_state.diagnostic_info.update({
                'error_time': st.session_state.last_error_time.isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': tb,
                'file_attempted': uploaded_file.name if uploaded_file else None
            })
            
            st.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
            st.warning("💡 **Tip:** Check that your CSV file has proper formatting and contains coordinate columns (longitude/latitude).")
    
    # Show controls only if data is loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        st.header("🗺️ Map Settings")
        
        # Get current columns
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Coordinate column selection with defaults
        lon_col_options = [col for col in cols if 'lon' in col.lower() or 'long' in col.lower() or 'x' in col.lower()]
        lat_col_options = [col for col in cols if 'lat' in col.lower() or 'y' in col.lower()]
        
        # Set defaults
        default_lon = st.session_state.default_lon or (lon_col_options[0] if lon_col_options else cols[0] if cols else None)
        default_lat = st.session_state.default_lat or (lat_col_options[0] if lat_col_options else cols[1] if len(cols) > 1 else None)
        
        with st.expander("📍 Coordinate Columns", expanded=True):
            longitude_col = st.selectbox(
                "Longitude Column",
                options=cols,
                index=cols.index(default_lon) if default_lon in cols else 0,
                help="Select the column containing longitude coordinates"
            )
            
            latitude_col = st.selectbox(
                "Latitude Column",
                options=cols,
                index=cols.index(default_lat) if default_lat in cols else 1 if len(cols) > 1 else 0,
                help="Select the column containing latitude coordinates"
            )
        
        # Value column selection
        if numeric_cols:
            default_value = st.session_state.default_value or ('median_house_value' if 'median_house_value' in numeric_cols else numeric_cols[0])
            with st.expander("💰 Value Column", expanded=True):
                value_col = st.selectbox(
                    "Value to Display on Map",
                    options=numeric_cols,
                    index=numeric_cols.index(default_value) if default_value in numeric_cols else 0,
                    help="Select the numeric column to visualize on the map"
                )
        else:
            value_col = None
            st.warning("⚠️ No numeric columns found in the dataset")
        
        st.header("🔍 Filters")
        
        # Create a copy for filtering
        filtered_df = df.copy()
        
        # Numeric filters
        if numeric_cols:
            with st.expander("📊 Numeric Filters"):
                for col in numeric_cols:
                    if col in [longitude_col, latitude_col]:
                        continue
                    
                    try:
                        min_val = float(filtered_df[col].min())
                        max_val = float(filtered_df[col].max())
                        
                        if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
                            continue
                        
                        # Skip if range is too large (could cause performance issues)
                        if max_val - min_val > 1e6 and len(filtered_df) > 10000:
                            continue
                        
                        current_min = min_val
                        current_max = max_val
                        
                        # Get current filter values from session state if they exist
                        if f'filter_{col}' in st.session_state:
                            current_min, current_max = st.session_state[f'filter_{col}']
                        
                        filter_vals = st.slider(
                            f"Filter {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(current_min, current_max),
                            step=max((max_val - min_val) / 100, 0.01),
                            help=f"Filter data by {col} range"
                        )
                        
                        st.session_state[f'filter_{col}'] = filter_vals
                        
                        # Apply filter
                        filtered_df = filtered_df[
                            (filtered_df[col] >= filter_vals[0]) & 
                            (filtered_df[col] <= filter_vals[1])
                        ]
                        
                    except Exception as e:
                        st.warning(f"Could not create filter for {col}: {str(e)}")
        
        # Categorical filters
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            with st.expander("🏷️ Categorical Filters"):
                for col in categorical_cols:
                    try:
                        unique_vals = filtered_df[col].dropna().unique().tolist()
                        if len(unique_vals) <= 20:  # Only show filter if reasonable number of unique values
                            selected_vals = st.multiselect(
                                f"Filter {col}",
                                options=unique_vals,
                                default=unique_vals,
                                help=f"Select values for {col}"
                            )
                            if selected_vals:
                                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                    except Exception as e:
                        st.warning(f"Could not create filter for {col}: {str(e)}")
        
        # Store filtered data and selections
        st.session_state.filtered_df = filtered_df
        st.session_state.longitude_col = longitude_col
        st.session_state.latitude_col = latitude_col
        st.session_state.value_col = value_col
        
        # Save defaults
        st.session_state.default_lon = longitude_col
        st.session_state.default_lat = latitude_col
        st.session_state.default_value = value_col

# Main content area
if st.session_state.data_loaded and st.session_state.filtered_df is not None:
    df = st.session_state.filtered_df
    
    # Show data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if st.session_state.value_col and st.session_state.value_col in df.columns:
            avg_value = df[st.session_state.value_col].mean()
            st.metric(f"Avg {st.session_state.value_col}", f"${avg_value:,.2f}")
    with col3:
        if st.session_state.value_col and st.session_state.value_col in df.columns:
            max_value = df[st.session_state.value_col].max()
            st.metric(f"Max {st.session_state.value_col}", f"${max_value:,.2f}")
    
    # Memory warning if dataset is large
    if st.session_state.memory_usage > 500:  # 500 MB threshold
        st.warning(f"⚠️ Large dataset detected ({st.session_state.memory_usage:.1f} MB). Performance may be affected.")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📊 Data Table", "📈 Statistics"])
    
    with tab1:
        st.header("Geographic Distribution")
        
        if (st.session_state.longitude_col and st.session_state.latitude_col and 
            st.session_state.value_col and st.session_state.value_col in df.columns):
            
            if df.empty:
                st.warning("⚠️ No data matches the current filters")
            else:
                try:
                    with st.spinner(".Rendering map..."):
                        # Validate coordinates
                        lon_col = st.session_state.longitude_col
                        lat_col = st.session_state.latitude_col
                        val_col = st.session_state.value_col
                        
                        # Check for valid coordinates
                        valid_coords = df[
                            (df[lon_col].notna()) & 
                            (df[lat_col].notna()) & 
                            (df[val_col].notna()) &
                            (df[lon_col] != 0) & 
                            (df[lat_col] != 0)
                        ]
                        
                        if len(valid_coords) == 0:
                            st.warning("⚠️ No valid coordinates found in the filtered data")
                        else:
                            # Create interactive map with plotly
                            fig = px.scatter_mapbox(
                                valid_coords,
                                lat=lat_col,
                                lon=lon_col,
                                color=val_col,
                                size=val_col,
                                color_continuous_scale="Viridis",
                                size_max=15,
                                zoom=8,
                                mapbox_style="carto-positron",
                                hover_data=[col for col in df.columns if col not in [lon_col, lat_col, val_col]][:5],
                                title=f"Housing Prices by Location ({val_col})"
                            )
                            
                            fig.update_layout(
                                margin={"r":0,"t":30,"l":0,"b":0},
                                height=600,
                                coloraxis_colorbar=dict(
                                    title=val_col,
                                    tickprefix="$" if any(x in val_col.lower() for x in ['price', 'value', 'cost']) else ""
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Alternative visualization if needed
                            with st.expander("🔧 Alternative Visualization Options"):
                                if st.button("Show Scatter Plot (Alternative View)"):
                                    fig2 = px.scatter(
                                        valid_coords,
                                        x=lon_col,
                                        y=lat_col,
                                        color=val_col,
                                        size=val_col,
                                        color_continuous_scale="Viridis",
                                        size_max=15,
                                        title=f"Location vs Housing Value ({val_col})"
                                    )
                                    fig2.update_layout(height=600)
                                    st.plotly_chart(fig2, use_container_width=True)
                                    
                                if st.button("Show Hexbin Plot (Large Datasets)"):
                                    try:
                                        fig3 = px.density_mapbox(
                                            valid_coords,
                                            lat=lat_col,
                                            lon=lon_col,
                                            z=val_col,
                                            radius=10,
                                            center=dict(lat=valid_coords[lat_col].mean(), lon=valid_coords[lon_col].mean()),
                                            zoom=8,
                                            mapbox_style="carto-positron",
                                            title=f"Density Map: {val_col}"
                                        )
                                        fig3.update_layout(height=600)
                                        st.plotly_chart(fig3, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not create hexbin plot: {str(e)}")
                                
                except Exception as e:
                    error_msg = f"❌ Error creating map: {str(e)}"
                    st.error(error_msg)
                    st.session_state.error_message = error_msg
                    st.session_state.diagnostic_info['map_error'] = str(e)
                    st.warning("💡 **Tip:** Try using the scatter plot alternative view or check your coordinate columns")
                    if st.button("Show Map Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.info("🔧 Please select longitude, latitude, and value columns in the sidebar to display the map")
    
    with tab2:
        st.header("Filtered Data Table")
        
        if not df.empty:
            # Show a sample of the data first
            show_all = st.checkbox("Show all data (may be slow for large datasets)")
            
            if show_all:
                st.dataframe(df, use_container_width=True)
            else:
                sample_size = min(100, len(df))
                st.write(f"Showing first {sample_size} rows of {len(df)} total:")
                st.dataframe(df.head(sample_size), use_container_width=True, height=400)
            
            # Data quality check
            with st.expander("🔍 Data Quality Check"):
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    st.warning("Dataset contains missing values:")
                    st.write(null_counts[null_counts > 0])
                else:
                    st.success("✅ No missing values detected")
            
            # Download button for filtered data
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_housing_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" class="download-button">📥 Download Filtered Data (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No data available. Try adjusting your filters.")
    
    with tab3:
        st.header("Data Statistics")
        
        if st.session_state.value_col and st.session_state.value_col in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Value Distribution")
                fig_hist = px.histogram(
                    df, 
                    x=st.session_state.value_col,
                    nbins=min(30, len(df)//100 + 10),
                    title=f"Distribution of {st.session_state.value_col}",
                    color_discrete_sequence=['#4e79a7']
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if 'ocean_proximity' in df.columns:
                    st.subheader("Value by Location")
                    fig_box = px.box(
                        df,
                        x='ocean_proximity',
                        y=st.session_state.value_col,
                        title=f"{st.session_state.value_col} by Ocean Proximity",
                        color='ocean_proximity'
                    )
                    fig_box.update_layout(height=300)
                    st.plotly_chart(fig_box, use_container_width=True)
        
        # Show correlation heatmap if multiple numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlations")
            try:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    aspect='auto'
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {str(e)}")
                
else:
    st.info("🏠 Welcome to the Housing Data Explorer! Upload your CSV file in the sidebar to get started.")
    
    # Show example data structure
    st.subheader("Example Data Format")
    st.markdown("""
    Your CSV should contain columns similar to this example:
    - longitude: -121.44, -121.44, -121.43, ...
    - latitude: 37.74, 37.73, 37.73, ...
    - median_house_value: 112500.0, 208100.0, 134700.0, ...
    - Other optional columns: housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
    
    **Tips for successful upload:**
    - Use CSV format with commas as separators
    - Include coordinate columns (longitude/latitude)
    - Ensure numeric columns contain only numbers
    - Keep file size under 100MB for best performance
    """)

# Footer with help information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <strong>💡 Help & Troubleshooting</strong><br>
    If you encounter errors:<br>
    1. Check that your CSV has proper formatting<br>
    2. Ensure coordinate columns contain valid numbers<br>
    3. Try a smaller dataset first<br>
    4. Use the diagnostic mode in the sidebar<br>
    5. Contact support with the diagnostic report
</div>
""", unsafe_allow_html=True)

# Auto-scroll to errors
if st.session_state.error_message:
    st.markdown("""
    <script>
        // Auto-scroll to the top when there's an error
        window.scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)
