import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import io
import base64

# Set page config
st.set_page_config(
    page_title="Housing Data Explorer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1.5rem;
    }
    .stButton button {
        background-color: #4e79a7;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏠 Housing Data Explorer")
st.markdown("Upload your housing dataset and explore prices with interactive maps and filters")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.default_lon = None
    st.session_state.default_lat = None
    st.session_state.default_price = None

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Data Controls")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Housing CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV with flexible separator detection
            content = uploaded_file.getvalue().decode('utf-8')
            if ';' in content[:100]:
                df = pd.read_csv(uploaded_file, sep=';')
            elif ',' in content[:100]:
                df = pd.read_csv(uploaded_file, sep=',')
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.data_loaded = False
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        st.header("🗺️ Map Settings")
        
        # Auto-detect coordinate columns
        lon_col_options = [col for col in df.columns if 'lon' in col.lower() or 'long' in col.lower() or 'x' in col.lower()]
        lat_col_options = [col for col in df.columns if 'lat' in col.lower() or 'y' in col.lower()]
        
        # Default selections
        default_lon = st.session_state.default_lon or (lon_col_options[0] if lon_col_options else df.columns[0])
        default_lat = st.session_state.default_lat or (lat_col_options[0] if lat_col_options else df.columns[1])
        
        # Coordinate column selection
        longitude_col = st.selectbox(
            "Longitude Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_lon) if default_lon in df.columns else 0,
            help="Select the column containing longitude coordinates"
        )
        
        latitude_col = st.selectbox(
            "Latitude Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_lat) if default_lat in df.columns else 1,
            help="Select the column containing latitude coordinates"
        )
        
        # Value column selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            default_price = st.session_state.default_price or ('median_house_value' if 'median_house_value' in numeric_cols else numeric_cols[0])
            value_col = st.selectbox(
                "Value to Display on Map",
                options=numeric_cols,
                index=numeric_cols.index(default_price) if default_price in numeric_cols else 0,
                help="Select the numeric column to visualize on the map"
            )
        else:
            value_col = None
            st.warning("⚠️ No numeric columns found in the dataset")
        
        st.header("🔍 Filters")
        
        # Numeric filters
        if numeric_cols:
            st.subheader("Numeric Filters")
            for col in numeric_cols:
                if col in [longitude_col, latitude_col]:
                    continue
                
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                # Skip if range is too small or contains NaN
                if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
                    continue
                
                filter_vals = st.slider(
                    f"Filter {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=(max_val - min_val) / 100 if max_val > min_val else 0.1,
                    help=f"Filter data by {col} range"
                )
                
                # Apply filter
                df = df[(df[col] >= filter_vals[0]) & (df[col] <= filter_vals[1])]
        
        # Categorical filters
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.subheader("Categorical Filters")
            for col in categorical_cols:
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) <= 20:  # Only show filter if reasonable number of unique values
                    selected_vals = st.multiselect(
                        f"Filter {col}",
                        options=unique_vals,
                        default=unique_vals,
                        help=f"Select values for {col}"
                    )
                    if selected_vals:
                        df = df[df[col].isin(selected_vals)]
        
        # Store filtered data in session state
        st.session_state.filtered_df = df
        st.session_state.longitude_col = longitude_col
        st.session_state.latitude_col = latitude_col
        st.session_state.value_col = value_col
        
        # Save defaults
        st.session_state.default_lon = longitude_col
        st.session_state.default_lat = latitude_col
        st.session_state.default_price = value_col

# Main content area
if st.session_state.data_loaded and hasattr(st.session_state, 'filtered_df'):
    df = st.session_state.filtered_df
    
    # Show data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        if st.session_state.value_col:
            avg_value = df[st.session_state.value_col].mean()
            st.metric(f"Avg {st.session_state.value_col}", f"${avg_value:,.2f}")
    with col3:
        if st.session_state.value_col:
            max_value = df[st.session_state.value_col].max()
            st.metric(f"Max {st.session_state.value_col}", f"${max_value:,.2f}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📊 Data Table", "📈 Statistics"])
    
    with tab1:
        st.header("Geographic Distribution")
        
        if st.session_state.longitude_col and st.session_state.latitude_col and st.session_state.value_col:
            if df.empty:
                st.warning("⚠️ No data matches the current filters")
            else:
                try:
                    # Create interactive map with plotly
                    fig = px.scatter_mapbox(
                        df,
                        lat=st.session_state.latitude_col,
                        lon=st.session_state.longitude_col,
                        color=st.session_state.value_col,
                        size=st.session_state.value_col,
                        color_continuous_scale="Viridis",
                        size_max=15,
                        zoom=8,
                        mapbox_style="carto-positron",
                        hover_data=['median_house_value'] if 'median_house_value' in df.columns else None,
                        title=f"Housing Prices by Location ({st.session_state.value_col})"
                    )
                    
                    fig.update_layout(
                        margin={"r":0,"t":30,"l":0,"b":0},
                        height=600,
                        coloraxis_colorbar=dict(
                            title=st.session_state.value_col,
                            tickprefix="$" if 'price' in st.session_state.value_col.lower() or 'value' in st.session_state.value_col.lower() else ""
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Alternative visualization if map has issues
                    if st.button("Show Scatter Plot (Alternative View)"):
                        fig2 = px.scatter(
                            df,
                            x=st.session_state.longitude_col,
                            y=st.session_state.latitude_col,
                            color=st.session_state.value_col,
                            size=st.session_state.value_col,
                            color_continuous_scale="Viridis",
                            size_max=15,
                            title=f"Location vs Housing Value ({st.session_state.value_col})"
                        )
                        fig2.update_layout(height=600)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error creating map: {str(e)}")
                    st.warning("💡 Try using the scatter plot alternative view or check your coordinate columns")
        else:
            st.info("🔧 Please select longitude, latitude, and value columns in the sidebar to display the map")
    
    with tab2:
        st.header("Filtered Data Table")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button for filtered data
        if not df.empty:
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_housing_data.csv" class="download-button">📥 Download Filtered Data (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with tab3:
        st.header("Data Statistics")
        
        if st.session_state.value_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Value Distribution")
                fig_hist = px.histogram(
                    df, 
                    x=st.session_state.value_col,
                    nbins=30,
                    title=f"Distribution of {st.session_state.value_col}",
                    color_discrete_sequence=['#4e79a7']
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Value by Location")
                if 'ocean_proximity' in df.columns:
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
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar controls to filter data and customize the map visualization. Hover over points on the map to see detailed information.")
