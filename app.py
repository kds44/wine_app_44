import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import logging
from datetime import datetime

# Import custom modules
from auth import login_form, logout, is_authenticated, get_current_user
from data_processor import WineQualityProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = WineQualityProcessor()
    st.session_state.data_loaded = False

def load_data_if_needed():
    # Loads and processes wine quality dataset if not already loaded
    if not st.session_state.data_loaded:
        logger.info("Loading wine quality dataset...")
        if st.session_state.processor.load_data():
            if st.session_state.processor.clean_and_feature_engineer():
                if st.session_state.processor.train_model():
                    st.session_state.data_loaded = True
                    logger.info("Data loaded and model trained successfully!")
                else:
                    logger.error("Failed to train model")
            else:
                logger.error("Failed to process data")
        else:
            logger.error("Failed to load dataset")

def main_dashboard():
    # Renders the main dashboard interface with tabs for different features
    st.title("🍷 Wine Quality Predictor")
    st.markdown("*Predict wine quality based on chemical properties*")
    
    # Welcome message
    user = get_current_user()
    st.sidebar.success(f"Welcome, {user}!")
    
    # Load data
    load_data_if_needed()
    
    if not st.session_state.data_loaded:
        logger.warning("Application data not loaded - user may need to wait")
        return
    
    # Sidebar for user logout
    logout()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predict Quality", "📊 Data Exploration", "📈 Visualizations", "🔧 Model Performance", "📋 Application Logs"])
    
    with tab1:
        prediction_interface()
    
    with tab2:
        data_exploration()
    
    with tab3:
        visualizations()
    
    with tab4:
        model_performance()
    
    with tab5:
        view_application_logs()

def prediction_interface():
    # Creates interactive interface for wine quality prediction
    st.header("Wine Quality Prediction")
    st.markdown("Enter wine properties to predict quality (scale 3-8)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity (g/L)", 4.0, 16.0, 8.0, 0.1)
        volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1, 1.2, 0.5, 0.01)
        citric_acid = st.slider("Citric Acid (g/L)", 0.0, 1.0, 0.3, 0.01)
        residual_sugar = st.slider("Residual Sugar (g/L)", 0.9, 5.0, 2.2, 0.1)
    
    with col2:
        chlorides = st.slider("Chlorides (g/L)", 0.02, 0.15, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("Free SO₂ (mg/L)", 0.8, 60.0, 15.0, 1.0)
        total_sulfur_dioxide = st.slider("Total SO₂ (mg/L)", 4.0, 200.0, 50.0, 1.0)
    
    with col3:
        ph = st.slider("pH Level", 2.5, 4.0, 3.3, 0.01)
        sulphates = st.slider("Sulphates (g/L)", 0.25, 1.5, 0.65, 0.01)
        alcohol = st.slider("Alcohol (% by volume)", 8.0, 16.0, 10.0, 0.1)
    
    if st.button("🔮 Predict Wine Quality", type="primary"):
        result = st.session_state.processor.predict_quality(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            ph, sulphates, alcohol
        )
        
        if result:
            prediction = result['prediction']
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Quality", f"{prediction:.1f}/8")
            with col2:
                quality_category = "Excellent" if prediction >= 7 else "Good" if prediction >= 6 else "Average" if prediction >= 5 else "Below Average"
                st.metric("Quality Category", quality_category)
            with col3:
                confidence = min(95, max(60, 85 - abs(prediction - 6) * 5))
                st.metric("Confidence", f"{confidence:.0f}%")
            
            logger.info(f"Prediction made by {get_current_user()}: {prediction:.2f}")

def data_exploration():
    # Displays data exploration and descriptive statistics
    st.header("📊 Data Exploration")
    
    # Get descriptive statistics
    stats = st.session_state.processor.get_descriptive_stats()
    data = st.session_state.processor.get_data_for_visualization()
    
    if stats and data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Samples**: {len(data):,}")
            st.write(f"**Features**: {len(data.columns)-1}")
            st.write(f"**Quality Range**: {data['quality'].min()} - {data['quality'].max()}")
            
            st.subheader("Quality Distribution")
            quality_counts = data['quality'].value_counts().sort_index()
            st.bar_chart(quality_counts)
        
        with col2:
            st.subheader("Average Alcohol by Quality")
            st.dataframe(stats['avg_alcohol_by_quality'].round(2))
            
            st.subheader("Basic Statistics")
            st.dataframe(stats['basic_stats'].round(2))

def visualizations():
    # Creates and displays three different types of data visualizations
    st.header("📈 Data Visualizations")
    
    data = st.session_state.processor.get_data_for_visualization()
    if data is None:
        st.error("No data available for visualization")
        return
    
    # Visualization 1: Scatter Plot
    st.subheader("1. 🔵 Scatter Plot - Alcohol vs Quality")
    fig1 = px.scatter(
        data, 
        x='alcohol', 
        y='quality',
        color='pH',
        size='fixed acidity',
        title="Wine Quality vs Alcohol Content",
        labels={'alcohol': 'Alcohol (%)', 'quality': 'Quality Score'}
    )
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualization 2: Correlation Heatmap
    st.subheader("2. 🔥 Correlation Heatmap")
    corr_features = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'pH',
        'sulphates',
        'alcohol',
        'quality'
    ]
    corr_matrix = data[corr_features].corr()
    
    fig2 = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale='RdBu_r',
        showscale=True
    )
    fig2.update_layout(title="Feature Correlation Matrix", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Visualization 3: Line Plot showing volatile relationship
    st.subheader("3. 📈 Line Plot - Volatile Acidity vs Quality")
    
    # Group by quality and calculate mean volatile acidity
    quality_volatile = data.groupby('quality')['volatile acidity'].mean().reset_index()
    
    fig3 = px.line(
        quality_volatile,
        x='quality',
        y='volatile acidity',
        title="Average Volatile Acidity by Quality Score",
        labels={'quality': 'Quality Score', 'volatile acidity': 'Average Volatile Acidity (g/L)'}
    )
    fig3.update_layout(height=400, xaxis_title="Quality Score", yaxis_title="Average Volatile Acidity")
    st.plotly_chart(fig3, use_container_width=True)

def model_performance():
    # Displays model performance metrics
    st.header("Model Performance")
    
    metrics = st.session_state.processor.get_model_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training R²", f"{metrics['train_r2']:.3f}")
            st.metric("Training MAE", f"{metrics['train_mae']:.3f}")
        
        with col2:
            st.metric("Test R²", f"{metrics['test_r2']:.3f}")
            st.metric("Test MAE", f"{metrics['test_mae']:.3f}")

def view_application_logs():
    # Displays application logs in the app
    st.header("📋 Application Logs")
    
    # User Information
    st.subheader("User Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Current User**: {get_current_user()}")
    with col2:
        st.write(f"**Session Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        st.write("**Log File**: app.log")
    
    st.markdown("---")
    
    try:
        with open('app.log', 'r') as f:
            logs = f.read()
        st.text_area("Log Content", value=logs, height=600, disabled=True)
    except FileNotFoundError:
        st.error("Log file 'app.log' not found.")
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")

def main():
    # Main application entry point - handles authentication and dashboard
    # Check authentication
    if not is_authenticated():
        st.title("🍷 Wine Quality Predictor")
        st.markdown("Please log in to access the application")
        login_form()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()