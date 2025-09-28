import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="🚚 Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, date
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow, handle if not available
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
    .danger-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    try:
        xgb_model = joblib.load('models/xgb_model.joblib')
        ct = joblib.load('models/column_transformer.joblib')
        
        lstm_model = None
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_model = load_model('models/lstm_model.h5')
            except:
                st.warning("LSTM model not found. Using XGBoost only.")
        
        return xgb_model, ct, lstm_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please make sure model files exist in the 'models' directory")
        return None, None, None

# Preprocessing function (simplified version)
def preprocess_single_order(order_data, ct):
    """Preprocess a single order for prediction"""
    try:
        # Create DataFrame
        df = pd.DataFrame([order_data])
        
        # Add required columns with defaults if missing
        defaults = {
            'Order_ID': 'PRED_001',
            'Delivery_Time': 25,  # Dummy value for preprocessing
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
        
        # Parse datetime features
        try:
            order_dt = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str))
            pickup_dt = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Pickup_Time'].astype(str))
            
            df['order_hour'] = order_dt.dt.hour.fillna(12)
            df['order_minute'] = order_dt.dt.minute.fillna(0)
            df['pickup_delay_mins'] = ((pickup_dt - order_dt).dt.total_seconds() / 60.0).fillna(30.0)
        except:
            df['order_hour'] = 12
            df['order_minute'] = 0
            df['pickup_delay_mins'] = 30.0
        
        # Calculate latitude difference
        df['lat_diff'] = abs(df['Store_Latitude'] - df['Drop_Latitude'])
        
        # Transform using fitted column transformer
        X = ct.transform(df)
        return X
        
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

# Main app
def main():
    # TensorFlow warning (moved here)
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not available. Only XGBoost predictions will work.")
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Delivery Delay Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict delivery delays with AI-powered models</p>', unsafe_allow_html=True)
    
    # Load models
    xgb_model, ct, lstm_model = load_models()
    
    if xgb_model is None:
        st.error("❌ Could not load models. Please check if model files exist.")
        return
    
    # Sidebar - Input form
    st.sidebar.markdown('<h2 class="sub-header">📋 Order Details</h2>', unsafe_allow_html=True)
    
    with st.sidebar.form("order_form"):
        st.markdown("### 👤 Agent Information")
        agent_age = st.slider("Agent Age", 18, 65, 30, help="Age of the delivery agent")
        agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1, help="Agent's average rating")
        
        st.markdown("### 📍 Location Details")
        col1, col2 = st.columns(2)
        with col1:
            store_lat = st.number_input("Store Latitude", value=12.9716, format="%.4f")
            drop_lat = st.number_input("Drop Latitude", value=12.9800, format="%.4f")
        with col2:
            store_lon = st.number_input("Store Longitude", value=77.5946, format="%.4f")
            drop_lon = st.number_input("Drop Longitude", value=77.6000, format="%.4f")
        
        st.markdown("### ⏰ Time Details")
        order_date = st.date_input("Order Date", value=date.today())
        
        col1, col2 = st.columns(2)
        with col1:
            order_time = st.time_input("Order Time", value=time(14, 30))
        with col2:
            pickup_time = st.time_input("Pickup Time", value=time(15, 0))
        
        st.markdown("### 🌤️ Conditions")
        weather = st.selectbox("Weather", 
                              ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"],
                              help="Current weather conditions")
        
        traffic = st.selectbox("Traffic", 
                              ["Light", "Medium", "Heavy", "Jam"],
                              help="Traffic density level")
        
        vehicle = st.selectbox("Vehicle Type", 
                              ["Bike", "Car", "Van", "Truck"],
                              help="Type of delivery vehicle")
        
        area = st.selectbox("Area Type", 
                           ["Urban", "Suburban", "Rural"],
                           help="Type of delivery area")
        
        category = st.selectbox("Order Category", 
                               ["Food", "Grocery", "Electronics", "Clothing", "Medicine", "Other"],
                               help="Category of items being delivered")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Delivery Delay", use_container_width=True)
    
    # Main content area
    if submitted:
        # Prepare order data
        order_data = {
            'Agent_Age': agent_age,
            'Agent_Rating': agent_rating,
            'Store_Latitude': store_lat,
            'Drop_Latitude': drop_lat,
            'Drop_Longitude': drop_lon,
            'Order_Date': str(order_date),
            'Order_Time': str(order_time),
            'Pickup_Time': str(pickup_time),
            'Weather': weather,
            'Traffic': traffic,
            'Vehicle': vehicle,
            'Area': area,
            'Category': category
        }
        
        # Preprocess data
        X = preprocess_single_order(order_data, ct)
        
        if X is not None:
            # Make predictions
            try:
                # XGBoost prediction
                xgb_prob = xgb_model.predict_proba(X)[0, 1]
                xgb_pred = int(xgb_prob >= 0.5)
                
                # LSTM prediction (if available)
                lstm_prob = None
                lstm_pred = None
                if lstm_model is not None:
                    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
                    lstm_prob = float(lstm_model.predict(X_lstm)[0, 0])
                    lstm_pred = int(lstm_prob >= 0.5)
                
                # Ensemble prediction
                if lstm_prob is not None:
                    ensemble_prob = (xgb_prob + lstm_prob) / 2
                else:
                    ensemble_prob = xgb_prob
                
                ensemble_pred = int(ensemble_prob >= 0.5)
                
                # Display results
                st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🤖 XGBoost Model</h3>
                        <h2>{xgb_prob:.1%}</h2>
                        <p>Delay Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if lstm_prob is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🧠 LSTM Model</h3>
                            <h2>{lstm_prob:.1%}</h2>
                            <p>Delay Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <h3>🧠 LSTM Model</h3>
                            <h2>N/A</h2>
                            <p>Not Available</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Ensemble</h3>
                        <h2>{ensemble_prob:.1%}</h2>
                        <p>Final Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Final verdict
                st.markdown("### 📊 Final Prediction")
                
                if ensemble_prob > 0.7:
                    st.markdown(f"""
                    <div class="danger-card">
                        <h2>🚨 HIGH RISK OF DELAY</h2>
                        <p>Probability: {ensemble_prob:.1%}</p>
                        <p>Expected delay likely. Consider alternative arrangements.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif ensemble_prob > 0.3:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h2>⚠️ MODERATE RISK</h2>
                        <p>Probability: {ensemble_prob:.1%}</p>
                        <p>Some delay possible. Monitor closely.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-card">
                        <h2>✅ LOW RISK - ON TIME</h2>
                        <p>Probability: {ensemble_prob:.1%}</p>
                        <p>Delivery expected to be on time!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed breakdown
                st.markdown("### 📈 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = ensemble_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Delay Risk %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk factors
                    st.markdown("#### 🔍 Risk Factors")
                    
                    risk_factors = []
                    if traffic in ['Heavy', 'Jam']:
                        risk_factors.append("🚦 Heavy Traffic")
                    if weather in ['Rainy', 'Stormy', 'Foggy']:
                        risk_factors.append("🌧️ Bad Weather")
                    if agent_rating < 3.5:
                        risk_factors.append("⭐ Low Agent Rating")
                    if vehicle == 'Bike' and weather in ['Rainy', 'Stormy']:
                        risk_factors.append("🏍️ Bike in Bad Weather")
                    
                    pickup_delay = (datetime.combine(order_date, pickup_time) - 
                                  datetime.combine(order_date, order_time)).total_seconds() / 60
                    if pickup_delay > 45:
                        risk_factors.append("⏰ Long Pickup Delay")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(factor)
                    else:
                        st.success("✅ No major risk factors detected")
                
                # Model comparison
                if lstm_prob is not None:
                    st.markdown("### 🔬 Model Comparison")
                    
                    comparison_data = {
                        'Model': ['XGBoost', 'LSTM', 'Ensemble'],
                        'Probability': [xgb_prob, lstm_prob, ensemble_prob],
                        'Prediction': ['Delay' if xgb_pred else 'On Time', 
                                     'Delay' if lstm_pred else 'On Time',
                                     'Delay' if ensemble_pred else 'On Time']
                    }
                    
                    df_comp = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(df_comp, x='Model', y='Probability', 
                               title="Model Predictions Comparison",
                               color='Probability',
                               color_continuous_scale='RdYlGn_r')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if ensemble_prob > 0.5:
                    st.markdown("""
                    - 🔄 **Consider rescheduling** if possible
                    - 📞 **Inform customer** about potential delay
                    - 🚗 **Assign experienced agent** if available
                    - 📍 **Optimize route** planning
                    - ⏰ **Add buffer time** to delivery estimate
                    """)
                else:
                    st.markdown("""
                    - ✅ **Proceed as planned**
                    - 📱 **Send tracking updates** to customer
                    - 🎯 **Maintain current schedule**
                    - 📊 **Monitor progress** regularly
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    else:
        # Welcome screen
        st.markdown("### 👋 Welcome to the Delivery Delay Predictor!")
        st.markdown("""
        This AI-powered application helps predict whether a delivery will be delayed based on various factors such as:
        
        - 👤 **Agent details** (age, rating)
        - 📍 **Location information** (pickup & drop coordinates) 
        - ⏰ **Time factors** (order time, pickup time)
        - 🌤️ **External conditions** (weather, traffic)
        - 🚚 **Logistics** (vehicle type, area, category)
        
        **👈 Fill in the order details in the sidebar and click 'Predict' to get started!**
        """)
        
        # Sample statistics or charts can go here
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🤖 **XGBoost Model**\nGradient boosting for structured data")
        
        with col2:
            st.info("🧠 **LSTM Model**\nDeep learning for sequential patterns")
        
        with col3:
            st.info("🎯 **Ensemble**\nCombined predictions for accuracy")

# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info("""
    **Models Used:**
    - XGBoost Classifier
    - LSTM Neural Network
    - Ensemble Averaging
    
    **Accuracy:** ~85-90%
    **Features:** 15+ engineered features
    """)
    
    st.markdown("### 🔧 Settings")
    show_debug = st.checkbox("Show Debug Info")
    
    if show_debug:
        st.markdown("### 🐛 Debug Information")
        try:
            xgb_model, ct, lstm_model = load_models()
            st.success("✅ Models loaded successfully")
            st.write(f"XGBoost available: {xgb_model is not None}")
            st.write(f"LSTM available: {lstm_model is not None}")
            st.write(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
        except:
            st.error("❌ Error loading models")

if __name__ == "__main__":
    main()