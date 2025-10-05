import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
from datetime import datetime, date

# Page configuration
st.set_page_config(page_title="✈️ AI Predictive Maintenance System 🛠️ ", layout='wide')

# Load the dataset
column_names = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=column_names)

# Sidebar
with st.sidebar:
    st.sidebar.image('images\download8.jfif')
    st.sidebar.subheader("This dashboard for Predictive Maintenance for Aircraft Engines")
    st.sidebar.write("")

    st.sidebar.subheader("🔍 Select a sensor to explore :")
    # List of available sensors
    sensor_list = [f"sensor_{i}" for i in range(1, 22)]
    # User selection
    selected_sensor = st.sidebar.selectbox("Choose Sensor", sensor_list)

    st.sidebar.write("")
    st.sidebar.markdown("Made by [Predictive Maintainance Team](https://github.com/Omayma-ali/PredictiveMaintenance)")

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Insights", "🤖 Prediction"])

# Home tab
with tab1:
    st.write("If you're an airline maintenance manager or an aviation safety engineer,\n"
            "you need a system that can predict engine failures before they occur.\n"
            "Aircraft maintenance operations have become increasingly complex due to "
            "changing operational factors and varying flight conditions.\n\n"
            "My project aims to help you make the right decisions and perform timely preventive maintenance"
            "by developing a predictive model that can identify potential failures based on operational data.")
   
    im1 = Image.open('images\Aircraft-microchip-800x500.jpg')
    im2 = Image.open('images\download7.jfif')
    im3 = Image.open('images\images8.jfif')
    im4 = Image.open('images\images10.jfif')
    im5 = Image.open('images\image9.png')

    # Images for predictive maintenance stages
    img1, img2 = st.columns((5, 5))
    img1.image(im1, caption='Data transmission stage from sensors')
    img2.image(im2, caption='Aircraft engine sensors')
    
    img3, img4 = st.columns((5, 5))
    img3.image(im3, caption='Processing sensor data and sending fault alerts')
    img4.image(im4, caption='Handling fault alerts')
    
    img5 = st
    img5.image(im5, caption='Safe travels all the time')

# Insights tab
with tab2:
    visual1, visual2 = st.columns((5, 5))

    with visual1:
        st.subheader(f"📈 Sensor Trend: {selected_sensor}")
        if selected_sensor in df.columns:
            fig = px.line(df, x='time_cycles', y=selected_sensor)
            st.plotly_chart(fig)
        else:
            st.warning("Selected sensor not in dataset.")

    with visual2:
        st.subheader(f"📈 Sensor Trend: {selected_sensor}")
        if selected_sensor in df.columns:
            fig2 = px.line(df, x='unit_number', y=selected_sensor)
            st.plotly_chart(fig2)
        else:
            st.warning("Selected sensor not in dataset.")

# Prediction tab
with tab3:
    model = joblib.load('predictive_maintainance_model.p')    

    all_features = ['time_cycles', 'unit_number', 'op_setting_1','op_setting_2','op_setting_3'] + [f"sensor_{i}" for i in range(1, 22)]
    model_features = ['time_cycles', 'unit_number', 'sensor_8', 'sensor_13', 'sensor_12',
                      'sensor_7', 'sensor_11', 'sensor_4', 'sensor_15']

    for feature in all_features:
        if feature not in st.session_state:
            st.session_state[feature] = 0.0
    if 'rul' not in st.session_state:
        st.session_state.rul = None

    # Custom CSS for animation
    st.markdown("""
    <style>
    .shake {
        animation: shake 0.5s;
        animation-iteration-count: infinite;
    }

    @keyframes shake {
        0% { transform: translate(1px, 1px) rotate(0deg); }
        10% { transform: translate(-1px, -2px) rotate(-1deg); }
        20% { transform: translate(-3px, 0px) rotate(1deg); }
        30% { transform: translate(3px, 2px) rotate(0deg); }
        40% { transform: translate(1px, -1px) rotate(1deg); }
        50% { transform: translate(-1px, 2px) rotate(-1deg); }
        60% { transform: translate(-3px, 1px) rotate(0deg); }
        70% { transform: translate(3px, 1px) rotate(-1deg); }
        80% { transform: translate(-1px, -1px) rotate(1deg); }
        90% { transform: translate(1px, 2px) rotate(0deg); }
        100% { transform: translate(1px, -2px) rotate(-1deg); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Layout row
    col_gen, col_rul = st.columns([3, 2])

    with col_gen:
        if st.button("🔄 Generate Random Sensor Data"):
            for feature in all_features:
                if feature == 'unit_number':
                    st.session_state[feature] = int(np.random.randint(1, 101))
                elif feature == 'time_cycles':
                    st.session_state[feature] = int(np.random.randint(1, 301))
                else:
                    st.session_state[feature] = round(np.random.uniform(-2, 2), 4)

    # Auto prediction
    try:
        input_data = []
        for feat in model_features:
            val = st.session_state[feat]
            if feat == 'unit_number':
                val = min(max(int(val), 1), 100)
            elif feat == 'time_cycles':
                val = int(val)
            input_data.append(val)

        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        predicted_rul = float(prediction[0]) if isinstance(prediction[0], (int, float, np.floating)) else float(prediction[0][0])
        st.session_state.rul = predicted_rul

    except Exception as e:
        st.session_state.rul = None
        st.error(f"Prediction failed: {e}")

    # RUL Output
    with col_rul:
        if st.session_state.rul is not None:
            st.metric("📉 Predicted RUL", f"{st.session_state.rul:.2f} cycles")

            if st.session_state.rul < 20:
                st.toast("🚨 Critical RUL! Act immediately.", icon="🚨")
                st.markdown("""
                <div class="shake" style='background-color:#ffe6e6;padding:15px;border-radius:10px;color:#a10000;font-weight:bold'>
                ⚠️ RUL is dangerously low! Immediate action required!
                </div>
                """, unsafe_allow_html=True)

                # Play alert sound
                st.markdown("""
                <audio autoplay>
                  <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
                </audio>
                """, unsafe_allow_html=True)

            elif st.session_state.rul < 50:
                st.toast("⚠️ RUL is getting low. Plan inspection.", icon="⚠️")
                st.markdown("""
                <div style='background-color:#fff6e6;padding:15px;border-radius:10px;color:#996600;font-weight:bold'>
                ⚠️ Consider inspecting the engine soon.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.toast("✅ RUL is healthy. No immediate action needed.", icon="✅")
                st.markdown("""
                <div style='background-color:#e6ffe6;padding:15px;border-radius:10px;color:#006600;font-weight:bold'>
                ✅ Engine condition looks good!
                </div>
                """, unsafe_allow_html=True)

    # Inputs section
    st.subheader("✍️ Input Sensor and Operational Values:")
    cols = st.columns(3)
    for idx, feature in enumerate(all_features):
        col = cols[idx % 3]

        if feature == 'unit_number':
            if st.session_state[feature] < 1:
                st.session_state[feature] = 1
            col.number_input(
                label=feature,
                min_value=1,
                max_value=100,
                step=1,
                value=int(st.session_state[feature]),
                key=feature
            )

        elif feature == 'time_cycles':
            col.number_input(
                label=feature,
                step=1,
                value=int(st.session_state[feature]),
                key=feature
            )

        else:
            col.number_input(
                label=feature,
                step=0.01,
                value=st.session_state[feature],
                key=feature
            )

