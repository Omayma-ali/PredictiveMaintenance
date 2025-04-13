import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
from datetime import datetime, date

# Setting page configuration
st.set_page_config(page_title="✈️ AI Predictive Maintenance System 🛠️ ", layout='wide')

# Loading data
column_names = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=column_names)

with st.sidebar:

    st.sidebar.image('download8.jfif')
    st.sidebar.subheader("This dashboard for Predictive Maintenance for Aircraft Engines")
    st.sidebar.write("")

    st.sidebar.subheader("🔍 Select a sensor to explore :")
    # قائمة بالـ sensors المتاحة
    sensor_list = [f"sensor_{i}" for i in range(1, 22)]  # sensors 1 → 21
    # اختيار المستخدم
    selected_sensor = st.sidebar.selectbox("Choose Sensor", sensor_list)
            
    st.sidebar.write("")
    st.sidebar.markdown("Made by [Predictive Maintainance Team](https://github.com/KarimXHamed/PredictiveMaintenance)")



# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Insights", "🤖 Prediction"])

# introduction
with tab1:
    st.write("If you're an airline maintenance manager or an aviation safety engineer,\n"
            "you need a system that can predict engine failures before they occur.\n"
            "Aircraft maintenance operations have become increasingly complex due to "
            "changing operational factors and varying flight conditions.\n\n"


            "My project aims to help you make the right decisions and perform timely preventive maintenance"
            "by developing a predictive model that can identify potential failures based on operational data.")
   
    im1 = Image.open('Aircraft-microchip-800x500.jpg')
    im2 = Image.open('download7.jfif')
    im3 = Image.open('images8.jfif')
    im4 = Image.open('images10.jfif')
    im5 = Image.open('image9.png')

    # Images2 Cards for predictive mainainance stages
    img1, img2 = st.columns((5, 5))
    # Display the images in the columns
    img1.image(im1, 
               caption='Data transmission stage from sensors')
    img2.image(im2, 
               caption='Air craft engine sensors')
    
    img3, img4 = st.columns((5, 5))
    img3.image(im3, 
              caption='Processing sensor data and sending fault alerts')
    img4.image(im4, 
                caption='Handling fault alerts')
    img5 = st
    img5.image(im5, 
              caption='Safe travels all the time')
    

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
            fig = px.line(df, x='time_cycles', y=selected_sensor)
            st.plotly_chart(fig)
        else:
            st.warning("Selected sensor not in dataset.")


# predicting Model
with tab3:

    model = joblib.load('predictive_maintainance_model.p')    
    # عدد الأعمدة المطلوبة (بناءً على شكل البيانات اللي بعتيه — بدون unit_number و time)
    all_features = ['time_cycles', 'unit_number', 'op_setting_1','op_setting_2','op_setting_3'] + [f"sensor_{i}" for i in range(1, 22)]

    # --- Initialize session state for each feature
    for feature in all_features:
        if feature not in st.session_state:
            st.session_state[feature] = 0.0

    # --- Generate random values
    if st.button("🔄 Generate Random Sensor Data"):
        for feature in all_features:
            st.session_state[feature] = round(np.random.uniform(-2, 2), 4)

    # --- Display inputs (filled with generated values)
    cols = st.columns(3)  # Display in 3 columns
    inputs = []

    for idx, feature in enumerate(all_features):
        col = cols[idx % 3]
        value = col.number_input(
            label=feature,
            value=st.session_state[feature],
            key=feature
        )
        inputs.append(value)

    # توقع الـ RUL
    # --- الأعمدة اللي الموديل متدرب عليها
    model_features = ['time_cycles', 'unit_number', 'sensor_8', 'sensor_13', 'sensor_12',
                  'sensor_7', 'sensor_11', 'sensor_4', 'sensor_15']

    if st.button("🚀 Predict RUL"):
        try:
            input_data = []
            for feat in model_features:
                val = st.session_state[feat]
                # ضبط القيم للـ unit_number و time_cycles
                if feat == 'unit_number':
                    val = min(max(int(val), 1), 100)  # محصور بين 1 و 100
                elif feat == 'time_cycles':
                    val = int(val)  # أي رقم صحيح
                input_data.append(val)

            input_array = np.array([input_data])
            prediction = model.predict(input_array)
            predicted_rul = float(prediction[0]) if isinstance(prediction[0], (int, float, np.floating)) else float(prediction[0][0])

            st.success(f"📉 Predicted Remaining Useful Life: **{predicted_rul:.2f}** cycles")

            if predicted_rul < 20:
                st.error("⚠️ Warning: Low Remaining Useful Life! Schedule maintenance soon.")
            elif predicted_rul < 50:
                st.warning("⚠️ Moderate RUL - plan for inspection.")
            else:
                st.info("✅ Equipment health is good.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

