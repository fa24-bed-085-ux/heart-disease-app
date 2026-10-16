import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_model.pkl')

st.title('Heart Disease Prediction App')
st.write('Enter the patient details below to check the prediction:')

age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=50, max_value=600, value=200)
max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)

if st.button('Predict'):
    input_data = pd.DataFrame([[age, sex, cholesterol, max_hr]], 
                              columns=['Age', 'Sex', 'Cholesterol', 'MaxHR'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.warning('High risk of heart disease! ⚠️')
    else:
        st.success('Low risk of heart disease. ✅')
