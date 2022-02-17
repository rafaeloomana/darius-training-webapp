import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns

st.title("Insurance Pricing App")
st.write("From the insurance data, we built a machine learning model for pricing insurance claims.")

st.sidebar.title("Insurance Pricing App Parameters")
st.sidebar.write("Tweak to change predictions")

#Age
age = st.sidebar.slider("Age", 0, 100, 24)

#BMI
bmi = st.sidebar.slider("BMI", 15, 40, 29)

#Number of children
num_children = st.sidebar.slider("Number of Children" , 0, 12, 1)

# Is Smoker
smoker = st.sidebar.radio("Smoker?" , ('yes', 'no'))

if smoker == 'yes':
    is_smoker = 1
else:
    is_smoker = 0

#Gender
gender = st.sidebar.radio("Gender", ('Female', 'Male'))

if gender == 'Male':
    is_female = 0
else:
    is_female = 1
    
#Region
region = st.sidebar.selectbox("Region", ['Nortwest', 'Northeast','Southwest', 'Southeast'])

if region == 'Northeast':
    loc_list = [1,0,0,0]
elif region == 'Nortwest':
    loc_list = [0,1,0,0]
elif region == 'Southeast':
    loc_list = [0,0,1,0]
elif region == 'Southwest':
    loc_list = [0,0,0,1]

st.subheader('Output Insurance Price')

filename="finalized_model.sav"

loaded_model = joblib.load(filename)

prediction=round(loaded_model.predict([[age, bmi, num_children, is_female, is_smoker] + loc_list])[0])

st.write(f"Suggested Insurance Price is: {prediction}")

#load data
data = pd.read_csv("insurance_regression.csv")
