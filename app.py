import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from random import randrange

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_pretrained_model(path):
    loaded_model = pickle.load(open(path, 'rb'))
    return loaded_model

def inference_model(model, data):
    """
    Here, we assume that the data
    """
    confidence = np.amax(model.predict_proba(data.reshape(1, -1)))
    decision = model.predict(data.reshape(1, -1))
    return confidence, decision

st.title("ADE: ADHD prediction from EveryDay features")
st.write("To use ADE, please fill the form below.")
with st.form("ADHD_form"):
    st.header("About the subject ")
    height_inch = st.number_input('Your Height (inches)')
    weight_pound = st.number_input('Your Weight (pounds)')

    sleep_hours = st.slider('How many hours do you sleep per day? (hours/day)', 0.0, 24.0)
    income_desc = st.radio(
        'What is your TOTAL COMBINED FAMILY INCOME for the past 12 months? ',
        ('Less than $5,000', '$5,000 through $11,999', '$12,000 through $15,999',
        '$16,000 through $24,999', '$25,000 through $34,999', '$35,000 through $49,999',
        '$50,000 through $74,999', '$75,000 through $99,999', '$100,000 through $199,999',
        '$200,000 and greater'))

    st.subheader("Screen time")
    st.text("Please fill the screen time during typical weekdays and weekends.")
    st.text("Here, the screen time is the sum of ")

    st.markdown(' - Watch videos (such as YouTube)')
    st.markdown(' - Play video games on a computer, console, phone or other device')
    st.markdown(' - Text on a cell phone, tablet, or computer')
    st.markdown(' - Visit social networking sites like Facebook, Twitter, Instagram')
    st.markdown(' - Video chat (Skype, Facetime, etc.)')

    screen_week = st.slider('Screen time during typical weekdays (hours/day)', 0.0, 24.0)

    screen_weekend = st.slider('Screen time during typical weekends (hours/day)', 0.0, 24.0)

    st.header("About the biological mother of the subject")
    drug_during_pregnancy = st.multiselect('Have you ever taken substances below during pregancy?',
                                    ['Tobacco', 'Alcohol', 'Marijuana', 'Cocaine/Crack',
                                    'Heroin/Morphine', 'Oxycontin', 'Any other drugs', 'None'])

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.success("Successfully submitted the form.")
        subject_bmi = (weight_pound * 703) / ((height_inch)**2)
        #st.write('BMI of the subject is ', subject_bmi)

        income_category_num = 0
        if income_desc == "Less than $5,000":
            income_category_num=1
        elif income_desc == "$5,000 through $11,999":
            income_category_num=2
        elif income_desc == "$12,000 through $15,999":
            income_category_num=3
        elif income_desc == "$16,000 through $24,999":
            income_category_num=4
        elif income_desc == "$25,000 through $34,999":
            income_category_num=5
        elif income_desc == "$35,000 through $49,999":
            income_category_num=6
        elif income_desc == "$50,000 through $74,999":
            income_category_num=7
        elif income_desc == "$75,000 through $99,999":
            income_category_num=8
        elif income_desc == "$100,000 through $199,999":
            income_category_num=9
        elif income_desc == "$200,000 and greater":
            income_category_num=10

        drug_used = True
        if 'None' in drug_during_pregnancy:
            drug_used = False

        avg_screen_time = (screen_week*5 + screen_weekend*2)/7

        model = load_pretrained_model("model.sav")
        data = np.array([float(subject_bmi), float(sleep_hours), income_category_num, float(avg_screen_time), drug_used])
        proba, res = inference_model(model, data)
        col1, col2 = st.columns(2)
        if res == 1:
            st.write("The subject is likely to be in the high-risk ADHD group")
            col1.metric("Prediction", "High Risk")
        else:
            st.write("The subject is not likely to be in the high-risk ADHD group")
            col1.metric("Prediction", "Low Risk")
        proba_str = f'{proba:.2f}'
        col2.metric("Model confidence", float(proba_str))
        st.write("Model confidence is a value between 0 and 1.")
        st.write("Higher confidence value means model is highly confident with it's prediction.")
                

        