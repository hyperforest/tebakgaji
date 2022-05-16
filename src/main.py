import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

title = '💵 TebakGaji'
subtitle = 'Predict salary for any job with machine learning'
footer = '''❤ Visit [repository](https://github.com/hyperforest/predict_salary_project)

Data courtesy of [PredictSalary](https://predictsalary.com)
'''
company_placeholder = 'Select company'

COMPANIES = ['Gojek', 'Shopee', 'Tiket.com', 'Tokopedia', 'Traveloka', 'Bukalapak', 'Other']
CITIES = ['Bandung', 'Denpasar', 'Jakarta', 'Semarang', 'Surabaya', 'Yogyakarta', 'Other']
COUNTVEC_DIR = './model/count_vectorizer.pkl'
MODEL_DIR = './model/catboost_model.pkl'

with open(COUNTVEC_DIR, 'rb') as file:
    count_vectorizer = pickle.load(file)

with open(MODEL_DIR, 'rb') as file:
    model = pickle.load(file)
    
def predict(data: pd.DataFrame):
    counts = count_vectorizer.transform(data.role).toarray().tolist()
    X = np.hstack([counts, data.drop('role', axis=1).values])
    
    y_pred = model.predict(X).tolist()
    return y_pred

def main():
    st.set_page_config(layout="centered", page_icon='💵', page_title='TebakGaji')
    st.title(title)
    st.write(subtitle)
    
    form = st.form("Job details")
    role = form.text_input('Job role')
    company = form.selectbox('Company', [company_placeholder] + COMPANIES)
    years_of_exp = form.number_input('Years of experience', min_value=0, max_value=30)
    city = form.selectbox('City', CITIES)
    other_city = form.text_input('Please type city here if you choose other')
    
    if city == 'Other':
        city = other_city

    valid_input = (
        (role != '')
        & (company != company_placeholder)
        & (city != '')
    )

    submit = form.form_submit_button("Predict!")
    if submit:
        if not valid_input:
            st.error('Please fill the form properly')
        else:
            data = {
                'role': role.lower(),
                'company': company.lower(),
                'years_of_exp': years_of_exp,
                'city': city.lower()
            }
            data = pd.Series(data).to_frame(name=0).T
            prediction = predict(data)[0]

            st.success('Predicted salary: IDR %.1fM' % prediction)

    st.write(footer)

if __name__ == '__main__':
    main()