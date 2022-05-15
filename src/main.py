import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

title = '💵 Predict Salary'

subtitle = '''Predict salary for any job with AI model

(Data courtesy of [PredictSalary](https://predictsalary.com/))
'''

COMPANIES = ['Gojek', 'Shopee', 'Tiket.com', 'Tokopedia', 'Traveloka', 'Bukalapak', 'Other']

countvec_dir = './model/count_vectorizer.pkl'
catboost_model_dir = './model/catboost_model.pkl'

with open(countvec_dir, 'rb') as file:
    count_vectorizer = pickle.load(file)

with open(catboost_model_dir, 'rb') as file:
    catboost_model = pickle.load(file)
    
def predict(data: pd.DataFrame):
    counts = count_vectorizer.transform(data.role).toarray().tolist()
    X = np.hstack([counts, data.drop('role', axis=1).values])
    
    y_pred = catboost_model.predict(X).tolist()
    return y_pred

def main():
    st.set_page_config(layout="centered", page_icon='💵', page_title=title)
    st.title(title)
    st.write(subtitle)
    
    form = st.form("Job Details")
    
    role = form.text_input('Job role')
    
    company_placeholder = 'Select company'
    company = form.selectbox('Company', [company_placeholder] + COMPANIES)
    valid_company = (company != company_placeholder)

    years_of_exp = form.slider('Years of Experience', min_value=0, max_value=30)
    
    city = form.text_input('City')
    
    submit = form.form_submit_button("Predict!")

    if submit:
        if not valid_company:
            st.error('Please select company')
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

if __name__ == '__main__':
    main()