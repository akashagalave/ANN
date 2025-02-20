import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

model = tf.keras.models.load_model('akash.h5')



with open('label_encoder.pkl','rb') as file:
    lebel_encoder=pickle.load(file)

with open('ohe_encoder.pkl','rb') as file:
    ohe_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Salary Estimation')

geography = st.selectbox('Geography',ohe_encoder.categories_[0])
gender = st.selectbox('Gender',lebel_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credut Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
Exited = st.selectbox('Exited',[0,1])



input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[lebel_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[Exited]
    
})

encoder = ohe_encoder.transform([[geography]]).toarray()
encoded_df=pd.DataFrame(encoder,columns=ohe_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)


prediction = model.predict(input_data_scaled)
st.write(f"Estimated Salary: {prediction[0][0]}")
