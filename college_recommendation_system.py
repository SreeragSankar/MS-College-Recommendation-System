# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 09:52:08 2025

@author: sankaranarayanan
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity 

def load_assests():
    scaler=pickle.load(open('C:/Users/sankaranarayanan/Desktop/ML Model as API/Python Code/scaler.pickle','rb'))
    features=pickle.load(open('C:/Users/sankaranarayanan/Desktop/ML Model as API/Python Code/features.pickle','rb'))
    df=pd.read_csv('C:/Users/sankaranarayanan/Desktop/cpdata.csv')
    return scaler,features,df

def input_vector(fees,package,country,country_list):
    roi = package/fees if fees!=0 else 1
    country_vector=[1 if c==country else 0 for c in country_list]
    return [fees,package,roi] + country_vector

def recommend(user_input,df,features,scaler):
    user_scaled=scaler.transform([user_input])
    df_scaled=scaler.transform(df[features])
    similarity_scores=cosine_similarity(user_scaled,df_scaled)[0]
    df['Similarity']=similarity_scores
    
    top5=df.sort_values(by='Similarity',ascending=False)[['College_Name','Country','Rank','Fees','Avg Package','ROI']].head(5)
    return top5

def main():
    st.title("🎓 MS in CS College Recommender")
    
    scaler,features,df=load_assests()
    country_list=[c for c in features if c not in['Fees','Avg Package','ROI']]
    
    fees = st.number_input("💰 Enter your affordable tuition fee (USD)", min_value=1000, step=1000)
    package = st.number_input("📈 Enter your expected salary/package (USD)", min_value=10000, step=1000)
    country = st.selectbox("🌍 Preferred Country", sorted(country_list))
    
    if st.button("Recommend Top 5 Colleges"):
        user_input=input_vector(fees,package,country,country_list)
        recommendation=recommend(user_input, df, features, scaler)
        st.subheader("🎯 Top 5 College Recommendations")
        st.table(recommendation.reset_index(drop=True))
        
if __name__ == "__main__":
    main()
