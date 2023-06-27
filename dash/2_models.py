import streamlit as st 
import pandas as pd 
import numpy as np
import joblib


knn = joblib.load("notebooks/knn.joblib")

bagging = joblib.load("notebooks/bagging.joblib")

tf = joblib.load("notebooks/tf.joblib")


huh = {'KNN': knn, 'Bagging': bagging, 'Нейронная сеть': tf}

def upload(model_option): 
    uploaded_file = st.file_uploader("Выберите файл .csv:") 

    if uploaded_file: 
        upload_data = pd.read_csv(uploaded_file, sep=";", encoding='utf-8') 
        upload_data.columns = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"] 

    if st.button("Получить предсказание", key='1'): 
        if not uploaded_file: 
            st.warning('Сначала нужно загрузить данные!') 
        else: 
            y_pred = huh[model_option].predict(upload_data) 

            if model_option == "Нейронная сеть": 
                y_pred = [np.argmax(pred) for pred in y_pred] 

            upload_data['hazardous'] = y_pred 

            st.write(upload_data) 
            st.download_button("Сохранить результат", upload_data.to_csv(index=False, encoding='utf-8'), "file.csv", "text/csv", key='download-csv') 


def input(model_option): 
    edited_df = st.data_editor(pd.DataFrame([{ 
        "est_diameter_min": "", 
        "est_diameter_max": "", 
        "relative_velocity": "", 
        "miss_distance": "", 
        "absolute_magnitude": "" 

    }]), num_rows="dynamic", use_container_width=False, width = 1000) 
    if st.button("Получить предсказание", key='2'): 
        try: 
            y_pred = huh[model_option].predict(edited_df) 

            if model_option == "Нейронная сеть": 
                y_pred = [np.argmax(pred) for pred in tf.predict(y_pred, verbose=None)]  

            edited_df['hazardous'] = y_pred 

            st.write(edited_df) 
            st.download_button("Сохранить результат", edited_df.to_csv(index=False, encoding='utf-8'), "file.csv", "text/csv", key='download-csv') 

        except: 

            st.warning('Некорректные данные!') 


def prediction(): 
    model_option = st.selectbox('Выберите модель для предсказания', ('KNN', 'Bagging', 'Нейронная сеть')) 
    tab1, tab2 = st.tabs(["Загрузка файла .csv", "Ручной ввод значений"]) 

    with tab1: 
        upload(model_option) 

    with tab2: 
        input(model_option)
 
prediction()
