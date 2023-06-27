import streamlit as st 
import pandas as pd
import plotly.express as px 
import plotly.figure_factory as ff


st.set_page_config(
    page_title="visualization",
    page_icon="📊",
)

DATA_PATH = ('./data/classification_pred.csv')

@st.cache_data 
def load_data(nrows):
    data = pd.read_csv(DATA_PATH, nrows=nrows)
    return data

data = load_data(30000)

@st.cache_data 
def corr_plot(): 
    my_fig = px.imshow(data.corr(), text_auto=True) 
    st.plotly_chart(my_fig, use_container_width=True) 

def box_plot(): 
    type = st.selectbox('Выберите признак', ['relative_velocity', 'est_diameter_min', 'est_diameter_max', 'miss_distance', 'absolute_magnitude'], key='1') 
    my_fig = px.box(data, y=type) 
    st.plotly_chart(my_fig, use_container_width=True) 

def chart_plot(): 
    type = st.selectbox('Выберите признак', ['relative_velocity', 'est_diameter_min', 'est_diameter_max', 'miss_distance', 'absolute_magnitude'], key='3') 
    fig = px.scatter(
    data,
    x="hazardous",
    y=type,
    color=type,
    color_continuous_scale="reds",
)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


st.markdown(
    """ # Визуализация данных 📊 """
)

st.markdown(
    """ ### Тепловя карта """
)

corr_plot()

st.markdown(
    """ ### Ящик с усами """
)

box_plot()

st.markdown(
    """ ### Распределение нецелевых признаков относительно целевого """
)

chart_plot()
