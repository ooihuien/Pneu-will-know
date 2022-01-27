import streamlit as st
from Home import show_home_page
from About import show_about_page
from User_guide import show_userguide_page
from Model import show_model_page
from Dashboard import show_dashboard_page

# st.set_page_config(layout="wide")

st.sidebar.title("Navigate to...")
navigator = st.sidebar.radio("", ("Home", "About", "User Guide", "Prediction Model", "Dashboard"))

st.markdown(
    """ 
    <style>
    .main{ 
    background-color: #F5F5F5;
    } 
    </style> 
    """,
    unsafe_allow_html=True )

if navigator == "Home":
    show_home_page()
if navigator == "About":
    show_about_page()
if navigator == "User Guide":
    show_userguide_page()
if navigator == "Prediction Model":
    show_model_page()
if navigator == "Dashboard":
    show_dashboard_page()



    # @st.cache
    # /Users/huienooi/Desktop/DSP_app/
