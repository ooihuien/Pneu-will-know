import streamlit as st
from About import font_color

def show_userguide_page():
    yol = "container_background.html"
    f = open(yol, 'r')
    contents = f.read()
    f.close()
    contents = contents.replace('smth', 'User Guide')
    st.markdown(contents, unsafe_allow_html=True)
    # st.title("User Guide")
    st.write("Kindly refer the [user manual] for more detailed guidance to use this web app. ")
    st.write("***")

    font_color("Home")
    st.write("The **Home** page shows some introduction about pneumonia disease such as "
             "definition, causes, symptoms, diagnosis method, high risk group and so on.")

    font_color("About")
    st.write("The **About** page briefly explains the problem statements, objectives, target users and data source of the project in which this web app is the final product. "
             "This page also shows the source code and author information.")

    font_color("Prediction Model")
    st.write("The **Prediction Model** page consists of a MobileNet-v2 pneumonia prediction model. "
             "The user is required to upload a chest X-ray image to to the model predict the diagnosis of pneumonia."
             "The probability of pneumonia will be shown as a result of prediction.")

    font_color("Dashboard")
    st.write("The **Dashboard** page shows 5 visualizations of pneumonia statistics including number of deaths from pneumonia,"
             "pneumonia death rate, pneumonia risk factors and so on.")