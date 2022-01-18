import streamlit as st
from PIL import Image
from About import font_color

def show_home_page():
    home_img = Image.open("images/Pneu-will-know.png")
    st.image(home_img, use_column_width=True)
    st.write("""##### This is a  pneumonia prediction web app using ***Chest X-ray*** images and it also shows dashboard about global pneumonia statistics!""")
    st.write("""***""")

    # st.subheader("What is Pneumonia?")
    font_color("What is Pneumonia?")
    col7, col8 = st.columns([4,3])
    with col7:
        st.write("Pneumonia is an acute respiratory infection of one or both lungs caused by bacteria, viruses, or fungi. "
                 "It is a serious infection in which the air sacs or so called alveoli are filled with pus and other liquids. "
                 "This will cause difficult breathing and reduced oxygen intake.")
    with col8:
        st.video('https://www.youtube.com/watch?v=K_r-kMJjh8Y')

    font_color("Symptoms of Pneumonia")
    col9, col10 = st.columns([4, 3])
    with col9:
        st.markdown("""
        * Cough with yellowish or blood-stained mucus
        * Fever
        * Shortness of breath
        * Rapid pulse
        """)
    with col10:
        st.markdown("""
        * Chest pain
        * Muscle pain
        * Extreme tiredness
        """)

    font_color("How to Diagnose Pneumonia?")
    col1, col2 = st.columns([5,4])
    with col1:
        st.markdown("""
        * Asking about patients’ recent health history
        * Physical exam using a stethoscope 
        * Chest X-ray
        * Chest CT-scan
        * Pulse oximetry test
        * Blood test
        * Sputum culture
        """)
    with col2:
        diagnosis_img = Image.open("images/pneumonia-diagnosis.png")
        st.image(diagnosis_img, use_column_width=True)

    font_color("High Risk Group for Getting Pneumonia")
    st.markdown("""
    * Children under age 2
    * Elderly over age 65
    * People with health-caused weakened immune system
    * People who smoke
    """)
    col3, col4, col5, col6 = st.columns(4)
    risk_grp_img1, risk_grp_img2, risk_grp_img3, risk_grp_img4 = load_image()
    with col3:
        st.image(risk_grp_img1, use_column_width=True)
    with col4:
        st.image(risk_grp_img2, use_column_width=True)
    with col5:
        st.image(risk_grp_img3, use_column_width=True)
    with col6:
        st.image(risk_grp_img4, use_column_width=True)
    st.text("\n")

    font_color("Some Statistics about Pneumonia")
    st.write("""
    1. Globally, pneumonia accounts for 14% of all deaths of children under 5 years old in 2019.\n
    2. Children deaths from pneumonia are highest in South Asia and sub-Saharan Africa.\n
    3. In Malayisa, pneumonia is the second highest cause of death (12.2%) in 2019.
    """)

@st.cache
def load_image():
    img1 = Image.open("images/high risk grp 1.png")
    img2 = Image.open("images/high risk grp 2.png")
    img3 = Image.open("images/high risk grp 3.png")
    img4 = Image.open("images/high risk grp 4.png")
    return img1,img2,img3,img4


