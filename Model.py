import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from skimage import exposure
import time

def show_model_page():
    yol = "container_background.html"
    f = open(yol, 'r')
    contents = f.read()
    f.close()
    contents = contents.replace('smth', 'Prediction Model')
    st.markdown(contents, unsafe_allow_html=True)

    # st.title("MobileNet Prediction Model")
    image_uploaded = st.file_uploader("Upload chest X-ray image", type=['jpeg'])
    st.write(type(image_uploaded))
    # st.write(image_uploaded)
    image = Image.open(image_uploaded)
    st.image(image, width= 224, caption="Uploaded image")
    img_array = np.array(image)

    clicked = st.button("Predict")

    if clicked:
        new_predict(img_array)

def new_predict(image):

    if image is None:
        print("Wrong path 555")
    else:
        image = cv2.resize(image,(224,224))
        print(image.shape)
        if len(image.shape)==2:
            image = np.dstack([image, image, image])
            print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        image = HE(image)
        image = np.expand_dims(image, axis=0)

    with st.spinner("Loading prediciton model......"):
        model = load_model()

    yhat = model.predict(image)

    # st.write(yhat[0])
    # st.write(yhat[1])
    # yhat = np.argmax(yhat, axis=1)
    # st.write(yhat)
    if (np.argmax(yhat, axis=1)==0):
        st.success("Model runs successfully!")
        st.table(yhat)
        # yhat.columns = ['Normal', 'Pneumonia']
        st.balloons()
        st.write("Congratulations! The result is normal with " + str(yhat[0][0]) + " probability.")

    if (np.argmax(yhat, axis=1)==1):
        st.success("Model runs successfully!")
        st.table(yhat)
        # yhat.columns = ['Normal', 'Pneumonia']
        st.write("Positive pneumonia with " + str(yhat[0][1]) + " probability.")


# @st.cache
def load_model():
    model = tf.keras.models.load_model('mobile_HE_3.hdf5')
    return model

def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

# my_bar = st.progress(0)
#
# for percent_complete in range(100):
#      time.sleep(0.1)
#      my_bar.progress(percent_complete + 1)