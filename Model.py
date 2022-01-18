import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from skimage import exposure
import pandas as pd
import time

def show_model_page():
    yol = "HTML/container_background.html"
    f = open(yol, 'r')
    contents = f.read()
    f.close()
    contents = contents.replace('smth', 'Prediction Model')
    st.markdown(contents, unsafe_allow_html=True)

    # st.title("MobileNet Prediction Model")
    image_uploaded = st.file_uploader("Upload chest X-ray image", type=['jpeg'])
    # st.write(type(image_uploaded))
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
        image = resize(image, (224,224))
        print(image.shape)
        if len(image.shape)==2:
            image = np.dstack([image, image, image])
            print(image.shape)
        image = image.astype(np.float32) / 255.
        image = HE(image)
        image = np.expand_dims(image, axis=0)

    with st.spinner("Loading prediciton model......"):
        model = load_model()

    yhat = model.predict(image)
#     print(type(yhat))
#     print(len(yhat))
#     print(yhat.shape)
#     print(yhat)

    yhat_col = ["Normal (Class 0)", "Pneumonia (Class 1)"]
    yhat_data = [yhat[0][0], yhat[0][1]]
    pred_table_cols = {"Class": yhat_col,"Probability": yhat_data}
    pred_table = pd.DataFrame(pred_table_cols)

    if (np.argmax(yhat, axis=1)==0):
        st.success("Model runs successfully!")
        st.dataframe(pred_table)
        explanation()
        st.balloons()
        result_str = "Congratulations! The predicted result is NORMAL with %.4f probability." % yhat[0][0]
        format(result_str)

    if (np.argmax(yhat, axis=1)==1):
        st.success("Model runs successfully!")
        st.dataframe(pred_table)
        explanation()
        result_str = "Predicted POSITIVE PNEUMONIA with %.4f probability." % yhat[0][1]
        format(result_str)


# @st.cache
def load_model():
    model = tf.keras.models.load_model('mobile_HE_3.hdf5')
    return model

def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

def explanation():
    st.markdown("* The sum of the probability of each class is 1.0.  \n"
                "* If the predicted probability of a class is higher than 0.5, the prediction result is assigned to that class.  \n"
                "* If the predicted probability of both classes are equal to 0.5, the prediction result is assigned to class 1 (Pneumonia).  \n"
                "* The higher the probability of a class, the higher the confidence of the prediction.")

def format(text):
    yol = "HTML/pred_result_text.html"
    f = open(yol, 'r')
    contents = f.read()
    f.close()
    contents = contents.replace('smth', text)
    st.markdown(contents, unsafe_allow_html=True)
# my_bar = st.progress(0)
#
# for percent_complete in range(100):
#      time.sleep(0.1)
#      my_bar.progress(percent_complete + 1)
