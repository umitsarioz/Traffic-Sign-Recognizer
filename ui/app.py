import time
import numpy as np
import requests
import streamlit as st
from PIL import Image


def add_button(uploaded_file=None, img: list = None):
    if st.button("Predict"):
        if uploaded_file is not None:
            payload = {'img_array': img}
            response = requests.post(url="http://backend:8040/predict", json=payload)
            if response.status_code == 200:
                with st.spinner('Predicting...'):
                    time.sleep(2)
                    predicted_caption = response.json()["message"]
                st.success("Prediction: " + predicted_caption)
            else:
                st.error("Fetching error for predicting traffic sign." + str(response.status_code))
        else:
            st.warning("Load an image.")


def add_image():
    uploaded_file = st.file_uploader("Load image file", type=["png", "jpg", "jpeg"])
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img.resize((200, 200)), caption=uploaded_file.name)
        img = np.array(img)
        print("Load img shape:",img.shape)
        img = img.tolist()


    return uploaded_file, img


def main():
    st.title("🚦Traffic Sign Recognizer")
    st.info("Load a png traffic sign image to recognize.")
    uploaded_file, img = add_image()
    add_button(uploaded_file=uploaded_file, img=img)


if __name__ == "__main__":
    main()
