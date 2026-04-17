import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("mnist_ann.h5")

st.title("MNIST Digit Classifier")

file = st.file_uploader("Upload Image", type=["png","jpg"])

if file:
    img = Image.open(file).convert('L').resize((28,28))
    img = np.array(img)/255.0
    img = img.reshape(1,784)

    pred = model.predict(img)
    st.image(img.reshape(28,28))
    st.write("Prediction:", np.argmax(pred))
