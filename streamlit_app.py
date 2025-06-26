import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import numpy as np

# Title and description
st.title("Plant Species Detector")
st.write("Upload an image of a plant to determine its species.")

# Load the model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return load_learner('plant_species_model.pkl')

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to fastai PILImage
    img = PILImage.create(np.array(image))
    pred, _, probs = model.predict(img)
    st.success(f"Species: {pred}")
    st.info(f"Confidence: {probs.max():.2f}") 