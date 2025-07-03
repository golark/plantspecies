import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import numpy as np
import os
from plant_tips import PlantTipsGenerator

# Title and description
st.title("Plant Species Detector")
st.write("Upload an image of a plant to determine its species.")

# Load the plant classification model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return load_learner('plant_species_model.pkl')

model = load_model()
tips_generator = PlantTipsGenerator()

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

    try:
        with st.spinner("Fetching care tips from Gemini..."):
            tips = tips_generator.generate_gemini_tips(str(pred))
        st.subheader("Plant Care Tips (Gemini)")
        st.write(tips)
    except Exception as e:
        st.warning(f"Could not generate care tips from Gemini: {e}")
        st.info("Make sure the GEMINI_API_KEY environment variable is set and valid.") 