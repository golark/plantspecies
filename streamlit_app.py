import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import numpy as np
import os
from plant_tips import PlantTipsGenerator

# Title and description
st.title("🌱 Plant Species Detector & Care Assistant")
st.write("Upload an image of a plant to determine its species and get personalized care tips using AI!")

# Sidebar for additional features
st.sidebar.header("🔍 Additional Features")

# Option to use RAG or not
use_rag = st.sidebar.checkbox("Use RAG (Retrieval-Augmented Generation) for care tips", value=True)

# Option to search for specific plants
search_option = st.sidebar.selectbox(
    "Choose an option:",
    ["Upload Image", "Search Plant by Name"]
)

# Load the plant classification model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return load_learner('plant_species_model.pkl')

model = load_model()

# Initialize tips generator with or without RAG
@st.cache_resource
def load_tips_generator(use_rag):
    return PlantTipsGenerator(use_rag=use_rag)

tips_generator = load_tips_generator(use_rag)

if search_option == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to fastai PILImage
        img = PILImage.create(np.array(image))
        pred, _, probs = model.predict(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Species:** {pred}")
        with col2:
            st.info(f"**Confidence:** {probs.max():.2f}")

        # Plant care tips section
        st.subheader("🌿 Plant Care Tips")
        
        # User question input
        user_question = st.text_input(
            "Ask a specific question about care (optional):",
            placeholder="e.g., How often should I water this plant in winter?"
        )
        
        try:
            with st.spinner("🤖 Generating personalized care tips..."):
                tips = tips_generator.generate_gemini_tips(str(pred), user_question=user_question)
            
            # Display tips with better formatting
            st.markdown("### 📋 Care Instructions")
            st.markdown(tips)
            
            # Show RAG status
            if use_rag:
                st.success("✅ Using RAG-enhanced AI for more accurate care tips!")
            else:
                st.info("ℹ️ Using direct AI generation (RAG disabled)")
                
        except Exception as e:
            st.error(f"❌ Could not generate care tips: {e}")
            st.info("💡 Make sure the GEMINI_API_KEY environment variable is set and valid.")

else:
    # Search by plant name
    st.subheader("🔍 Search Plant Care by Name")
    
    plant_name = st.text_input(
        "Enter plant species name:",
        placeholder="e.g., Monstera deliciosa, Ficus lyrata, Aloe vera"
    )
    
    if plant_name:
        # User question input
        user_question = st.text_input(
            "Ask a specific question about care (optional):",
            placeholder="e.g., How often should I water this plant in winter?"
        )
        
        try:
            with st.spinner("🤖 Searching for care information..."):
                tips = tips_generator.generate_gemini_tips(plant_name, user_question=user_question)
            
            # Display tips with better formatting
            st.markdown("### 📋 Care Instructions")
            st.markdown(tips)
            
            # Show RAG status
            if use_rag:
                st.success("✅ Using RAG-enhanced AI for more accurate care tips!")
            else:
                st.info("ℹ️ Using direct AI generation (RAG disabled)")
                
        except Exception as e:
            st.error(f"❌ Could not generate care tips: {e}")
            st.info("💡 Make sure the GEMINI_API_KEY environment variable is set and valid.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌱 Powered by AI & RAG Technology | Plant care tips enhanced with ChromaDB knowledge base</p>
</div>
""", unsafe_allow_html=True) 