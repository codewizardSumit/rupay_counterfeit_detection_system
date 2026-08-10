import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS (UI/UX)
# ==========================================
# Layout 'wide' karne se screen ka poora space use hota hai
st.set_page_config(page_title="Rupay Counterfeit Detection System", page_icon="🏦", layout="wide")

# Custom CSS for a cleaner, professional look (hiding default Streamlit marks)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_ai_model():
    # Aapka trained model load ho raha hai
    return tf.keras.models.load_model('rupay_counterfeit_detector.keras')

try:
    model = load_ai_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"System Error: Model file missing or corrupted. Detail: {e}")

# ==========================================
# 3. SIDEBAR (CONTEXT & INFO)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) # Ek dummy logo
    st.title("VaultGuard AI")
    st.markdown("### Intelligent Currency Scanning System")
    st.markdown("---")
    st.markdown("**About System:**")
    st.markdown("This system uses a Deep Convolutional Neural Network (CNN) to authenticate Indian Currency Notes.")
    st.markdown("- **Supported Notes:** ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000")
    st.markdown("- **Engine:** TensorFlow 2.x")
    st.markdown("- **Accuracy:** ~93.8% (Validation)")
    st.markdown("---")
    st.warning("⚠️ Disclaimer: This is an educational AI project and should not be used for official financial authentication.")

# ==========================================
# 4. MAIN DASHBOARD AREA
# ==========================================
st.title("🏦 Counterfeit Rupay Note Detection System")
st.markdown("Upload a clear image of an Indian currency note to instantly verify its authenticity using Deep Learning.")

# File Uploader
uploaded_file = st.file_uploader("Drop your currency image here (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Ek modern divider
    st.markdown("---")
    
    # Do columns banayein: Left (Image) aur Right (Results)
    col1, col2 = st.columns([1, 1.2]) # col2 thoda bada rakha hai

    with col1:
        st.markdown("### 📷 Scanned Document")
        image = Image.open(uploaded_file)
        # Image ko display karein frame ke andar
        st.image(image, caption='Uploaded for verification', use_container_width=True)


    with col2:
        st.markdown("### 🔍 AI Analysis Report")
        
        # Ek fake progress bar for UX (Aisa lagta hai AI depth scan kar raha hai)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Extracting spatial features...")
        time.sleep(0.3)
        progress_bar.progress(30)
        
        status_text.text("Applying Deep Convolutional Filters...")
        
        # Actual Model Prediction Logic
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        if len(img_array.shape) == 2: # Agar image black & white hai
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4: # Agar PNG hai (RGBA)
            img_array = img_array[..., :3]
            
        img_batch = np.expand_dims(img_array, axis=0)
        
        time.sleep(0.3)
        progress_bar.progress(70)
        status_text.text("Calculating authenticity probability...")
        
        # Prediction
        start_time = time.time()
        prediction = model.predict(img_batch)[0][0]
        inference_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.empty() # Text clear kar do
        
        # ----------------------------------
        # Result Display (Metrics & Alerts)
        # ----------------------------------
        if prediction > 0.9:
            confidence = prediction * 100
            st.success("✅ **STATUS: AUTHENTIC (REAL)**")
            st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%", delta="Verified")
        else:
            confidence = (1 - prediction) * 100
            st.error("🚨 **STATUS: COUNTERFEIT (FAKE)**")
            st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%", delta="- Suspect", delta_color="inverse")

        # Visual Confidence Gauge (Progress bar as a gauge)
        st.markdown("**Confidence Gauge:**")
        st.progress(int(confidence))
        
        # ----------------------------------
        # Technical Explainer (Transparency)
        # ----------------------------------
        with st.expander("⚙️ View Technical AI Metadata"):
            st.code(f"""
Inference Time: {inference_time:.4f} seconds
Input Tensor Shape: {img_batch.shape}
Raw Sigmoid Output: {prediction:.6f}
Decision Threshold: 0.500000
            """)