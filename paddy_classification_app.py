import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pillow_heif import register_heif_opener
from tensorflow.keras.applications.efficientnet import preprocess_input

# Register HEIC support
register_heif_opener()

# Load model
model = tf.keras.models.load_model("paddy_disease_model.keras")

# Class labels
class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'blast', 'brown_spot',
    'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]

CONFIDENCE_THRESHOLD = 0.6

# Set page config
st.set_page_config(
    page_title="🌾 Paddy Leaf Disease Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        background-color: #f0fdf4;
        padding: 2rem 4rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .title {gf
        color: #2e7d32;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .description {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 2em;
    }
    .footer {
        font-size: 0.9em;
        text-align: center;
        margin-top: 3rem;
        color: #888;
    }
    .stRadio > div {
        flex-direction: row;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🌾 Paddy Leaf Disease Classifier</div>", unsafe_allow_html=True)

st.markdown("<div class='description'>Upload or capture a paddy leaf image to detect possible diseases using our deep learning model powered by EfficientNet.</div>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("🌿 About the App")
st.sidebar.markdown("This tool helps farmers and researchers quickly identify paddy leaf diseases with high accuracy using a deep learning model.")
st.sidebar.markdown("Developed with ❤️ using TensorFlow and Streamlit.")
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("1. Choose image input method.\n2. Upload or capture an image.\n3. View the prediction instantly.")

# Input method selection
option = st.radio("Choose input method:", ('📁 Upload an Image', '📸 Take a Photo'))

image = None

# Preprocessing function (corrected to match training pipeline)
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Upload
if option == '📁 Upload an Image':
    uploaded_file = st.file_uploader("Upload a paddy leaf image", type=["jpg", "jpeg", "png", "heic"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='📤 Uploaded Image', use_column_width=True)
        except Exception:
            st.error("❌ Unable to read image. Please upload a valid photo.")

# Camera
elif option == '📸 Take a Photo':
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption='📷 Captured Photo', use_column_width=True)

# Prediction
if image is not None:
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    confidence = np.max(prediction)
    top_two_probs = np.sort(prediction[0])[-2:]
    prob_gap = top_two_probs[1] - top_two_probs[0]

    if confidence < CONFIDENCE_THRESHOLD or prob_gap < 0.2:
        st.warning("⚠️ The image might not be a valid paddy leaf or is unclear. Please try again.")
    else:
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"✅ Predicted Disease: **{predicted_class.replace('_', ' ').title()}** (Confidence: {confidence:.2f})")

# Footer
st.markdown("<div class='footer'>\nMade with 💚 by Suvendu | EfficientNet Model | Streamlit UI\n</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
