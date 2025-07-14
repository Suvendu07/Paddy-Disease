# Paddy Leaf Disease Classification

This project is a Deep Learning model to classify diseases in paddy leaves using EfficientNet architecture.

## Features

- Classifies 9 classes of paddy leaf diseases.
- Supports image upload and live camera capture.
- Handles common image formats including HEIC (from iPhones).
- Built with TensorFlow and deployed with Streamlit.

## Setup

1. Clone this repository.

2. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

- Choose image input method: upload or take a photo.
- Upload or capture a paddy leaf image.
- Get instant disease prediction with confidence filtering.

## Notes

- Model file `paddy_disease_model.keras` should be in the same directory as `app.py`.
- HEIC image support requires `pillow-heif`.

## License

MIT License

---

Made with ❤️ by Suvendu
