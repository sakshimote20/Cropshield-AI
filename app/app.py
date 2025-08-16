import streamlit as st
import mysql.connector
import tensorflow as tf
import numpy as np
import io
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime
import base64
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope

# --- Load environment variables ---
env_path = Path(__file__).parent / ".env"
if env_path.exists():  # Local development
    load_dotenv(env_path)

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "cropshield_db"),
    "port": int(os.getenv("MYSQL_PORT", 3306))
}
MODEL_PATH = os.getenv("MODEL_PATH", "app/model/cropshield_ai.h5")

# --- Set Background ---
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/jpg;base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.jpg')

# --- Create a Dummy Cast Layer ---
class DummyCast(Layer):
    def __init__(self, dtype=None, **kwargs):
        super(DummyCast, self).__init__(**kwargs)
        self.dtype_ = dtype

    def call(self, inputs):
        return inputs

# --- Load class indices ---
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

# --- Load Model with Custom Object ---
custom_objects = {'Cast': DummyCast}
with custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- MySQL Connection ---
try:
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_name VARCHAR(255),
        predicted_class VARCHAR(100),
        confidence FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_blob LONGBLOB
    )''')
    conn.commit()
except mysql.connector.Error as e:
    st.error(f"Database connection failed: {e}")
    conn, cursor = None, None

# --- Batch Prediction ---
st.sidebar.header("Batch Prediction")
batch_files = st.sidebar.file_uploader(
    "Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

def predict_batch(images):
    for uploaded_file in images:
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        result = class_names[predicted_class]

        if confidence < 30:
            st.warning(f"Prediction: {result} with low confidence ({confidence:.2f}%)")
        else:
            st.success(f"Prediction: {result} ({confidence:.2f}% confidence)")

        if cursor:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_data = img_bytes.getvalue()
            cursor.execute(
                '''INSERT INTO predictions (image_name, predicted_class, confidence, image_blob, timestamp)
                   VALUES (%s, %s, %s, %s, %s)''',
                (result, result, confidence, img_data, datetime.now())
            )
            conn.commit()

if batch_files:
    predict_batch(batch_files)

# --- Tabs ---
tab1, tab2 = st.tabs(["Upload an Image", "Search Past Predictions by Plant Name"])

with tab1:
    st.subheader("📄 Upload an Image")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        result = class_names[predicted_class]

        if confidence < 30:
            st.warning("The model is uncertain about the disease. Please consult an expert.")
        else:
            st.success(f"Prediction: {result} ({confidence:.2f}% confidence)")

        if cursor:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_data = img_bytes.getvalue()
            cursor.execute(
                '''INSERT INTO predictions (image_name, predicted_class, confidence, image_blob, timestamp)
                   VALUES (%s, %s, %s, %s, %s)''',
                (result, result, confidence, img_data, datetime.now())
            )
            conn.commit()

with tab2:
    st.subheader("🔍 Search Past Predictions by Plant Name")
    search_term = st.text_input("Enter Plant Name to Search")

    if st.button("Search") and cursor:
        cursor.execute(
            "SELECT image_name, predicted_class, confidence, image_blob, timestamp "
            "FROM predictions WHERE image_name LIKE %s", (f"%{search_term}%",)
        )
        results = cursor.fetchall()

        if results:
            for image_name, predicted_class, confidence, image_blob, timestamp in results:
                st.info(f"🌿 **Plant**: {image_name}\n\n🧪 **Prediction**: {predicted_class}\n\n🎯 **Confidence**: {confidence:.2f}%\n\n🕓 **Timestamp**: {timestamp}")
        else:
            st.warning("No results found.")
