# 🌿 CropShield AI – Smart Crop Health Diagnosis

CropShield AI is a deep learning–based web application that detects crop diseases from plant images, provides instant predictions, and stores results for future analysis — helping farmers improve crop management.  

Built using **TensorFlow/Keras (MobileNetV2)**, **Streamlit**, and **MySQL**.

---

🚀 Live App: *[Add your deployment link here]*  
📊 Training Notebook: [View on Kaggle](your-kaggle-notebook-link)  
🌱 Dataset: [View on Kaggle](your-kaggle-dataset-link)  
🤖 Trained Model: [Download from Kaggle](your-kaggle-model-link)

---

## ✨ Features
- 🌿 **Disease Detection** – Upload crop images and get instant predictions with confidence scores.  
- 📂 **Batch Prediction** – Upload multiple images for faster diagnosis.  
- 🗄️ **Database Logging** – Predictions, images, and timestamps stored in MySQL.  
- 🔍 **History Search** – Query past predictions by plant name.  
- 🎨 **Streamlit UI** – Simple, user-friendly interface with background image.  
- 🔐 **Secure Config** – Uses `.env` for credentials and model path.  

---

## 🧠 Tech Stack
- **Deep Learning**: TensorFlow, Keras (MobileNetV2)  
- **Frontend**: Streamlit  
- **Database**: MySQL (`mysql-connector-python`)  
- **Data Handling**: NumPy, Pandas, Pillow  
- **Config Management**: python-dotenv  

---

## 📂 Folder Structure
cropshield-ai/
│
├── app/
│ ├── app.py # Main Streamlit app
│ ├── background.jpg # App background image
│ ├── class_indices.json # Class label mapping
│ ├── train.csv, val.csv, test.csv # Dataset splits 
│ ├── .env # Local DB credentials 
│ ├── .env.example # Example env file 
│ └── model/
│ └── cropshield_ai.h5 # Trained model
│
├── training/
│ └── cropshield_train.ipynb # Kaggle notebook for model training
│
├── requirements.txt
├── README.md
└── .gitignore



---

## ⚙️ Setup Instructions (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/cropshield-ai.git
cd cropshield-ai

---

2️⃣ Install dependencies
pip install -r requirements.txt

---

3️⃣ Configure environment variables
Copy .env.example → .env inside app/:
# Windows
copy app\.env.example app\.env
# macOS/Linux
cp app/.env.example app/.env
Edit app/.env with your own values:
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=cropshield_db
MODEL_PATH=app/model/cropshield_ai.h5

---

4️⃣ Add the trained model
Place your trained cropshield_ai.h5 file in:
app/model/

---

5️⃣ Run the app
streamlit run app/app.py

---

🔐 Environment Variables
Variable	Description
MYSQL_HOST	MySQL server host (e.g. localhost)
MYSQL_PORT	MySQL port (default 3306)
MYSQL_USER	MySQL username
MYSQL_PASSWORD	MySQL password
MYSQL_DB	MySQL database name
MODEL_PATH	Path to trained .h5 model

---

📊 Model Details
Architecture: MobileNetV2 (transfer learning)

Classes: 76 crop diseases

Validation Accuracy: ~93.91%

Validation Loss: 0.2071

Training: Performed on Kaggle

---

👩‍💻 Author
**Sakshi Santosh Mote
4th Year Artificial Intelligence & Data Science

GitHub: @sakshimote20
LinkedIn: [Add your LinkedIn link here]

---

## 📌 Future Improvements

📱 Mobile-friendly UI for farmers

🔄 Retrain model with larger datasets for higher accuracy

📥 Export predictions as reports (PDF/Excel)

