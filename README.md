# 🌿 CropShield AI

A CNN-based deep learning model that detects crop diseases from plant images with ~94% accuracy.
It provides instant predictions, supports batch uploads, and stores results in MySQL for future analysis — helping farmers improve crop management.

Built using TensorFlow/Keras (MobileNetV2), Streamlit, and MySQL.

---
- **Dataset**: [View on Kaggle](https://www.kaggle.com/datasets/sakshimote/plant-disease-dataset)
- **Training Notebook**: [View on Kaggle](https://www.kaggle.com/code/sakshimote/notebookca02a93ab5/edit)
- **Trained Model**: [View on Kaggle](https://www.kaggle.com/datasets/sakshimote/plant-disease-detection-model)

---

## 🚀 Features

- 🌿 **Disease Detection**: Upload crop images and get predictions with confidence scores.
- 📂 **Batch Prediction**: Upload multiple images for faster diagnosis.
- 🗄️ **Database Logging**: Predictions, images, and timestamps stored in MySQL
- 🔍 **Search History**: Query past predictions by plant name
- 🎨 **Streamlit UI**: Simple and user-friendly interface with background image
- 🔐 **Secure Config**: Uses .env for database credentials and model path
---

## 🧠 Tech Stack

| Tool/Library      | Purpose |
|-------------------|---------|
| `Streamlit`       | UI and frontend |
| `NumPy, Pandas, Pillow`       | Data Handling and Preprocessing |
| `TensorFlow/Keras (MobileNetV2)` | Deep Learning Model |
| `MySQL (mysql-connector-python)`           | Database |
| `dotenv`          | Environment variable management |

---

## 📁 Folder Structure

```
cropshield-ai/
│
├── app/
│   ├── app.py                # Main Streamlit app
│   ├── background.jpg        # App background image
│   ├── class_indices.json    # Class label mapping
│   ├── train.csv, val.csv, test.csv   # Dataset splits (metadata only)
│   ├── .env                  # Local DB credentials (NOT committed)
│   ├── .env.example          # Example env file (committed)
│   └── model/
│       └── cropshield_ai.h5  # Trained model (local only, gitignored)
│
├── training/
│   └── cropshield_train.ipynb   # Kaggle notebook for model training
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakshimote20/Cropshield-AI
   cd Cropshield-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Set up environment variables**

    Copy .env.example → .env inside app/:

   ```env
   # Windows
   copy app\.env.example app\.env

   ```
   Edit app/.env with your own values:

 ```env
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   MYSQL_DB=cropshield_db
   MODEL_PATH=app/model/cropshield_ai.h5

   ```
4. **Add The Trained Model**
   Place your trained .h5 file in:
   
   ```bash
   
   app/model/

   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📸 Screenshot

<img width="1889" height="885" alt="Screenshot 2025-04-22 231155" src="https://github.com/user-attachments/assets/cbe39dd3-c161-4b18-ba0f-e449e2f2c072" />


---

## 📊 Model Details

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Classes**: 76 crop diseases
- **Validation Accuracy**: ~93.91%
- **Validation Loss**: 0.2071
- **Training**: Performed on Kaggle
---

## 👩‍💻 Author

**Sakshi Mote**  
4th Year AI & Data Science Student  
GitHub: [@sakshimote](https://github.com/sakshimote20)

---




