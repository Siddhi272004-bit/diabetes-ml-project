# 🔥 Diabetes Prediction Model with ROC Curve 💯

## 📌 Overview
This project builds a **Diabetes Prediction Model** using Machine Learning techniques. The model is trained on the `diabetes.csv` dataset and evaluates its performance using an **ROC Curve**. The model is then deployed using **Flask**.

## 📂 Project Structure
```
├── app.py                 # Flask web application
├── model.py               # Machine learning model script
├── diabetes.csv           # Dataset used for training
├── diabetes_model.pkl     # Saved ML model
├── Figure_1.png           # Model visualization
├── ROC CURVE.png          # ROC Curve of the model
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Siddhi272004-bit/diabetes-ml-project.git
cd diabetes-ml-project
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```sh
python app.py
```

Access the web app at **http://127.0.0.1:5000/** in your browser.

## 📊 Model Performance
The **ROC Curve** is used to evaluate the performance of the model.

![ROC Curve](ROC%20CURVE.png)

## 🤖 Technologies Used
- **Python**
- **Streamlit** (for web deployment)
- **Scikit-learn** (for machine learning)
- **Matplotlib & Seaborn** (for data visualization)
- **Pandas & NumPy** (for data manipulation)

## 🎯 Features
- Train a machine learning model on **diabetes dataset**.
- Save the trained model using **Pickle**.
- Serve predictions through a **Flask web app**.
- Visualize performance using **ROC Curve**.

## 📌 Future Improvements
- Implement **JWT Authentication** for secure API access.
- Deploy using **AWS/GCP/Heroku**.
- Improve model performance with **Hyperparameter Tuning**.

## 🛠 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the **MIT License**.

---
### ⭐ Don't forget to **star** this repository if you found it helpful! 🚀
