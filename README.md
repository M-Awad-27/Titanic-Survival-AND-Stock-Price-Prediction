# 🚀 Machine Learning Projects Showcase

Welcome to this repository containing two end-to-end Machine Learning projects, fully implemented in interactive Jupyter Notebooks. Each project is modular, self-contained, and features comprehensive Exploratory Data Analysis (EDA), data preprocessing pipelines, model selection, evaluation, and visualizations.

---

## 📂 Repository Structure

```text
d:\Antigravity\
├── README.md                               # Project documentation (this file)
├── .gitignore                              # Git exclusion rules
│
├── 🚢 Titanic Survival Classification/
│   ├── Titanic_Survival_Classification.ipynb  # Comprehensive EDA, training & evaluation
│   └── data/
│       └── titanic.csv                      # Titanic dataset from Kaggle
│
└── 📈 Stock Price Prediction/
    ├── Stock_Price_Prediction.ipynb         # EDA, Linear Regression & LSTM training
    └── data/
        ├── AAPL_2020-01-01_2023-01-01.csv   # Historical Apple stock data (Yahoo Finance)
        └── AAPL_predictions_comparison.png  # Visual evaluation of stock forecasts
```

---

## 🚢 Project 1: Titanic Survival Classification

### 📝 Overview
This project constructs binary classification models to predict passenger survival on the Titanic based on passenger characteristics (age, gender, ticket class, fare, family size, etc.) sourced from Kaggle's classic dataset.

### ⚙️ Pipeline & Methods
1. **Exploratory Data Analysis (EDA):**
   - Visualized survival rates by demographics (gender, class, age distributions).
   - Examined correlation matrices of numerical attributes.
2. **Data Cleansing & Imputation:**
   - Handled missing data for `Age` (median imputation grouped by passenger class/gender) and `Embarked`.
   - Extracted social titles (e.g., Mr, Mrs, Miss, Master, Rare) from passenger names to infer status.
3. **Feature Engineering:**
   - Engineered new features such as `FamilySize` (Parch + SibSp + 1) and `IsAlone`.
   - Categorized fare ranges and applied one-hot encoding to categorical attributes.
4. **Model Training & Evaluation:**
   - Evaluated multiple classifier algorithms:
     - **Logistic Regression** (baseline classification)
     - **Random Forest Classifier**
     - **Gradient Boosting Classifier**
   - Compared performance metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.

---

## 📈 Project 2: Stock Price Prediction

### 📝 Overview
This project predicts historical stock closing prices utilizing both traditional regression techniques and deep learning architectures. It utilizes Apple Inc. (AAPL) stock data spanning 2020 to 2023.

### ⚙️ Pipeline & Methods
1. **Data Preprocessing & Scaling:**
   - Extracted temporal features from historical stock information (Open, Close, High, Low, Volume).
   - Applied normalization utilizing `MinMaxScaler` to aid neural network training stability.
   - Built a sliding window/sequence generator (e.g., sequence length of 60 days to predict the next day's price).
2. **Modeling Approaches:**
   - **Baseline:** Linear Regression model mapping basic lagged indicators.
   - **Deep Learning:** Long Short-Term Memory (LSTM) recurrent neural network built in PyTorch to capture sequential/temporal patterns.
3. **Performance Metrics:**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Visualized true stock closing prices vs. predictions made by both models.

---

## 🛠️ Setup & Installation

To run these notebooks locally, follow the steps below:

### 1. Clone the Repository
```bash
git clone https://github.com/M-Awad-27/Titanic-Survival-AND-Stock-Price-Prediction.git
cd Titanic-Survival-AND-Stock-Price-Prediction
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows (Command Prompt/PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter torch
```
> **Note:** PyTorch is required to run the LSTM model inside the Stock Price Prediction notebook.

### 4. Start Jupyter Lab / Notebook
Launch Jupyter to explore and run the project files:
```bash
jupyter notebook
```

---

## 📊 Summary of Notebook Features
* **Self-Contained Data Loaders:** Automatically load and cache local CSV datasets or raise warnings if files are missing.
* **Interactive Visualizations:** Includes rich charts using `matplotlib` and `seaborn` embedded directly in the notebook outputs.
* **Modern Code Structure:** Clean, well-commented PyTorch training loops, and sklearn preprocessing pipelines.
