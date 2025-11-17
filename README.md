
## **Path Loss Prediction using Classical Machine Learning**

This project implements a complete machine learning pipeline for predicting wireless path loss (in dB) using classical regression models.
The dataset contains transmitter–receiver coordinates, 3D distance, LOS/NLOS indicator, building statistics, and antenna angles.
All deep learning and hybrid ensemble techniques have been removed to keep the workflow clean and lightweight.

---

## **Features**

* Full preprocessing pipeline (train-test split + standardization)
* Comparison of multiple classical ML models:

  * Linear Regression
  * Ridge
  * Lasso
  * ElasticNet
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * AdaBoost
  * XGBoost
  * CatBoost
* Automatic RMSE and R² evaluation
* Visual comparison of model performance
* Saves final results to a CSV file

---

## **Project Structure**

```
├── synthetic_pathloss_noisy.csv
├── pathloss_all_models_pipeline_ml_only_clean.py
├── requirements.txt
├── all_models_comparison.csv      # generated after running the script
└── README.md
```

---

## **Requirements**

The project uses only classical ML libraries.
All dependencies have been listed in **requirements.txt**.

Here is the expected content of your `requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
```

---

## **Installation**

Follow these steps to set up the environment.

### **1. Create a virtual environment (recommended)**

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**

```
python3 -m venv venv
source venv/bin/activate
```

---

### **2. Install all dependencies**

Once the virtual environment is active:

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This will install:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* catboost

---

## **How to Run the Script**

Run the following command:

```
python pathloss_all_models_pipeline_ml_only_clean.py
```

The script will:

1. load and preprocess the dataset
2. train and evaluate all classical ML models
3. display RMSE and R² metrics
4. generate two comparison plots
5. save the final table as **all_models_comparison.csv**

---

## **Output Files**

### **1. Model Comparison Table**

Saved automatically as:

```
all_models_comparison.csv
```

Includes:

| Model | RMSE | R² |

### **2. Visualizations**

Two charts are displayed:

* RMSE comparison (lower is better)
* R² comparison (higher is better)

---

## **Notes**

* Ensure `synthetic_pathloss_noisy.csv` exists in the project root.
* To add hyperparameter tuning or logging, you can extend the script easily.
* GPU is **not required**, since the project uses only classical ML.

---
• a notebook version
• GitHub-ready badges and styling for this README
