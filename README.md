# Real Estate Price Prediction (Machine Learning Project)

![Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction/blob/main/banner_image.png)

This project predicts real estate prices in Melbourne using machine learning models such as **XGBoost**, **LightGBM**, **Random Forest**, and **Linear Regression**.  
The workflow includes **data cleaning**, **feature engineering**, **model comparison**, **explainability with SHAP**, and a **Gradio web app** for real-time predictions.

---

## Project Overview

This project follows an end-to-end workflow:

- Cleaning and preparing the dataset  
- Creating new features  
- Training and comparing models  
- Tuning the models  
- Explaining predictions using SHAP  
- Building a small Gradio app  

---

## Key Features

- **Data Preprocessing**: handle missing values, remove outliers, drop unused columns  
- **Feature Transformation**: normalise `Price` using Yeo–Johnson  
- **Model Training**: Linear Regression, Random Forest, XGBoost, LightGBM  
- **Model Tuning**: improved results using GridSearchCV  
- **Explainability**: SHAP values and permutation importance  
- **Web App**: simple Gradio UI with input validation  

---

## Real-World Applications

- **Agents**: estimate listing prices  
- **Buyers or Sellers**: understand property value  
- **Analysts**: study trends and housing behaviour  

---

## Gradio Web App

The Gradio app takes:

- number of rooms  
- distance from city  
- land size  
- building area  
- property type  

and returns a **real-time price prediction**.

### Input Safety Features
- no blank inputs  
- no negative values  
- very high values are capped  
- friendly error messages  

**Run in Colab:**  
(Add Colab link here)

---

## Dataset Details

- **File**: `DATA2.csv`  
- **Source**: Melbourne Housing Market (Kaggle)  
- **Target**: `Price` (Yeo–Johnson transformed)  
- **Main Features**:  
  - `Rooms`  
  - `Distance`  
  - `Landsize`  
  - `BuildingArea`  
  - `Regionname`  
  - `PropertyType`  

---

## Technologies Used

- **Python**  
- **Google Colab / Jupyter Notebook**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **Scikit-Learn**  
- **XGBoost and LightGBM**  
- **SHAP and permutation importance**  
- **Gradio**  

---

## Model Performance

| Model             | Train R² | Test R² | MAE   | RMSE  | Notes                      |
|------------------|---------:|--------:|------:|------:|----------------------------|
| Linear Regression | 0.776    | 0.775   | 0.365 | 0.474 | Baseline model             |
| Random Forest     | 0.964    | 0.843   | 0.291 | 0.396 | Good for non-linear data   |
| XGBoost           | 0.913    | 0.856   | 0.283 | 0.380 | Strong balanced model      |
| LightGBM          | 0.887    | 0.862   | 0.278 | 0.372 | Best before tuning         |
| Tuned XGBoost     | —        | 0.862   | 0.274 | —     | Improved after tuning      |
| Tuned LightGBM    | —        | 0.866   | 0.274 | —     | **Best overall**           |

---

## Feature Importance

The most important features based on SHAP and permutation importance were:

1. **Distance from city**  
2. **Number of rooms**  
3. **Property type**  
4. **Region name**  
5. **Landsize**

### SHAP Plot

![SHAP](https://github.com/svvd-m/real-estate-price-prediction/blob/main/SHAP.png)

---

## Testing & Validation Summary

| Test Case | Result |
|-----------|--------|
| Blank input | Rejected |
| Negative or zero input | Rejected |
| Very large input | Handled safely |
| Normal mid-range input | Accurate |
| Edge cases | Stable output |

---

## How to Reproduce

1. **Clone the repository**
   ```bash
   git clone https://github.com/svvd-m/real-estate-price-prediction.git
   ```

2. **Install requirements**
```bash
pip install -r requirements.txt
 ```

3. **Run the notebook or open the Gradio app**

- **Skills Demonstrated**
- **end-to-end ML pipeline**
- **data cleaning and feature engineering**
- **model comparison and evaluation**
- **hyperparameter tuning**
- **SHAP explainability**
- **building a small UI with Gradio**
- **safe input validation**
