# Real Estate Price Prediction using Machine Learning 

![Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction/blob/main/banner_image.png)

This project predicts real estate prices using **machine learning algorithms**, specifically **XGBoost, LightGBM, Random Forest, and Linear Regression**. The model is trained on a dataset of real estate transactions and evaluates feature importance using **SHAP values and Permutation Importance**.

This repository contains:  
- A **Google Colab notebook** for running the project.  
- A **public dataset** (`DATA2.csv`) stored in GitHub for easy access.  
- A complete **machine learning pipeline** from data preprocessing to model evaluation.

---

## **Project Overview**

Real estate price prediction is an important task for buyers, sellers, and real estate agents. This project applies **multiple machine learning models** to analyze housing prices and provides insights into the most significant features influencing property values.

### **Key Features**  
- **Data Preprocessing**: Handling missing values, feature selection, and transformation.  
- **Feature Engineering**: **Target** normalization using Yeo–Johnson transformation to reduce skewness.  
- **Model Training**: Comparing different regression models to find the best predictor.  
- **Feature Importance**: Analyzing key factors affecting house prices using SHAP values and Permutation Importance.  
- **Hyperparameter Tuning**: Optimizing XGBoost and LightGBM performance using GridSearchCV.

### Real-World Applications

- **For Real Estate Agents**  
  - Set competitive listing prices based on property features.  
  - Identify undervalued properties for quick investment opportunities.

- **For Home Buyers & Sellers**  
  - Estimate the fair price of a home before buying/selling.  
  - Compare different properties based on features.

- **For Investors & Financial Analysts**  
  - Predict housing market trends based on historical data.  
  - Optimize real estate portfolios by analyzing location-based pricing trends.

### Future Improvements

- **Deploy as a Web App**  
  A Gradio-based UI (coming soon) will allow users to enter property details and get instant price predictions.

- **Improve Data & Features**  
  Integrate real-time market data and economic indicators (interest rates, inflation, etc.).

- **Enhance Model Performance**  
  Try deep learning models (LSTMs, CNNs) for advanced price forecasting and Bayesian hyperparameter optimization.

---

## Try It Now (No Installation Required)

1. **Open the Google Colab notebook**  
   [Run in Google Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)

2. **Dataset Automatically Loaded from GitHub**  
   No manual downloads required.

3. **Just Run the Notebook!**  
   Click **Runtime → Run all** and watch the pipeline run end-to-end.

---

## **Dataset Details**  
- **File Name:** `DATA2.csv`  
- **Source:** [Melbourne Housing Market (Kaggle)](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)  
- **Target Variable:** `Price` (transformed via Yeo–Johnson)  
- **Features Include:**  
  - `Distance`: Distance from city center  
  - `BuildingArea`: Size of the property  
  - `YearBuilt`: Year of construction  
  - `Landsize`: Land area in square meters  
  - `Property Type`: House, Apartment, or Unit  
  - …and more geographic and sale metadata.

---

## **Technologies Used**  
- **Python**  
- **Google Colab**  
- **Pandas, NumPy, Matplotlib, Seaborn** (Data processing & visualization)  
- **Scikit-Learn** (Machine learning pipeline)  
- **XGBoost & LightGBM** (Boosting algorithms)  
- **SHAP & Permutation Importance** (Feature explainability)

---

## Model Performance & Insights

| Model               | Train R² | Test R² | MAE   | RMSE  | Best Use Case               |
|---------------------|---------:|--------:|------:|------:|-----------------------------|
| **Linear Regression** | 0.780  | 0.771  | 0.360 | 0.472 | Quick baseline              |
| **Random Forest**     | 0.964  | 0.841  | 0.290 | 0.394 | Handles nonlinearities well |
| **XGBoost**           | 0.915  | 0.853  | 0.282 | 0.378 | Highly accurate             |
| **LightGBM**          | 0.891  | 0.854  | 0.279 | 0.377 | **Best overall**            |

**LightGBM** achieved the highest Test R² (0.854) and lowest RMSE (0.377), making it the top choice for deployment.

---

## **Feature Importance Analysis**  
SHAP and permutation importance revealed the top five drivers of house price:

1. **Distance**  
2. **Rooms**  
3. **Property Type**  
4. **Region name**  
5. **Landsize**

### SHAP Summary Plot

![SHAP](https://github.com/svvd-m/real-estate-price-prediction/blob/4f6400b08ef7284055f65637571e111e7312b1b8/img.png)

---

## How to Reproduce This Project Locally

1. **Clone this repository**  
   ```bash
   git clone https://github.com/svvd-m/real-estate-price-prediction.git
   ```
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook locally** using Jupyter Notebook or Google Colab.

##  How This Project Demonstrates My Skills
- **Data Science & Machine Learning**: Complete ML pipeline with preprocessing, feature engineering, and evaluation.  
- **Model Optimization**: Hyperparameter tuning for XGBoost and LightGBM.  
- **Business & Domain Knowledge**: Applied ML to real estate pricing for data-driven insights.  
- **Python & Libraries**: Used Pandas, Scikit-Learn, XGBoost, LightGBM, and SHAP.  

##  Next Steps & Future Improvements
- **Build a Web App** – Interactive Gradio UI for real-time predictions.  
- **Deploy on Hugging Face Spaces** – Make the model publicly available for easy testing.  
- **Expand Dataset** – Include additional features like interest rates, crime rates, and school quality.  


##  **Links**    
- **GitHub Repo**: [Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction)  
- **Colab Notebook**: [Run in Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)  
  

