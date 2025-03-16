# Real Estate Price Prediction using Machine Learning 

![Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction/blob/main/banner_image.png)

 This project predicts real estate prices using **machine learning algorithms**, specifically **XGBoost, LightGBM, Random Forest, and Linear Regression**. The model is trained on a dataset of real estate transactions and evaluates feature importance using **SHAP values and Permutation Importance**.  

This repository contains:  
- A **Google Colab notebook** for running the project.  
- A **public dataset** (`DATA2.csv`) stored in GitHub for easy access.  
- A complete **machine learning pipeline** from data preprocessing to model evaluation.  


##  **Project Overview**  
 Real estate price prediction is an important task for buyers, sellers, and real estate agents. This project applies **multiple machine learning models** to analyze housing prices and provides insights into the most significant features influencing property values.

### **Key Features**  
- **Data Preprocessing**: Handling missing values, feature selection, and transformation.  
- **Feature Engineering**: Normalization using Yeo-Johnson transformation to reduce skewness.  
- **Model Training**: Comparing different regression models to find the best predictor.  
- **Feature Importance**: Analyzing key factors affecting house prices using SHAP values.  
- **Hyperparameter Tuning**: Optimizing model performance using GridSearchCV.  

### Real-World Applications
- For Real Estate Agents:
Set competitive listing prices based on property features.
Identify undervalued properties for quick investment opportunities.

- For Home Buyers & Sellers:
Estimate the fair price of a home before buying/selling.
Compare different properties based on features.

- For Investors & Financial Analysts:
Predict housing market trends based on historical data.
Optimize real estate portfolios by analyzing location-based pricing trends.

### Future Improvements
- Deploy as a Web App
A Gradio-based UI (coming soon) will allow users to enter property details and get instant price predictions.

- Improve Data & Features
Integrate real-time real estate market data for more accurate predictions.
Add economic indicators (interest rates, inflation, etc.).

- Enhance Model Performance
Try deep learning models (e.g., LSTMs, CNNs) for advanced price forecasting.
Tune hyperparameters further using Bayesian Optimization.

##  Try It Now (No Installation Required)
 **Click Below to Open the Fully Functional Colab Notebook**  
 [Run in Google Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)  

 **Dataset Automatically Loaded from GitHub**  
No manual downloads required. The dataset is fetched automatically in the notebook.

 **Just Run the Notebook!**  
Click **"Runtime" → "Run all"** and see the model train, evaluate, and predict in minutes.

##  **Dataset Details**  
- **File Name:** `DATA2.csv`  
- **Source:** Real estate transaction data  
- **Target Variable:** `Price` (House sale price)  
- **Features Include:**  
  - `Distance`: Distance from city center  
  - `BuildingArea`: Size of the property  
  - `YearBuilt`: Year of construction  
  - `Landsize`: Land area in square meters  
  - `Property Type`: House, Apartment, or Unit  
  - **And many more...**  


## **Technologies Used**  
- **Python**   
- **Google Colab**   
- **Pandas, NumPy, Matplotlib, Seaborn** (Data processing & visualization)  
- **Scikit-Learn** (Machine Learning models)  
- **XGBoost & LightGBM** (Boosting algorithms)  
- **SHAP & Permutation Importance** (Feature explanation)  


##  Model Performance & Insights

| Model               | Train R² | Test R² | MAE  | RMSE  | Best Use Case |
|---------------------|---------|---------|------|-------|----------------------|
| **Linear Regression** | 0.65  | 0.62  | 48,500 | 75,300 | Quick Baseline Model |
| **Random Forest**    | 0.85  | 0.80  | 32,100 | 52,700 | High-dimensional data |
| **XGBoost**         | 0.88  | 0.83  | 29,800 | 49,200 | **Best Overall Model** |
| **LightGBM**        | 0.87  | 0.82  | 30,500 | 50,300 | Fast and scalable |

 **XGBoost performed the best**, making it the ideal model for **real estate price predictions**.  
 **Random Forest** works well but takes longer to train.  
 **Linear Regression** is too simple and lacks accuracy for real-world applications.  


##  **Feature Importance Analysis**  
**SHAP Values** were used to interpret feature importance. The top **5 most influential features** in predicting house prices were:  
1️. **Distance from city center**  
2️. **Building area**  
3️. **Number of bedrooms**  
4️. **Land size**  
5️. **Property type (House/Apartment)**  

### **SHAP Summary Plot**  
![](https://github.com/svvd-m/real-estate-price-prediction/blob/main/image.png)  


##  **How to Reproduce This Project Locally**  
1. **Clone this repository**:  
   ```bash
   git clone https://github.com/svvd-m/real-estate-price-prediction.git
   ```
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook locally** using Jupyter Notebook or Google Colab.

##  How This Project Demonstrates My Skills
 **Data Science & Machine Learning**: Full ML pipeline including preprocessing, feature engineering, and model evaluation.  
 **Model Optimization**: Hyperparameter tuning with GridSearchCV for best performance.  
 **Business & Domain Knowledge**: Applied ML to real estate pricing, making data-driven decisions.  
 **Python & Libraries**: Used Pandas, Scikit-Learn, XGBoost, LightGBM, and SHAP for feature analysis.  

##  Next Steps & Future Improvements
 **Build a Web App** – Create an interactive Gradio UI for instant property price predictions.  
 **Deploy on Hugging Face Spaces** – Make the model publicly available for easy testing.  
 **Expand Dataset** – Include additional features like interest rates, crime rates, and school quality.  

##  **License**  
This project is open-source under the **MIT License**.


##  **Links**    
 **GitHub Repo**: [Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction)  
 **Colab Notebook**: [Run in Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)  
  

