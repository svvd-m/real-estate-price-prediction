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

##  **Run the Project in Google Colab**  
### Step 1: Open the Colab Notebook  
Click the link below to open the project in **Google Colab**:  
 **[Run in Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)**  

### Step 2: Load the Dataset  
The dataset is stored in **this GitHub repository**.  
In the Colab notebook, the dataset is automatically fetched using the following code:  

```python
# Load dataset directly from GitHub
file_url = "https://raw.githubusercontent.com/svvd-m/real-estate-price-prediction/main/DATA2.csv"
df = pd.read_csv(file_url)
```

### Step 3: Run the Notebook  
1. Click **"Runtime"** → **"Run all"** in Google Colab.  
2. The models will train, evaluate, and visualize results automatically.  


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


##  **Machine Learning Models Evaluated**  
| Model               | Train R² | Test R² | MAE  | RMSE  |  
|---------------------|---------|---------|------|-------|  
| **Linear Regression** | 0.65  | 0.62  | 48,500 | 75,300 |  
| **Random Forest**    | 0.85  | 0.80  | 32,100 | 52,700 |  
| **XGBoost**         | 0.88  | 0.83  | 29,800 | 49,200 |  
| **LightGBM**        | 0.87  | 0.82  | 30,500 | 50,300 |  


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


##  **Contributing**  
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**.


##  **License**  
This project is open-source under the **MIT License**.


##  **Links**    
 **GitHub Repo**: [Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction)  
 **Colab Notebook**: [Run in Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)  
  

