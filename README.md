#  Real Estate Price Prediction

This project predicts real estate prices using **XGBoost, LightGBM, and Random Forest**. The dataset is preprocessed and transformed to improve model accuracy.

##  Features:
- Handles **missing values** and categorical encoding.
- Uses **Yeo-Johnson transformation** for normalizing skewed data.
- Trains **multiple models** and selects the best one.
- Provides **SHAP and permutation importance** for explainability.

##  How to Run:
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/svvd-m/real-estate-price-prediction.git
   cd real-estate-price-prediction

2. Install Dependancies
   pip install -r requirements.txt

3. Run the Script
   python real_estate_price_prediction.py

## Model Performance:
Model    	    Train R²	Test R²	MAE	  RMSE
XGBoost     	0.92	    0.89	  25.3	34.8
LightGBM    	0.91	    0.87	  26.1	36.2
RandomForest	0.89	    0.85	  28.5	38.6

## Author:
- Saad
