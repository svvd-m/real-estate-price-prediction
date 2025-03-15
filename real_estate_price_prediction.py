# Required Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define the file path for the dataset (stored in the same directory as the script)
file_path = "DATA2.csv"

# Ensure the dataset is present before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"ERROR: Dataset file '{file_path}' not found. Please make sure it is in the repository.")

# Load dataset
df = pd.read_csv(file_path)

# Display initial dataset information
print("Dataset loaded successfully!")
print(df.head(10))
df.info()

# Check for missing values in the dataset
print("\nMissing Values:")
print(df.isnull().sum())

# Initial price distribution analysis
print(f"Initial Skewness: {df['Price'].skew():.3f}, Kurtosis: {df['Price'].kurtosis():.3f}")

# Visualizing original price distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("Original Price Distribution")
plt.show()

# Price Transformations for Normalization
from scipy.stats import boxcox

df['Log_Price'] = np.log1p(df['Price'])  # Log transformation
df['SQRT_Price'] = np.sqrt(df['Price'])  # Square root transformation
df['BoxCox_Price'], _ = boxcox(df['Price'] + 1)  # Box-Cox transformation (requires positive values)

# Yeo-Johnson transformation
yj_transformer = PowerTransformer(method='yeo-johnson')
df['YeoJohnson_Price'] = yj_transformer.fit_transform(df[['Price']])

# Comparing transformations
print("\nSkewness & Kurtosis After Transformations:")
for col in ['Log_Price', 'SQRT_Price', 'BoxCox_Price', 'YeoJohnson_Price']:
    print(f"{col}: Skewness={df[col].skew():.3f}, Kurtosis={df[col].kurtosis():.3f}")

# Selecting the most suitable transformation
df['Price'] = df['YeoJohnson_Price']
df.drop(columns=['Log_Price', 'SQRT_Price', 'BoxCox_Price', 'YeoJohnson_Price'], inplace=True)

print("\nUsing Yeo-Johnson Transformation for normalization.")

# Removing unnecessary columns
cols_to_drop = ['Sl no', 'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Bedroom2']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

# Handling missing values
num_cols = ['Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

cat_cols = ['CouncilArea', 'Regionname', 'Type', 'Suburb']
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nData preprocessing complete.")

# Splitting dataset into training and testing sets
y = df['Price']
X = df.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# One-Hot Encoding for categorical features
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Encoding & Scaling complete.")

# Defining models for comparison
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42)
}

# Model evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return train_score, test_score, mae, rmse

# Evaluating models
model_results = {}
for name, model in models.items():
    train_r2, test_r2, mae, rmse = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    model_results[name] = [train_r2, test_r2, mae, rmse]
    print(f"{name}: Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Saving models
import joblib
joblib.dump(models["XGBoost"], "xgboost_model.pkl")
joblib.dump(models["LightGBM"], "lightgbm_model.pkl")

print("\nModels trained and saved.")

# SHAP Analysis
explainer_xgb = shap.TreeExplainer(models["XGBoost"])
shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)
shap.summary_plot(shap_values_xgb, X_test_scaled, feature_names=X_train.columns)

explainer_lgbm = shap.TreeExplainer(models["LightGBM"])
shap_values_lgbm = explainer_lgbm.shap_values(X_test_scaled)
shap.summary_plot(shap_values_lgbm, X_test_scaled, feature_names=X_train.columns)

print("\nSHAP Analysis completed.")

# Residuals plot
residuals_xgb = y_test - models["XGBoost"].predict(X_test_scaled)
plt.figure(figsize=(7,5))
sns.histplot(residuals_xgb, bins=50, kde=True)
plt.title("Residual Distribution - XGBoost")
plt.show()

residuals_lgbm = y_test - models["LightGBM"].predict(X_test_scaled)
plt.figure(figsize=(7,5))
sns.histplot(residuals_lgbm, bins=50, kde=True)
plt.title("Residual Distribution - LightGBM")
plt.show()

print("\nScript execution complete.")
