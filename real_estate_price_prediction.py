# Install dependencies if missing (use requirements.txt in GitHub)
try:
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import shap
    from scipy.stats import boxcox
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.inspection import permutation_importance
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Some required libraries are missing. Install them using: pip install -r requirements.txt")

# Load dataset from GitHub
import os
import urllib.request

GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY/main/DATA2.csv"
file_path = "DATA2.csv"

# Download dataset if not present locally
if not os.path.exists(file_path):
    print("Downloading dataset from GitHub...")
    urllib.request.urlretrieve(GITHUB_RAW_URL, file_path)

# Load dataset
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print(df.head(10))

# Basic Data Overview
print("\nData Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Check the distribution of 'Price'
print(f"Initial Skewness: {df['Price'].skew():.3f}, Kurtosis: {df['Price'].kurtosis():.3f}")

# Visualize raw price distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("Original Price Distribution")
plt.show()

# Apply transformations
df['Log_Price'] = np.log1p(df['Price'])  
df['SQRT_Price'] = np.sqrt(df['Price'])  
df['BoxCox_Price'], _ = boxcox(df['Price'] + 1)  
yj_transformer = PowerTransformer(method='yeo-johnson')
df['YeoJohnson_Price'] = yj_transformer.fit_transform(df[['Price']])

print("\nSkewness & Kurtosis After Transformations:")
print(f"YeoJohnson_Price: Skewness={df['YeoJohnson_Price'].skew():.3f}, Kurtosis={df['YeoJohnson_Price'].kurtosis():.3f}")

# Use Yeo-Johnson transformed price
df['Price'] = df['YeoJohnson_Price']
df.drop(columns=['Log_Price', 'SQRT_Price', 'BoxCox_Price', 'YeoJohnson_Price'], inplace=True)
print("\nSelected Yeo-Johnson Transformation for Normalization.")

# Data preprocessing
df.drop(columns=['Sl no', 'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Bedroom2'], inplace=True, errors='ignore')

num_cols = ['Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']
for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

cat_cols = ['CouncilArea', 'Regionname', 'Type', 'Suburb']
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nData Preprocessing Completed.")

# Train-test split
y = df['Price']
X = df.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# One-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Encoding & Scaling Complete.")

# Train and Evaluate Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42)
}

model_results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test_scaled))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled)))
    model_results[name] = [test_score, mae, rmse]
    print(f"{name}: Test R²: {test_score:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Feature Importance for XGBoost
xgb_importance = permutation_importance(models["XGBoost"], X_test_scaled, y_test, n_repeats=5, random_state=42)
xgb_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_importance.importances_mean}).sort_values(by='Importance', ascending=False)

print("\nXGBoost Feature Importance:")
print(xgb_feature_importance.head(10))

print("\nModels trained successfully.")
