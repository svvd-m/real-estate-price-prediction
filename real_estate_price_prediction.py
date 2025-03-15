# Install dependencies if needed
#!pip install --upgrade scikit-learn xgboost lightgbm shap

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import gunicorn

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

# XGBoost & LightGBM
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load dataset (Ensure DATA2.csv is in the GitHub repository)
file_path = "DATA2.csv"
df = pd.read_csv(file_path)

print("Dataset successfully loaded.")
display(df.head(10))

# Basic Data Overview
print("\nDataset Information:")
df.info()

print("\nMissing Values:")
display(df.isnull().sum())

# Check the distribution of 'Price'
print(f"Initial Skewness: {df['Price'].skew():.3f}, Kurtosis: {df['Price'].kurtosis():.3f}")

# Visualize price distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title("Original Price Distribution")
plt.show()

from scipy.stats import boxcox

# Transformations to reduce skewness
df['Log_Price'] = np.log1p(df['Price'])  # Log transformation
df['SQRT_Price'] = np.sqrt(df['Price'])  # Square root transformation

# Box-Cox transformation (requires positive values)
df['BoxCox_Price'], _ = boxcox(df['Price'] + 1)  # Adding 1 to avoid zero-values issue

# Yeo-Johnson transformation (handles zero & negative values)
yj_transformer = PowerTransformer(method='yeo-johnson')
df['YeoJohnson_Price'] = yj_transformer.fit_transform(df[['Price']])

# Compare distributions visually
fig, axes = plt.subplots(2, 2, figsize=(12,10))
sns.histplot(df['Log_Price'], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title("Log Price Distribution")

sns.histplot(df['SQRT_Price'], bins=50, kde=True, ax=axes[0,1])
axes[0,1].set_title("Square Root Price Distribution")

sns.histplot(df['BoxCox_Price'], bins=50, kde=True, ax=axes[1,0])
axes[1,0].set_title("Box-Cox Price Distribution")

sns.histplot(df['YeoJohnson_Price'], bins=50, kde=True, ax=axes[1,1])
axes[1,1].set_title("Yeo-Johnson Price Distribution")

plt.show()

# Use Yeo-Johnson transformation as it produces the most normal distribution
df['Price'] = df['YeoJohnson_Price']
df.drop(columns=['Log_Price', 'SQRT_Price', 'BoxCox_Price', 'YeoJohnson_Price'], inplace=True)

print("\nApplied Yeo-Johnson transformation for normalization.")

# Drop unnecessary columns
cols_to_drop = ['Sl no', 'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Bedroom2']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

# Fill missing numeric values with median
num_cols = ['Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
            'Lattitude', 'Longtitude', 'Propertycount', 'YearSold', 'MonthSold', 'YearMonthSold']

for col in num_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
cat_cols = ['CouncilArea', 'Regionname', 'Type', 'Suburb']
for col in cat_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nData preprocessing completed.")

# Split dataset into features and target
y = df['Price']
X = df.drop('Price', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# One-Hot Encode categorical variables
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align training and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Encoding and scaling completed.")

# Feature importance analysis
from sklearn.inspection import permutation_importance
perm_importance_xgb = permutation_importance(XGBRegressor(), X_test_scaled, y_test, scoring="r2", n_repeats=5, random_state=42)
xgb_feature_importance = pd.DataFrame({"Feature": X_train.columns, "Importance": perm_importance_xgb.importances_mean})
xgb_feature_importance = xgb_feature_importance.sort_values(by="Importance", ascending=False)

print("\nXGBoost Feature Importance:")
display(xgb_feature_importance.head(10))

# Compute VIF for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("\nVIF Analysis (Features with High Multicollinearity):")
display(vif_data[vif_data["VIF"] > 5])
