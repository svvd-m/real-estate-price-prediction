# Real Estate Price Prediction using Machine Learning

![Real Estate Price Prediction](https://github.com/svvd-m/real-estate-price-prediction/blob/main/banner_image.png)

This project predicts real estate prices using advanced machine learning algorithms such as **XGBoost**, **LightGBM**, **Random Forest**, and **Linear Regression**. The model is trained on real-world Melbourne housing data and includes **interpretability tools** (SHAP & Permutation Importance) and a **Gradio-powered web app** for real-time predictions.

---

## Project Overview

This end-to-end project builds a pipeline for:
- Cleaning and transforming real estate transaction data
- Engineering features including **Yeo–Johnson normalization** of target values
- Comparing and tuning machine learning models
- Explaining predictions using SHAP values
- Deploying a safe and functional Gradio web app

---

## Key Features

- **Data Preprocessing**: Handle missing values, drop irrelevant columns, remove outliers  
- **Feature Transformation**: Normalize `Price` using Yeo–Johnson method  
- **Model Training**: Linear Regression, Random Forest, XGBoost, LightGBM  
- **Hyperparameter Tuning**: Optimize boosting models with GridSearchCV  
- **Feature Importance**: Visualized with SHAP & permutation importance  
- **Web Deployment**: Clean, validated Gradio UI for end users

---

## Real-World Applications

- **Agents**: Recommend competitive listing prices  
- **Buyers/Sellers**: Estimate fair value before transactions  
- **Analysts**: Study location-based price trends and feature influence

---

## Gradio Web App

The deployed Gradio app allows users to input:
- Number of rooms  
- Distance from city center  
- Land size  
- Building area  
- Property type  

and receive a **real-time house price prediction**.

### Input Safety Features
- No blank/negative values allowed  
- Caps extreme inputs to avoid skew  
- Friendly error messages for invalid cases

**Run it directly via Colab:**  
[Launch Gradio App in Colab](https://colab.research.google.com/drive/1tjUv2aYLBIJAG39ViEM_vrSZnFT3De10?usp=sharing)

---

## Dataset Details

- **Filename**: `DATA2.csv`  
- **Source**: [Melbourne Housing Market (Kaggle)](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)  
- **Target**: `Price` (transformed via Yeo–Johnson)  
- **Notable Features**:  
  - `Distance`, `Rooms`, `Landsize`, `BuildingArea`, `Regionname`, `Property Type`

---

## Technologies Used

- **Python**  
- **Google Colab / Jupyter Notebook**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **Scikit-Learn**  
- **XGBoost & LightGBM**  
- **SHAP, Permutation Importance**  
- **Gradio** (Web Interface)

---

## Model Performance

| Model             | Train R² | Test R² | MAE   | RMSE  | Notes                      |
|------------------|---------:|--------:|------:|------:|----------------------------|
| Linear Regression| 0.776    | 0.775   | 0.365 | 0.474 | Baseline                   |
| Random Forest    | 0.964    | 0.843   | 0.291 | 0.396 | Good, nonlinear            |
| XGBoost          | 0.913    | 0.856   | 0.283 | 0.380 | Accurate and efficient     |
| LightGBM         | 0.887    | 0.862   | 0.278 | 0.372 | Top performer pre-tuning   |
| Tuned XGBoost    | —        | 0.862   | 0.274 |   —   | Optimized via GridSearchCV |
| Tuned LightGBM   | —        | 0.866   | 0.274 |   —   | **Best overall**           |

---

## Feature Importance

SHAP and permutation importance revealed the most influential features:

1. **Distance from city center**  
2. **Number of rooms**  
3. **Property type**  
4. **Region name**  
5. **Landsize**

### SHAP Plot

![SHAP](https://github.com/svvd-m/real-estate-price-prediction/blob/9f468b023d0333d6a9c8544a5aef93ab9e2c285e/SHAP.png)

---

## Testing & Validation Summary

| Test Case | Status |
|-----------|--------|
| Blank input | ✅ Rejected |
| Negative/zero input | ✅ Rejected |
| Large values | ✅ Clipped and handled |
| Real-world mid-range input | ✅ Accurate prediction |
| Extreme edge cases | ✅ Stable and bounded output |

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

3. **Run the notebook or launch the Gradio app from Google Colab**

## Skills Demonstrated

- ML pipeline construction
- Feature selection and transformation
- Model evaluation and tuning
- Interpretability with SHAP
- UI development with Gradio
- Input validation and safe deployment

## Next Steps

- Deploy on Hugging Face Spaces
- Add economic indicators (e.g., interest rates)
- Include interactive location visualizations
- Use real-time APIs for updated housing data








