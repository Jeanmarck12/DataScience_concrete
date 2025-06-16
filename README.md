# Concrete Strength Prediction App

This is an interactive Streamlit app that predicts **concrete compressive strength** based on mix design parameters such as cement, water, aggregate, and age. The app allows users to:
- Select features (X) and target variable (Y) for modeling
- Adjust train-test split ratio
- Train a linear regression model
- View model performance metrics (MSE, MAE, R²)
- Visualize actual vs predicted concrete strength
- Generate automated EDA reports using ydata-profiling

## Project files
- `streamlit_app.py` — Main Streamlit app code
- `requirements.txt` — Dependencies for the app
- `concrete_data.csv` — Dataset used for training and evaluation
- `README.md` — Project description (this file)

##  How to run locally
1️ Clone this repository:
```bash
git: https://github.com/Jeanmarck12/DataScience_concrete.git
cd concrete-strength-app
