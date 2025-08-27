import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Prediction App")

# Sidebar hyperparameters
st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 300, step=10, value=100)
max_depth = st.sidebar.slider("Max Depth", 2, 50, step=2, value=10)

# Load CSV
st.header("1. Load Dataset")
uploaded_file = st.file_uploader("Upload your crop_yield.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Preprocessing
    st.header("2. Data Preprocessing")
    features = df.drop(columns=['Yield'])
    target = df['Yield']

    features_encoded = pd.get_dummies(features, columns=['Crop', 'Season', 'State'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.success("✅ Data preprocessed successfully!")

    # Model training with RandomForest
    st.header("3. Train Random Forest Regressor")
    with st.spinner("Training model..."):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
    st.success("🎯 Model trained with selected hyperparameters!")

    # Evaluation
    st.header("4. Model Evaluation")
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    # User Prediction
    st.header("5. Predict Yield (Custom Input)")
    user_input = {}
    for col in features.columns:
        if df[col].dtype == object:
            user_input[col] = st.selectbox(f"{col}", sorted(df[col].unique()))
        else:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features_encoded.columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = rf.predict(input_scaled)
    st.success(f"🌱 Predicted Crop Yield: {prediction[0]:.2f}")

else:
    st.info("📂 Upload your `crop_yield.csv` file to continue.")

