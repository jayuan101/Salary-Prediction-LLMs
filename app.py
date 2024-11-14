import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

# Streamlit UI for Salary Prediction
st.title("Salary Prediction Application")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Salary Data CSV", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data Sample:")
    st.write(df.head())

    # Data cleaning and preprocessing
    st.write("Data Cleaning and Preprocessing...")
    df = df.dropna()  # Drop rows with any missing values
    label_encoder = LabelEncoder()
    df['Education Level'] = label_encoder.fit_transform(df['Education Level'])

    # Splitting data into features (X) and target (y)
    x = df[['Education Level', 'Years of Experience']]
    y = df[['Salary']]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9598)

    # Display shapes of the train and test sets
    st.write("Training and Testing Data Shapes:")
    st.write("X_train:", x_train.shape)
    st.write("X_test:", x_test.shape)
    st.write("y_train:", y_train.shape)
    st.write("y_test:", y_test.shape)

    # Define models
    model_options = {
        'Linear Regression': LinearRegression(),
        'KNeighbors Regressor': KNeighborsRegressor(),
        'DecisionTree Regressor': DecisionTreeRegressor(),
        'RandomForest Regressor': RandomForestRegressor()
    }

    # Model selection and training
    selected_model = st.selectbox("Select a model to train", list(model_options.keys()))

    if st.button("Train and Evaluate Model"):
        # Train the selected model
        model = model_options[selected_model]
        model.fit(x_train, y_train)
        
        # Predictions and accuracy calculation
        y_pred = model.predict(x_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        accuracy = (1 - error) * 100
        st.write(f"Accuracy of {selected_model}: {accuracy:.2f}%")

    # Compare the accuracies of all models
    if st.checkbox("Compare All Models"):
        accuracies = {}
        for model_name, model in model_options.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            error = mean_absolute_percentage_error(y_test, y_pred)
            accuracy = (1 - error) * 100
            accuracies[model_name] = accuracy
        
        st.write("Model Accuracies:")
        for model_name, accuracy in accuracies.items():
            st.write(f"{model_name}: {accuracy:.2f}%")
