import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load data
df = pd.read_csv('Salary Data.csv')
df1 = df.dropna()
label_encoder = LabelEncoder()
df1['Education Level'] = label_encoder.fit_transform(df1['Education Level'])

# Prepare features and target
x = df1[['Education Level', 'Years of Experience']]
y = df1[['Salary']]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9598)

# Model options
model_options = {
    'Linear Regression': LinearRegression(),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'DecisionTree Regressor': DecisionTreeRegressor(),
    'RandomForest Regressor': RandomForestRegressor()
}

# Streamlit UI
st.title("Salary Prediction Application")
selected_model = st.selectbox("Select a model to train", list(model_options.keys()))

if st.button("Train and Evaluate Model"):
    model = model_options[selected_model]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    error = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = (1 - error) * 100
    st.write(f"Accuracy of {selected_model}: {accuracy:.2f}%")