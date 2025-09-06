import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

# Streamlit UI
st.title("💼 Salary Prediction Application")

# ============================
# Load CSV automatically
# ============================
CSV_FILE = "salary_data.csv"  # <-- put your CSV file here
try:
    df = pd.read_csv(CSV_FILE)
    st.success(f"✅ Loaded '{CSV_FILE}' successfully!")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

st.subheader("Data Sample")
st.write(df.head())

# ============================
# Data Cleaning & Preprocessing
# ============================
st.subheader("Data Cleaning and Preprocessing")
df = df.dropna()  # Drop rows with missing values
label_encoder = LabelEncoder()
df['Education Level'] = label_encoder.fit_transform(df['Education Level'])

# Features and target
X = df[['Education Level', 'Years of Experience']]
y = df['Salary']  # 1D target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9598)
st.write("Training and Testing Data Shapes:")
st.write(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
st.write(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ============================
# Model Options
# ============================
model_options = {
    'Linear Regression': LinearRegression(),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=9598),
    'Random Forest Regressor': RandomForestRegressor(random_state=9598)
}

selected_model = st.selectbox("Select a model to train", list(model_options.keys()))

trained_model = None
if st.button("Train and Evaluate Model"):
    model = model_options[selected_model]
    model.fit(X_train, y_train)
    trained_model = model

    y_pred = model.predict(X_test)
    error = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = (1 - error) * 100
    st.success(f"Accuracy of {selected_model}: {accuracy:.2f}%")

# Compare all models
if st.checkbox("Compare All Models"):
    accuracies = {}
    for name, model in model_options.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = mean_absolute_percentage_error(y_test, y_pred)
        accuracy = (1 - error) * 100
        accuracies[name] = accuracy

    st.subheader("📊 Model Accuracies")
    for name, acc in accuracies.items():
        st.write(f"{name}: {acc:.2f}%")

# ============================
# Predict Salary for User Input
# ============================
if trained_model:
    st.sidebar.subheader("Predict Your Salary 💰")

    # User input for Education Level
    edu_input = st.sidebar.selectbox(
        "Select Education Level",
        options=label_encoder.classes_
    )
    edu_encoded = label_encoder.transform([edu_input])[0]

    # User input for Years of Experience
    exp_input = st.sidebar.number_input(
        "Enter Years of Experience",
        min_value=0,
        max_value=50,
        value=1
    )

    # Predict salary
    user_X = [[edu_encoded, exp_input]]
    predicted_salary = trained_model.predict(user_X)[0]
    st.sidebar.success(f"Predicted Salary: ${predicted_salary:,.2f}")
