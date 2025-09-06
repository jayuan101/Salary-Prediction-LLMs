import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
df = df.dropna()
label_encoder = LabelEncoder()
df['Education Level'] = label_encoder.fit_transform(df['Education Level'])

X = df[['Education Level', 'Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9598)

# ============================
# Train default model (Linear Regression)
# ============================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================
# Sidebar: Predict Your Salary Always
# ============================
st.sidebar.subheader("Predict Your Salary 💰")

# Education Level selectbox
edu_input = st.sidebar.selectbox(
    "Select Education Level",
    options=label_encoder.classes_,
    index=0
)
edu_encoded = label_encoder.transform([edu_input])[0]

# Years of Experience slider
exp_input = st.sidebar.slider(
    "Enter Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

# Predict salary
user_X = [[edu_encoded, exp_input]]
predicted_salary = model.predict(user_X)[0]
st.sidebar.success(f"Predicted Salary: ${predicted_salary:,.2f}")

# ============================
# Show Training Accuracy
# ============================
y_pred = model.predict(X_test)
error = mean_absolute_percentage_error(y_test, y_pred)
accuracy = (1 - error) * 100
st.subheader("Model Accuracy")
st.write(f"Linear Regression Accuracy: {accuracy:.2f}%")

# ============================
# Data Table
# ============================
st.subheader("Data Overview")
st.dataframe(df)
