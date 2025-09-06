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
st.title("💼 Salary Prediction Application with Multiple ML Models")

# ============================
# Load CSV automatically
# ============================
CSV_FILE = "salary_data.csv"  # <-- place your CSV here
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
# Sidebar: ML Model & User Input
# ============================
st.sidebar.subheader("Predict Your Salary 💰")

# Select ML Model
model_options = {
    "Linear Regression": LinearRegression(),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=9598),
    "Random Forest Regressor": RandomForestRegressor(random_state=9598)
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model = model_options[selected_model_name]

# Train selected model
model.fit(X_train, y_train)

# Show model accuracy
y_pred = model.predict(X_test)
error = mean_absolute_percentage_error(y_test, y_pred)
accuracy = (1 - error) * 100
st.sidebar.write(f"Model Accuracy: {accuracy:.2f}%")

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
# Main Page: Model Comparison Table
# ============================
st.subheader("📊 Model Comparison")

accuracies = {}
for name, m in model_options.items():
    m.fit(X_train, y_train)
    y_pred_m = m.predict(X_test)
    error_m = mean_absolute_percentage_error(y_test, y_pred_m)
    acc = (1 - error_m) * 100
    accuracies[name] = acc

st.dataframe(pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy (%)']))

# ============================
# Data Overview
# ============================
st.subheader("Data Overview")
st.dataframe(df)
