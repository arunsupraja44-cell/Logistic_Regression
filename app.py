import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Breast Cancer Prediction Using Logistic Regression")

df = pd.read_csv(r'Reduced_Important_Columns_Dataset.csv')
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

if st.checkbox("Show Dataset"):
    st.dataframe(df)

if st.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

if st.checkbox("Show Description"):
    st.write(df.describe())

if st.checkbox("Show Diagnosis Countplot"):
    sns.countplot(x=df["diagnosis"])
    st.pyplot()


X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.write("Accuracy:", accuracy_score(y_test, y_pred))



st.header("Prediction")
inputs = []
for col in X.columns:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    df_input = pd.DataFrame([inputs], columns=X.columns)
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)
    if pred[0] == 1:
        st.error("Malignant (Cancer Positive)")
    else:
        st.success("Benign (Cancer Negative)")
