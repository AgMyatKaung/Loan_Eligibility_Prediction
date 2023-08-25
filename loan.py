import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv('Training Data.csv')

# Drop the 'Risk_Flag' column
X = data.drop(columns=['Risk_Flag'])

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
X['Married/Single'] = label_encoder.fit_transform(X['Married/Single'])
X['House_Ownership'] = label_encoder.fit_transform(X['House_Ownership'])
X['Car_Ownership'] = label_encoder.fit_transform(X['Car_Ownership'])
X['Profession'] = label_encoder.fit_transform(X['Profession'])
X['CITY'] = label_encoder.fit_transform(X['CITY'])
X['STATE'] = label_encoder.fit_transform(X['STATE'])

# Split the data into features and labels
y = data['Risk_Flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    'Perceptron': Perceptron(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC()
}

# Train and evaluate models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Print the best model and its accuracy
print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)

# Streamlit UI
st.title("Loan Prediction App")

# Collect input from user
income = st.number_input("Income")
age = st.number_input("Age")
experience = st.number_input("Experience")
married_single = st.selectbox("Marital Status", ['Married', 'Single'])
house_ownership = st.selectbox("House Ownership", ['Own', 'Rent', 'Mortgage'])
car_ownership = st.selectbox("Car Ownership", ['Yes', 'No'])
profession = st.selectbox("Profession", X['Profession'].unique())
city = st.selectbox("City", X['CITY'].unique())
state = st.selectbox("State", X['STATE'].unique())
current_job_years = st.number_input("Current Job Years")
current_house_years = st.number_input("Current House Years")

# Preprocess user input
user_input = pd.DataFrame({
    'Income': [income],
    'Age': [age],
    'Experience': [experience],
    'Married/Single': [label_encoder.transform([married_single])[0]],
    'House_Ownership': [label_encoder.transform([house_ownership])[0]],
    'Car_Ownership': [label_encoder.transform([car_ownership])[0]],
    'Profession': [label_encoder.transform([profession])[0]],
    'CITY': [label_encoder.transform([city])[0]],
    'STATE': [label_encoder.transform([state])[0]],
    'CURRENT_JOB_YRS': [current_job_years],
    'CURRENT_HOUSE_YRS': [current_house_years]
})

# Standardize user input
user_input = scaler.transform(user_input)

# Predict using the best model
prediction = best_model.predict(user_input)

# Display prediction
if prediction[0] == 0:
    st.write("Congratulations! You are eligible for a loan.")
else:
    st.write("Sorry, you are not eligible for a loan.")
