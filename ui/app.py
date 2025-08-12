import streamlit as str
import joblib
import pandas as pd

str.title('Heart Disease Diagnosis Prediction')

age = str.number_input("Age (In years)")
sex = str.selectbox("Sex", ("Male", "Female"))
cp = str.selectbox("Chest Pain", (1, 2, 3, 4))
trestbps = str.number_input("Resting Blood Pressure (In mm Hg)")
chol = str.number_input("Serum cholesterol")
fbs = str.selectbox("Fasting blood sugar", ("True", "False"))
restecg = str.selectbox("Resting electrocardiographic results", (0, 2))
thalach = str.number_input("Maximum heart rate achieved")
exang = str.selectbox("Exercise induced angina", ("yes", "no"))
oldpeak = str.number_input("ST depression induced by exercise relative to rest")
slope = str.selectbox("The slope of the peak exercise ST segment", (1, 2, 3))
ca = str.selectbox("Number of major vessels", (0, 1, 2, 3))
thal = str.selectbox("Thalassemia", ("normal", "fixed defect", "reversible defect"))

X = [[]]
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']

sex_t = 1
if sex == "Male":
    sex_t = 1
else:
    sex_t = 0

fbs_t = 1
if sex == "True":
    fbs_t = 1
else:
    fbs_t = 0

exang_t = 1
if exang == "yes":
    exang_t = 1
else:
    exang_t = 0

thal_t = 3
if thal == "normal":
    thal_t = 3
elif thal == "fixed defect":
    thal_t = 6
else:
    thal_t = 7

X[0].append(age)
X[0].append(sex_t)
X[0].append(cp)
X[0].append(trestbps)
X[0].append(chol)
X[0].append(fbs_t)
X[0].append(restecg)
X[0].append(thalach)
X[0].append(exang_t)
X[0].append(oldpeak)
X[0].append(slope)
X[0].append(ca)
X[0].append(thal_t)

X = pd.DataFrame(X, columns=cols)

model = joblib.load("../models/final_model.joblib")['support_vectors_randomized']

num = model.predict(X)

result = ""

if num == 0:
    result = "absence"
else:
    result = "presence"

if result == "presence":
    str.markdown(
        f"<h2 style='color: red; text-align: center;'>Heart Disease Diagnosis: {result.capitalize()}</h2>",
        unsafe_allow_html=True
    )
else:
    str.markdown(
        f"<h2 style='color: green; text-align: center;''>Heart Disease Diagnosis: {result.capitalize()}</h2>",
        unsafe_allow_html=True
    )



