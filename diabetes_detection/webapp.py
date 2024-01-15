# description: diabetes detection using machine learning

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



# title and subtitle
st.write('''
# Diabetes Detection
Using machine learning to detect diabetes
''')

# open and display an image
image = Image.open('media/machine-learning.jpeg')
st.image(image, caption='ML', use_column_width=True)

# read data
df = pd.read_csv('data/diabetes.csv')
st.subheader('Data Info')
st.dataframe(df)

# show statistics
st.write(df.describe())

# show chart
chart = st.bar_chart(df)

# split the data into independent 'X' and dependent 'Y' variables
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# split the data-set into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('dpf', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    user_data = {
        'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi, 'dpf': dpf, 'age': age
    }

    # transform data into dataframe
    features_df = pd.DataFrame(user_data, index=[0])
    return features_df


user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

# create and train the model
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)

# show the model metrics
st.subheader('Model Test Accuracy Score:')
score = accuracy_score(y_test, random_forest_classifier.predict(x_test)) * 100
st.write(f'{score}%')

# store the models predictions in a variable
prediction = random_forest_classifier.predict(user_input)

# display the classification
st.subheader('Classification:')
st.write(prediction)
