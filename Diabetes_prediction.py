import pickle
import sklearn
import numpy as np
# To display images
from PIL import Image
import streamlit as st



# loading the saved model
loaded_model = pickle.load(open("trained_model.sav",'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for iine instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    #display image
    img = Image.open("Diabetes.jpg")
    new_image = img.resize((700, 200))
    st.image(new_image)
    # lets display
    # st.image(img, width=700)

    # giving a title
    st.title('Diabetes Prediction Web App')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunctiom = st.text_input('Dibetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for prediction
    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunctiom, Age)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
