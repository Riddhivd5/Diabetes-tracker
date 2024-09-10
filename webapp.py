import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("C:/Users/Riddhi/Desktop/projects/Diabetes-SVM/trained_model.sav", 'rb'))

def diabetes_prediction(input_data):
    input_data_array = np.asarray(input_data)

    #reshape because we're only taking one data point
    input_data_reshaped = input_data_array.reshape(1, -1)

    print(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return("This person is not diabetic")
    else:
        return("This person is diabetic")


def main():
    st.title('Diabetes Prediction System')
    
    pregnancies = st.text_input("Number of pregnancies")
    glucose = st.text_input("Glucose Level")
    bp = st.text_input("BP Level")
    skin_thickness = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DPF = st.text_input("Pedigree Function")
    age = st.text_input("Age of the person")

    #code for prediction
    diagnosis = ''      #return string will be stored here

    if st.button('Test Result'):
        diagnosis = diabetes_prediction([pregnancies,glucose,bp,skin_thickness,insulin,BMI,DPF,age])

    st.success(diagnosis)      #printing



if __name__=='__main__':
    main()


