import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open("C:/Users/Riddhi/Desktop/projects/Diabetes-SVM/trained_model.sav", 'rb'))

input_data = (8,183,64,0,0,23.3,0.672,32)

input_data_array = np.asarray(input_data)

#reshape because we're only taking one data point
input_data_reshaped = input_data_array.reshape(1, -1)

print(input_data_reshaped)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("This person is not diabetic")
else:
    print("This person is diabetic")