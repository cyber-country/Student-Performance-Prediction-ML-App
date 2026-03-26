import streamlit as st
from model import model_train
import numpy as np
model,scaler,accuracy,c=model_train()
st.title("Exam prediction")
st.write("Predict whether a student will Pass or Fail based on study hours and previous marks.")
st.write("The model accuracy is :-",accuracy)
st.write("The model confusion matrix:-",c)
print(c)
hour=st.number_input("Enter study hour")
prev=st.number_input("Enter Previous marks")
if st.button("Predict"):
    data=np.array([[hour,prev]])
    data=scaler.transform(data)
    ouput=model.predict(data)
    if ouput[0]==1:
        ouput="Pass"
    else:
        ouput="Fail"
    st.write("prediction:-",ouput)
