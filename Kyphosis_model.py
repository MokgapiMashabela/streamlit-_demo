import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the model to predict on the data 
pickle_in = open('final_model.sav', 'rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(age, number, start): 

	prediction = classifier.predict( 
		[[age, number, start]]) 
	print(prediction) 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("Kyphosis Prediction") 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	age = st.text_input("Age", "Type Here") 
	number = st.text_input("Number", "Type Here") 
	start = st.text_input("Start", "Type Here") 
	result ="" 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		result = prediction(age, number, start) 
	st.success('The output is {}'.format(result)) 
	
if __name__=='__main__': 
	main() 
