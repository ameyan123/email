# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:21:53 2023

@author: ameya
"""

import numpy as np
import pickle
import streamlit as sp


loaded_model=pickle.load(open(r"C:/Users/ameya/projects/spam.pkl",'rb'))
loaded_model2=pickle.load(open(r"C:/Users/ameya/projects/vector.pkl",'rb'))

def spam(input):
    
    input_data=loaded_model2.transform(input)
    prediction=loaded_model.predict(input_data)
    print(prediction)

    if (prediction[0]==1):
        return 'Ham mail'
    else:
        return 'Spam mail'
    
def main():
    
    sp.title('EMAIL SPAM PREDICTION WEB APP')
    
    Message=sp.text_input("MESSAGE")

    prediction=''
    
    if sp.button('RESULT'):
        prediction=spam([Message])
        
    sp.success(prediction)  
    
    
    
if __name__ == '__main__':
    main()
    
    