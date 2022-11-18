
import pickle
import time


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import requests
from flask import Flask, render_template, request


app = Flask(__name__)
model=pickle.load(open('heart.pkl','rb'))
scale=pickle.load(open('scale.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=["POST","GET"])
def predict():
    input_feature=[x for x in request.form.values()]
    feature_values=[np.array(input_feature)]
    names=[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    data=pandas.DataFrame(feature_values,columns=names)
    data=scale.fit_transform(data)
    data=pandas.DataFrame(data,columns=names)
    prediction =model.predict(data)
    pred_prob=model.predict_proba(data)
    print(prediction)
    if prediction == "Yes":
        return render_template("chance.html")
    else:
        return render_template("nochance.html")
if __name__ == "__main__": 
  app.run(debug=True)            
