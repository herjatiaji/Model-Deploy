"""
Created on 2020-11-25
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""

import pandas as pd
import joblib
from flask import Flask, redirect, url_for, request, render_template
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = Flask(__name__)

@app.route("/sentence/")
def index():
	return render_template('index.html')

@app.route('/sentence/result/', methods=["POST"])
def prediction_result():
    #receive parameter sent by client
    sentence = request.form.get('sentence')

    #start making prediction
    transformer = TfidfTransformer()
    #load the vectorizer 
    vector_model = 'vectorizer.model'
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=joblib.load(open(vector_model, "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array([sentence])))
    #load the model
    filename = 'mlp_tfidf.model'
    loaded_model= joblib.load(filename)
    #make new prediction
    result = loaded_model.predict(tfidf)
    #print(result)
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='Negative'
        elif(int_result==1):
            decision='Neutral'
        else:
            decision='Positive'
    
    #print('The sentence is ', decision)
    #return the output and load the result.html
    return render_template('result.html', sentence=sentence, status=decision)

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5002')
    app.run()