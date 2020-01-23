#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 03:08:11 2020

@author: Shoaib
"""

from flask import Flask,jsonify,request
import numpy as np
import pickle
from flask_cors import CORS, cross_origin

model = pickle.load(open('diabetesmodel.pkl', 'rb'))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/main',methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    p = list(map(float,(f"{data['age']} {data['sex']} {data['bmi']} {data['bp']} {data['s1']} {data['s2']} {data['s3']} {data['s4']} {data['s5']} {data['s6']}").split(' ')))
    t = np.reshape(p,(1,-1))
    result = model.predict(t)
    return jsonify({ 'result': result[0]})
                        
@app.route('/getter')
@cross_origin()
def predicter():
    return jsonify({ 'result': 'sachin'})                        
               

if __name__ == '__main__':    
    app.run(debug=True,port=5000)
