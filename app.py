#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 03:08:11 2020

@author: Shoaib
"""

from flask import Flask,jsonify,request
import numpy as np
import pickle

model = pickle.load(open('diabetesmodel.pkl', 'rb'))

app = Flask(__name__)

if __name__ == '__main__':
    @app.route('/',methods=['POST'])
    def predict():
        data = request.get_json()
        p = list(map(float,(f"{data['age']} {data['sex']} {data['bmi']} {data['bp']} {data['s1']} {data['s2']} {data['s3']} {data['s4']} {data['s5']} {data['s6']}").split(' ')))
        t = np.reshape(p,(1,-1))
        result = model.predict(t)
        return jsonify({ 'result': result[0]})
        
    app.run(debug=True)
