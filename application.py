from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__,static_url_path="/static")
app=application

#load piclke file
scaler=pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("regression.pkl", "rb"))


#route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction
@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    result = 0
    
    if request.method=='POST':
        hours_studied=float(request.form.get("hours_studied"))
        previous_score = float(request.form.get('previous_score'))
        actiivity = float(request.form.get('activities'))
        sleep_hour = float(request.form.get('sleep_hours'))
        sample_paper = float(request.form.get('paper_solved'))

        new_data=scaler.transform([[
            hours_studied,previous_score,actiivity,sleep_hour,sample_paper
        ]])

        predict=model.predict(new_data)
            
        return render_template('output.html',result = predict)

    else:
        return render_template('predict.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")