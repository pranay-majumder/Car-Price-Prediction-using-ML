# pip install -r requirements.txt (Do at First)
# python application.py

from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

application=Flask(__name__)
app=application

# This Line Important for CSS File Loading and Background Image Loading
app.static_folder = 'static'

cors=CORS(app)

model=pickle.load(open('Model/RandomForestRegression.pkl','rb'))

car=pd.read_csv('Dataset/Cleaned_Car_Prediction_File.csv')

@app.route('/',methods=['GET','POST'])
def index():
    # Only For Categorical Column
    company=sorted(car['Company name'].unique())
    car_model=sorted(car['Name'].unique())
    year=sorted(car['Year'].unique(),reverse=True)
    fuel_type=car['Fuel_Type'].unique()
    transmission=car['Transmission'].unique()

    # This part I Handeled inside The html.
    #company.insert(0,'Select Company')
    #car_model.insert(0,'Select Car')

    return render_template('index.html',company_html=company, car_model_html=car_model, year_html=year, fuel_type_html=fuel_type,
                           transmission_html=transmission)


@app.route('/predictdata',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():

    companys=sorted(car['Company name'].unique())
    car_models=sorted(car['Name'].unique())
    years=sorted(car['Year'].unique(),reverse=True)
    fuel_types=car['Fuel_Type'].unique()
    transmissions=car['Transmission'].unique()
    companys.insert(0,'Select Company')

    
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    transmission=request.form.get('transmission')
    driven=request.form.get('kilo_driven')

        # Check for null or empty values, If any null value is encountered then again we redirect to same page, in order
        # to prevent the submission of the form.
    if None in [company, car_model, year, fuel_type, transmission, driven] or "" in [company, car_model, year, fuel_type, transmission, driven]:
        # Redirect to an same page if any field is empty
        return redirect('http://127.0.0.1:5000/')

    prediction=model.predict(pd.DataFrame(columns=['Name', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
       'Company name'], data=np.array([car_model,year,driven,fuel_type,transmission,company]).reshape(1, 6)))
    
    #print(prediction)
    #return str(np.round(prediction[0],2))

    # We can render Our Result to some other page Also in order to Avoid Confusion.
    #return render_template('index.html',result=str(np.round(prediction[0],2)))
    
    # Solution Find
    return render_template('index.html',company_html=companys, car_model_html=car_models, year_html=years, fuel_type_html=fuel_types,
                           transmission_html=transmissions,result=str(np.round(prediction[0],2)))


if __name__=='__main__':
    app.run(debug=True)

