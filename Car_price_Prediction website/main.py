from flask import Flask,render_template,request
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app=Flask(__name__)

data=pd.read_csv('used_car_data.csv')
df=data[['Year','Kilometers_Driven','Mileage','Engine','Power','Seats','Price']]
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,:-1],df.iloc[:,-1])

#Feature Scaling
scale=StandardScaler()
scale.fit(x_train)
x_train_trans=scale.transform(x_train)

#Applying Linear Regression
model=LinearRegression()
model.fit(x_train_trans,y_train)


@app.route('/')
def page():
    return render_template('page.html')

@app.route('/aftersubmit/',methods=["GET","POST"])
def aftersubmit():
    km=float(request.form.get('km'))
    year=float(request.form.get('year'))
    kmpl=float(request.form.get('kmpl'))
    cc=float(request.form.get('cc'))
    bhp=float(request.form.get('bhp'))
    seats=float(request.form.get('seats'))

    v=scale.transform([[year,km,kmpl,cc,bhp,seats]])
    pred_price=model.predict(v)
    price=round(pred_price[0],2)

    return render_template('price.html', price=price)

app.run(debug=True,host='localhost',port=88)    



