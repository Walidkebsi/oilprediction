#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:55:04 2023

@author: walidkebsi
"""
from dash import Dash 
from dash import dcc
from dash import html
import pandas as pd
#---------------------------CREATION DU FICHIER VIDE PY------------------------------------
import os
  
# Specify the path
path = '/Users/walidkebsi/Downloads/pyready_trader_go 3'
  
# Specify the file name
file = 'app2.py'
  
# Before creating
dir_list = os.listdir(path) 
print("List of directories and files before creation:")
print(dir_list)
print()
  
# Creating a file at specified location
with open(os.path.join(path, file), 'w') as fp:
    fp.write("app2.py")
    # To write data to new file uncomment
    # this fp.write("New file created")
  
# After creating 
dir_list = os.listdir(path)
print("List of directories and files after creation:")
print(dir_list)

#---------------------------CREATION DU FICHIER VIDE PY------------------------------------

#---------------------------MAIN ALGO - DASHBORD --------------------------------------------
oil_data = pd.read_csv("oil.csv", index_col='Date', parse_dates=True)

#-----------Modèle de prédiction régression linéaire-------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#------Création d'une classe Log1p Transformer pour ajouter une méthode get_param
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Log1pTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)
    
    def inverse_transform(self, X):
        return np.expm1(X)


# Valeur Nan normal because the rolling begins after 30 days. 
# Method :use the ewm
oil_data= oil_data.rename(columns={'Close/Last' : 'Close'})
oil_data['volatility'] = oil_data['Close'].ewm(alpha=0.6).std()

# Data cleaning 
oil_data.replace('NaN', None, inplace=True)
oil_data = oil_data.dropna(axis=0, how='any')
mean_volatility = oil_data['volatility'].mean()
oil_data['volatility'].fillna(mean_volatility, inplace=True)

oil_data = oil_data[oil_data['Close'] >= 0]


# Check data
print('Valeurs NaN : \n', oil_data.isnull().sum())

#Data training 

y = oil_data['Close']
X = oil_data.drop(['Close', 'Volume', 'High', 'Low'], axis=1)

#prédiction en fontion de la volatilité observé et du price d'ouverture de la session


# Encode categorical variables - no needed - only numeric data


# Normalization 

X_MinMax = MinMaxScaler().fit_transform(X) # NO outlier.

print('Cest X', X)
print('Cest X normalisé', X_MinMax)

# if outlier : should use RobustScaler

# Define pipelines for data preprocessing

preprocessor_linear = Pipeline([('scaler', MinMaxScaler())])

#preprocessor_tree = Pipeline([
    #('scaler', StandardScaler()),
    #('onehot', pd.get_dummies(X, columns=['Sector']))
#]). We don't need any non-linear model here.

# Define models to train
linear_model = LinearRegression()

lasso_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', Lasso())
])

ridge_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', Ridge())
])

svr_model = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('model', SVR())
])

# Define transformed target regressor for non-negative predictions
tt_reg = TransformedTargetRegressor(regressor=ridge_model, transformer=Log1pTransformer())

models = [('Linear Regression', linear_model),
          ('Lasso Regression', lasso_model),
          ('Ridge Regression', ridge_model),
          ('SVR', svr_model),
          ('Transformed Target Ridge', tt_reg)]

for name, model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: mean score = {np.mean(scores):.3f}, std deviation = {np.std(scores):.3f}')
    

model = LinearRegression()



X_train, X_test, y_train, y_test = train_test_split(X_MinMax,y, test_size=0.2, shuffle=(1))

model.fit(X_train, y_train)

print('Le score du model est de :', model.score(X_test, y_test))

predictions = model.predict(X_test) 

real_predictions = np.exp(predictions) - 1



#-----------Modèle de prédiction régression linéaire-------------------------------

#---------------------------MAIN ALGO - DASHBORD --------------------------------------------

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Oil Prediction Price",),
        html.P(
            children="Interactive Dashboard for oil price prediction",
        ),
        
        html.Div([
            dcc.Graph( id = 'graph_price',
                      
                      figure={ "data":[go.Scatter(x=oil_data.index, y= oil_data['Close'], mode='lines')],
                              'layout' : {'title' : "Oil Price Evolution"}})
        ]),

        html.Div([
            dcc.Graph(id='graph_volatility',
                      figure ={'data': [go.Scatter(x=oil_data.index, y = oil_data['volatility'], mode="lines")],
                              "layout" : {"title":"Volatility Evolution "}})
        ]),

        html.Div([
            html.H2('Price Prediction'),
            
            html.Div([
                html.Label('volatility : '),
                dcc.Input(id='input_volatility', type='number', value=0)
            ]),
            
            html.Div([
                html.Label('Open : '),
                dcc.Input(id='input_Open', type='number', value=0)
            ]),
           
            
            html.Button('prediction', id='button_prediction'),
            
            html.Div(id='output_prediction')
        ])
    ]
)

def prediction_price(n_clicks, volatility, Open):
    
    X_values = [[volatility, Open]]
    Y_new = model.predict(X_values)
    return html.Div("Forecasted price is {} USD".format(round(Y_new[0], 2)))

@app.callback(
    Output('output_prediction', 'children'),
    [Input('button_prediction', 'n_clicks')],
    [State('input_volatility', 'value'),
     State('input_Open', 'value')]
     
     
    
)

def update_output(n_clicks, volatility, Open):
    if n_clicks is None:
        return
    else:
        prediction_result = prediction_price(n_clicks, volatility,Open)
        return prediction_result
    
    
    #---------------------------MAIN ALGO - DASHBORD --------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)