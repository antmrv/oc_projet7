# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:34:11 2023

@author: F7936
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# Explicabilité
import lime
from lime import lime_tabular
# API
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json


# Créer l'objet app
app = FastAPI()

# Modèle gradient boosting classifier
pickle_in = open("classifier.pkl", "rb")
model = pickle.load(pickle_in)

# Données utilisées pour entraîner le modèle
pickle_in = open("train_set.pkl", "rb")
data_train = pickle.load(pickle_in)

# Données de test
pickle_in = open("test_set.pkl", "rb")
data_test = pickle.load(pickle_in)

# http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message' : "Bienvenue sur l'outil de scoring de crédit"}

class Client(BaseModel):
    client_id: int  

        
### id pour test : 379761 ###

# Informations client
@app.post('/show')
def client_informations(idt_test:Client):
    """ 
    Indique les informations d'un client (test set)
    """
    idt = idt_test.client_id
    client_test = data_test[data_test['SK_ID_CURR'] == idt]\
    .drop(['SK_ID_CURR', 'TARGET'], axis = 1).to_json(orient = 'records')

    return client_test

# Prédictions
@app.post('/predict_proba')
def default_prediction(idt_test:Client):
    idt = idt_test.client_id
    client_test = data_test[data_test['SK_ID_CURR'] == idt]\
    .drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    
    # Prediction pour un client donné
    prediction = model.predict_proba(client_test)[0][1]
    return {'predict_proba' : prediction}

explainer = lime_tabular.LimeTabularExplainer(
    training_data = np.array(data_train.drop(['SK_ID_CURR', 'TARGET'], axis = 1)),
    feature_names = data_train.drop(['SK_ID_CURR', 'TARGET'], axis =1).columns,
    class_names = [0, 1],
    mode = 'classification')

@app.post('/explain')
def explain_feature(idt_test:Client):
    """
    Renvoi la prediction par LIME ainsi que l'explication de la decision pour l'individu teste
    """
    idt = idt_test.client_id
    client_test = data_test[data_test['SK_ID_CURR'] == idt].drop(['SK_ID_CURR', 'TARGET' ], axis = 1).values
    exp = explainer.explain_instance(
        data_row = client_test[0], 
        predict_fn = model.predict_proba)
    return json.dumps(exp.as_list())


# Run l'api avec uvicorn http:://127.0.0.1:8000, utiliser la commande : uvicorn main:app --reload
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)