# imports
import sys

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

import pickle
# Funções utilizadas

def relacao_nome_bairro(x):
    mask = teste['nome'] == x
    if teste.loc[mask, 'bairro'].values[0] in teste.loc[mask, 'nome'].values[0]:
        return 'yes'
    elif teste.loc[mask, 'bairro_group'].values[0] in teste.loc[mask, 'nome'].values[0]:
        return 'yes'
    return 'no'
    
def dummies(X):
    X = X.copy()
        
    objectos = X.select_dtypes(include='object', exclude=None)

    not_objectos = X.select_dtypes(include=None, exclude=object).reset_index()
    try:
        objectos = pd.get_dummies(data = objectos, drop_first = True).reset_index()
        
    except:
        pass
    X = pd.merge(objectos,not_objectos, on = 'id')
    X = X.set_index('id')
    return X
    
def process(df):
    df = df.copy()
    def clear_nome(x):
        wordo = [word.strip("!,.-/()\" '").lower() for word in x.split(' ')]
        wordo = ' '.join(wordo)
        return wordo
    
    df['nome'] = df['nome'].apply(clear_nome)
    df['bairro'] = df['bairro'].apply(clear_nome)
    df['bairro_group'] = df['bairro_group'].apply(clear_nome)
    
    return df

# Carregar arquivo para predição
if __name__ == "__main__":
	if len(sys.argv) > 1:
		arq = sys.argv[1]
    	
teste = pd.read_csv(arq)
teste = teste.set_index('id')

# Carregar features_without_price
with open('features_without_price.txt', 'r') as arq:
	features_without_price = arq.read()
	features_without_price = features_without_price.split(',')
features_without_price


teste = process(teste)
teste['rel_nome_bairro'] = teste['nome'].apply(relacao_nome_bairro)
teste = teste[features_without_price]
teste = dummies(teste)

# Carregar features
features_regr = pd.read_csv('features.csv')
features_regr = list(features_regr['0'].values)
features_regr

# Processar teste
features_regr_df = pd.DataFrame(columns=features_regr)


for column in features_regr_df:
    if column in teste.columns:
        features_regr_df[column] = teste[column].values
    else:
        features_regr_df[column] = False
		
for column in features_regr_df:
    if column in teste.columns:
        features_regr_df[column] = teste[column].values[0]
    else:
        features_regr_df[column] = False

# Carregar modelo
regr = pickle.load(open('linear_regression_model.pkl', 'rb'))

print('O valor previsto para o imóvel sugerido é $',round(regr.predict(features_regr_df)[0], 0))
