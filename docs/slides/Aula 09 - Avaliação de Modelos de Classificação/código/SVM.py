# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:20:03 2019

@author: marco
"""

import pandas as pd

################## Classificação com SVM ################## 

# Treinamento 
from sklearn.svm import SVC
classificador = SVC(kernel = 'sigmoid', C = 0.9, gamma = 'auto', random_state = 1)
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste 
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)

# Resultados na base de treinamento, para verificar overfitting
previsoes_treinamento = classificador.predict(previsores_treinamento)
acuracia_treinamento = accuracy_score(classe_treinamento, previsoes_treinamento)
matriz_treinamento = confusion_matrix(classe_treinamento, previsoes_treinamento)


