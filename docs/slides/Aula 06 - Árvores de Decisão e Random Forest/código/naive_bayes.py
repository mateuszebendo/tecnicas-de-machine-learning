# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:34:44 2024

@author: marco
"""

############## Classificação com Naive Bayes ##########

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

#  Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# teste
previsoes = classificador.predict(previsores_teste)

#análise de resultados
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste,previsoes)
matriz_teste = confusion_matrix(classe_teste,previsoes)