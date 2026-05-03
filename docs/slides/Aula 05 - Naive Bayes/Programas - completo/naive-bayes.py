# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:02:00 2019

@author: marco
"""

################## Classificação com Naive Bayes ################## 

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

# Treinamento (automaticamente o sklearn já faz a correção laplaciana)
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste 
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e MATRIZ DE CONFUSÃO)
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)


# Resultados na base de treinamento, para verificar overfitting
# previsoes_treinamento = classificador.predict(previsores_treinamento)
# acuracia_treinamento = accuracy_score(classe_treinamento, previsoes_treinamento)
# matriz_treinamento = confusion_matrix(classe_treinamento, previsoes_treinamento)



