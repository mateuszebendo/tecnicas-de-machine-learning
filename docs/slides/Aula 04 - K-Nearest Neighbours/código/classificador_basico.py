# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:47:14 2024

@author: marco
"""

############ Classificação por classe majoritaria ############

# resultado minimo
contagem = classe_treinamento['Risk'].value_counts()
classe_majoritaria = contagem.idxmax()

previsoes = pd.Series([classe_majoritaria]).repeat(classe_teste.size)

#análise de resultados
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste,previsoes)
matriz_teste = confusion_matrix(classe_teste,previsoes)