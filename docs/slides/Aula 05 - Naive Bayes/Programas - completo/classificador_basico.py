# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:03:40 2023

@author: marco
"""

import pandas as pd

################## Classificação por classe majoritária ################## 

# Resultado mínimo - ajuste pela classe majoritária
contagem = classe_treinamento['Risk'].value_counts()
classe_majoritaria = contagem.idxmax()

previsoes = pd.Series([classe_majoritaria]).repeat(classe_teste.size)

from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)