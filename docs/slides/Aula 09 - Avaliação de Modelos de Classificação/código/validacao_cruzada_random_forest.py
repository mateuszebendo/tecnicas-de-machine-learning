# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:31:02 2024

@author: marco
"""

########################


# Classificador
from sklearn.ensemble import RandomForestClassifier

# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state = 1)
acuracias = []
matrizes = []
metricas = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0],1))):
    #Melhor configuração do modelo
    classificador = RandomForestClassifier(criterion='gini',
                                           max_depth=3,
                                           n_estimators=5,
                                           max_features=3,
                                           random_state=0)

    #  Treinamento
    classificador.fit(previsores.iloc[indice_treinamento], classe.iloc[indice_treinamento,0])
    
    # teste
    previsoes = classificador.predict(previsores.iloc[indice_teste])
    
    acuracia = accuracy_score(classe.iloc[indice_teste,0], previsoes)
    
    metricas.append(precision_recall_fscore_support(classe.iloc[indice_teste,0], previsoes))
    matrizes.append(confusion_matrix(classe.iloc[indice_teste,0], previsoes))
    acuracias.append(acuracia)
    
##################3 Resultado Final ####################
#Matriz de confusão média
matriz_media = np.mean(matrizes, axis = 0)
matriz_desvio_padrao = np.std(matrizes, axis = 0)
#Métricas médias
acuracias = np.asarray(acuracias)
acuracia_final_media = acuracias.mean()
acuracia_final_desvio_padrao =acuracias.std()
metricas_medias = np.mean(metricas, axis = 0)
metricas_desvio_padrao = np.std(metricas, axis=0)
# obs: 
# cada linha: precisão, recall, f1score 
# cada coluna: classe
    
    
    
    
    
    
    