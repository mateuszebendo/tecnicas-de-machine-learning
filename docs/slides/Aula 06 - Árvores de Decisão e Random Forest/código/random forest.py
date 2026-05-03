# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:46:44 2024

@author: marco
"""

####### Configuração da árvore de decisão ##################
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

acuracias_treinamento = []
acuracias_teste = []


# testar varios valores de K
h = range(1,30)
for i in h:
    #construir o modelo
    classificador = RandomForestClassifier(criterion='gini',
                                           max_depth=i,
                                           n_estimators=9,
                                           max_features=7,
                                           random_state=0)
    # treinar modelo
    classificador.fit(previsores_treinamento, classe_treinamento)
    acuracias_treinamento.append(classificador.score(previsores_treinamento,classe_treinamento))
    acuracias_teste.append(classificador.score(previsores_teste,classe_teste))

plt.plot(h, acuracias_treinamento, label="acuracia de treinamento")
plt.plot(h, acuracias_teste, label="acuracia de teste")
plt.ylabel("Acuracia")
plt.xlabel("Valor de altura")
plt.legend()

############## Classificação com random forest ##########

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# gereção da árvore
classificador = RandomForestClassifier(criterion='gini',
                                       max_depth=15,
                                       n_estimators=9,
                                       max_features=7,
                                       random_state=0)

#  Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# teste
previsoes = classificador.predict(previsores_teste)

#análise de resultados
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste,previsoes)
matriz_teste = confusion_matrix(classe_teste,previsoes)



# Visualizando a importância das caracteristicas
import matplotlib.pyplot as plt
import numpy as np
n_features = previsores.columns.size
plt.barh(range(n_features), classificador.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")






























