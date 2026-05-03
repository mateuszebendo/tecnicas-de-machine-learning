# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:46:44 2024

@author: marco
"""

####### Configuração da árvore de decisão ##################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

acuracias_treinamento = []
acuracias_teste = []


# testar varios valores de K
h = range(1,25)
for i in h:
    #construir o modelo
    classificador = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=i,
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

############## Classificação com árvores de decisão ##########

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# gereção da árvore
classificador = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=16,
                                       random_state=0)

#  Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# teste
previsoes = classificador.predict(previsores_teste)

#análise de resultados
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste,previsoes)
matriz_teste = confusion_matrix(classe_teste,previsoes)



#  Exportando a arvore
from sklearn.tree import export_graphviz
export_graphviz(classificador, 
                out_file="tree.dot", 
                class_names=["bad", "good"],
                feature_names=cols_previsores, 
                impurity=False, 
                filled=True)


#  Visualizando a arvore
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# Visualizando a importância das caracteristicas
import matplotlib.pyplot as plt
import numpy as np
n_features = previsores.columns.size
plt.barh(range(n_features), classificador.feature_importances_, align='center')
plt.yticks(np.arange(n_features), previsores.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")






























