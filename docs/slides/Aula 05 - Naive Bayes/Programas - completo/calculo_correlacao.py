# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:14:55 2023

@author: marco
"""


import pandas as pd
import numpy as np


###################### Descobrindo maiores correlações com objetivo ##############################
#  Executar antes da padronização


# Considerando todas as características
base_caracteristicas_e_objetivo = pd.concat([previsores, classe], axis=1)

corr = base_caracteristicas_e_objetivo.corr()

#  plotar matriz de correlação de pearson
import seaborn as sns
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, 
    xticklabels=True, 
    yticklabels=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

####################### Removendo colunas correlacionadas #####################

base = pd.read_csv('credit_data.csv')
nova_base = base.drop('Duration', 1)


# =============================================================================
#                       SALVANDO BASE EM ARQUIVO
# =============================================================================

pd.DataFrame.to_csv(nova_base, 'credit_data_reduzida.csv', index=False)

############################################################################### 


