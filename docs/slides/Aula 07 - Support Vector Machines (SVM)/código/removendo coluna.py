# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:05:04 2025

@author: marco
"""

###################### Removendo colunas correlacionadas ######################

base = pd.read_csv('credit_data.csv')
nova_base = base.drop(columns=['Duration'])

# ============================================
#              SALVANDO BASE EM ARQUIVO
# ============================================

nova_base.to_csv('credit_data_reduzida.csv', index=False)
