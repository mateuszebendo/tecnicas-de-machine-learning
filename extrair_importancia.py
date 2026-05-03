import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scripts.pre_processamento import preparar_dados

# 1. Preparar dados
X, y = preparar_dados('dados/mushrooms.csv')

# 2. Treinar a melhor Árvore (max_depth=8 como no seu config)
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
modelo.fit(X, y)

# 3. Mapear importâncias do One-Hot Encoding de volta para os atributos originais
importancias = modelo.feature_importances_
features_nomes = X.columns

# Agrupar as importâncias por prefixo (ex: odor_f, odor_n -> odor)
importancia_original = {}
for i, nome in enumerate(features_nomes):
    # O prefixo é a parte antes do primeiro underscore '_' gerado pelo get_dummies
    if '_' in nome:
        original = nome.split('_')[0]
    else:
        original = nome
    
    importancia_original[original] = importancia_original.get(original, 0) + importancias[i]

# Converter para DataFrame e formatar
df_imp = pd.DataFrame(list(importancia_original.items()), columns=['Atributo', 'Importancia'])
df_imp['Importancia (%)'] = (df_imp['Importancia'] * 100).round(2)
df_imp = df_imp.sort_values(by='Importancia', ascending=False).head(5)

print(df_imp.to_string(index=False))
