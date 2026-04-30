import pandas as pd
base = pd.read_csv('mushrooms.csv')
describe = base.describe()

classe = base['class']
previsores = base.drop('class', axis=1)

## ----------- Separando variável classe -----------
from sklearn.preprocessing import LabelEncoder

labelencoder_class = LabelEncoder()

# Codificação da variável alvo 'class'
classe = labelencoder_class.fit_transform(classe)
classe = classe.astype('int64')

## --- Transformar as variáveis categóricas (nominais) em valores numéricos ---

from sklearn.preprocessing import LabelBinarizer

cols_previsores = [
  "cap-shape", 
  "cap-surface", 
  "cap-color", 
  "bruises", 
  "odor", 
  "gill-attachment", 
  "gill-spacing", 
  "gill-size", 
  "gill-color", 
  "stalk-shape", 
  "stalk-root", 
  "stalk-surface-above-ring", 
  "stalk-surface-below-ring", 
  "stalk-color-above-ring", 
  "stalk-color-below-ring", 
  "veil-type", 
  "veil-color", 
  "ring-number", 
  "ring-type", 
  "spore-print-color", 
  "population", 
  "habitat"
]


def aplicar_variaveis_dummy(df, colunas):
  labelbinarizer = LabelBinarizer()
  
  for coluna in colunas:
    # Pega a coluna e transforma em matriz
    variaveis_dummy = labelbinarizer.fit_transform(df[coluna])
    
    # O SEGREDO ESTÁ AQUI: Tratar colunas binárias (apenas 2 opções)
    if len(labelbinarizer.classes_) == 2:
      # Se tem 2 categorias, o Binarizer gera apenas 1 coluna.
      # O '1' sempre representa a segunda classe da lista (em ordem alfabética).
      classe_representada = labelbinarizer.classes_[1] 
      novas_variaveis_dummy = [f"{coluna}_{classe_representada}"]
    else:
      # Comportamento normal para 3 ou mais categorias
      novas_variaveis_dummy = [f"{coluna}_{classe}" for classe in labelbinarizer.classes_]
  
    # Cria o DataFrame (adicionado index=df.index para segurança)
    df_variaveis_dummy = pd.DataFrame(variaveis_dummy, 
                                      columns=novas_variaveis_dummy, 
                                      index=df.index)
    
    # Junta no DataFrame original e apaga a coluna antiga
    df = df.join(df_variaveis_dummy)
    df = df.drop(coluna, axis=1)
      
  return df

previsores = aplicar_variaveis_dummy(previsores, cols_previsores)

## --- Balanceamento com Undersampling ---

class_count =  base['class'].value_counts()
print('Class poisonous:', class_count['p'])
print('Class edible:', class_count['e'])
print('Proportion:', round(class_count['p'] / class_count['e'], 2), ': 1')
class_count.plot(kind='bar', title='Count (target)',color = ['#1F77B4', '#FF7F0E']);

#  Class poisonous: 3916
#  Class edible: 4208
#  Proportion: 0.93 : 1

##! Breakpoint
print(previsores)