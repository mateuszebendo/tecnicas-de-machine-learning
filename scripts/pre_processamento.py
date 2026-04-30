import pandas as pd

def preparar_dados(caminho_arquivo='../dados/mushrooms.csv'):
    """
    Carrega o dataset, trata valores faltantes e aplica codificação categórica.
    """
    # 1. Carregamento dos dados
    df = pd.read_csv(caminho_arquivo)
    
    # 2. Tratamento do dado faltante '?' no atributo stalk-root
    df['stalk-root'] = df['stalk-root'].replace('?', 'desconhecido')
    
    # 3. Separação entre Variáveis Preditoras (X) e Variável Alvo (y)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 4. Codificação da Variável Alvo (y)
    # A literatura estatística recomenda transformar a classe de interesse no valor 1.
    # Como o perigo é o cogumelo venenoso, mapeamos 'p' (poisonous) para 1 e 'e' (edible) para 0.
    y = y.map({'e': 0, 'p': 1})
    
    # 5. Codificação das Variáveis Preditoras (X) - One-Hot Encoding
    # Utilizamos o get_dummies do Pandas. O parâmetro drop_first=True remove a primeira
    # categoria de cada coluna para evitar a armadilha das variáveis dummy (multicolinearidade),
    # uma prática fundamental descrita em "The Elements of Statistical Learning" (Hastie et al.).
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded, y

# Bloco de teste
if __name__ == "__main__":
    X, y = preparar_dados()
    print("Dados preparados com sucesso!")
    print(f"Dimensão de X após One-Hot Encoding: {X.shape}")
    print(f"Dimensão de y: {y.shape}")