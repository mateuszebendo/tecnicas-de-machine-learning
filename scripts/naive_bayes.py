from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    GaussianNB, 
    BernoulliNB, 
    MultinomialNB, 
    ComplementNB, 
    CategoricalNB
)
from sklearn.metrics import accuracy_score, confusion_matrix
from pre_processamento import preparar_dados
from visualizacao import plotar_matriz_confusao, plotar_curva_roc, plotar_curva_pr, relatorio_metricas

def executar_naive_bayes(tipo='gaussian'):
    # 1. Importação dos dados do nosso módulo central
    X, y = preparar_dados()
    
    # 2. Divisão de Treino e Teste (mesmos parâmetros para comparação justa)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Instanciação e Treinamento do Modelo
    if tipo == 'gaussian':
        modelo = GaussianNB()
    elif tipo == 'bernoulli':
        modelo = BernoulliNB()
    elif tipo == 'multinomial':
        modelo = MultinomialNB()
    elif tipo == 'complement':
        modelo = ComplementNB()
    elif tipo == 'categorical':
        modelo = CategoricalNB()
    else:
        raise ValueError(f"Tipo de modelo desconhecido: '{tipo}'")
        
    modelo.fit(X_train, y_train)
    
    # 4. Previsões
    y_pred = modelo.predict(X_test)
    # Extrai as probabilidades da classe 1 (Venenoso) para as curvas
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # 5. Extração de Métricas
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    nome_modelo = f"Naive Bayes {tipo.capitalize()}"
    print(f"--- Resultados {nome_modelo} ---")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    
    relatorio_metricas(y_test, y_pred, nome_modelo)
    plotar_matriz_confusao(cm, nome_modelo)
    plotar_curva_roc(y_test, y_proba, nome_modelo)
    plotar_curva_pr(y_test, y_proba, nome_modelo)

if __name__ == "__main__":
    tipos_nb = ['gaussian', 'bernoulli', 'multinomial', 'complement', 'categorical']
    
    for t in tipos_nb:
        executar_naive_bayes(tipo=t)
        print("\n" + "="*50 + "\n")