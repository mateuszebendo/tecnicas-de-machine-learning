from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from pre_processamento import preparar_dados
from visualizacao import plotar_matriz_confusao, plotar_curva_roc, plotar_curva_pr, relatorio_metricas

def executar_svm(params, nome_modelo="SVM"):
    # 1. Preparação dos dados
    X, y = preparar_dados()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Padronização Obrigatória (Z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Modelo configurado via dicionário de parâmetros
    # Desempacotamos o dicionário usando **params
    # Mantemos probability=True e random_state hardcoded para garantir as curvas e reprodutibilidade
    modelo = SVC(**params, random_state=1, probability=True)
    
    # 4. Treinamento
    modelo.fit(X_train_scaled, y_train)
    
    # 5. Previsões
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    # 6. Avaliação
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- Resultados {nome_modelo} ---")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    
    relatorio_metricas(y_test, y_pred, nome_modelo)
    plotar_matriz_confusao(cm, nome_modelo)
    plotar_curva_roc(y_test, y_proba, nome_modelo)
    plotar_curva_pr(y_test, y_proba, nome_modelo)

if __name__ == "__main__":
    
    # 5 CONFIGURAÇÕES PARA SVM
    configs = [
        # 1. Linear: Cria uma linha de separação reta.
        {'nome': 'SVM (Linear)', 'params': {'kernel': 'linear', 'C': 1.0}},
        
        # 2. RBF Padrão (Radial Basis Function).
        {'nome': 'SVM (RBF Padrão)', 'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}},
        
        # 3. RBF Complexo (Alto C): Aumentar o C penaliza fortemente os erros no treino. Ele tenta criar uma fronteira perfeita (risco maior de overfitting).
        {'nome': 'SVM (RBF C=10)', 'params': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}},
        
        # 4. Polinomial: Cria fronteiras de decisão em formato de polinômio.
        {'nome': 'SVM (Polinomial Grau 3)', 'params': {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'gamma': 'scale'}},
        
        # 5. Sigmoid: Tenta simular uma rede neural de uma camada matemática.
        {'nome': 'SVM (Sigmoid)', 'params': {'kernel': 'sigmoid', 'C': 0.9, 'gamma': 'auto'}}
    ]
    
    for cfg in configs:
        print(f"\nExecutando {cfg['nome']} com os parâmetros: {cfg['params']}")
        executar_svm(params=cfg['params'], nome_modelo=cfg['nome'])
        print("\n" + "="*50)