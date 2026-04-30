from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pre_processamento import preparar_dados
from visualizacao import plotar_matriz_confusao, plotar_curva_roc, plotar_curva_pr, relatorio_metricas

def executar_rede_neural(params, nome_modelo="Rede Neural (MLP)"):
    # 1. Preparação dos dados
    X, y = preparar_dados()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Padronização Obrigatória (Z-score) - Crucial para Redes Neurais
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Modelo MLP configurado via dicionário
    # Mantemos max_iter alto e random_state fixo para reprodutibilidade
    modelo = MLPClassifier(**params, max_iter=2000, random_state=1)
    
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
    
    # 5 CONFIGURAÇÕES PARA REDES NEURAIS (MLP)
    configs = [
        # 1. A sua original: Uma camada escondida com 10 neurônios, usando SGD (Stochastic Gradient Descent)
        {'nome': 'MLP (Original SGD)', 'params': {'hidden_layer_sizes': (10,), 'activation': 'relu', 'solver': 'sgd', 'tol': 1e-6}},
        
        # 2. Rede "Profunda": Duas camadas escondidas (50 neurônios na 1ª, 25 na 2ª) usando o otimizador ADAM (padrão moderno)
        {'nome': 'MLP (Profunda 50x25 Adam)', 'params': {'hidden_layer_sizes': (50, 25), 'activation': 'relu', 'solver': 'adam'}},
        
        # 3. Função de Ativação Tanh: Muda a matemática dos neurônios (de ReLU para Tangente Hiperbólica). Útil em alguns cenários não-lineares.
        {'nome': 'MLP (1 Camada Tanh)', 'params': {'hidden_layer_sizes': (20,), 'activation': 'tanh', 'solver': 'adam'}},
        
        # 4. Solver L-BFGS: Otimizador extremamente rápido e potente, altamente recomendado para datasets pequenos (com poucos milhares de linhas).
        {'nome': 'MLP (L-BFGS Rápido)', 'params': {'hidden_layer_sizes': (10, 5), 'activation': 'relu', 'solver': 'lbfgs'}},
        
        # 5. Alta Regularização (Anti-Overfitting): Uma rede larga (100 neurônios), mas com o parâmetro 'alpha' alto para punir a rede se ela tentar decorar os dados.
        {'nome': 'MLP (Alta Regularização)', 'params': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.05}}
    ]
    
    for cfg in configs:
        print(f"\nExecutando {cfg['nome']} com os parâmetros: {cfg['params']}")
        executar_rede_neural(params=cfg['params'], nome_modelo=cfg['nome'])
        print("\n" + "="*50)