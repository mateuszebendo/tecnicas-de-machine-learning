from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pre_processamento import preparar_dados
from visualizacao import plotar_matriz_confusao, plotar_curva_roc, plotar_curva_pr, relatorio_metricas, plotar_importancia_variaveis

def executar_arvore(config, tipo='decision_tree'):
    # 1. Dados e Divisão
    X, y = preparar_dados()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Instanciação do Modelo usando desempacotamento de dicionário (**config)
    if tipo == 'decision_tree':
        modelo = DecisionTreeClassifier(**config)
        nome_modelo = f"Árvore (depth={config.get('max_depth')})"
    elif tipo == 'random_forest':
        modelo = RandomForestClassifier(**config)
        nome_modelo = "Random Forest"
        
    # 3. Treinamento
    modelo.fit(X_train, y_train)
    
    # 4. Previsões e Probabilidades
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # 5. Métricas e Prints para o Arquivo de Investigação
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- Resultados {nome_modelo} ---")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    
    # 6. Geração de Gráficos e Importância
    relatorio_metricas(y_test, y_pred, nome_modelo)
    plotar_matriz_confusao(cm, nome_modelo)
    plotar_curva_roc(y_test, y_proba, nome_modelo)
    plotar_curva_pr(y_test, y_proba, nome_modelo)
    
    # Extração das variáveis mais importantes (Feature Importance)
    # X.columns contém os nomes das características geradas pelo One-Hot Encoding
    plotar_importancia_variaveis(modelo.feature_importances_, X.columns, nome_modelo)

if __name__ == "__main__":
    # Configurações para exploração de hiperparâmetros
    configs = [
        # ==========================================
        # 5 CONFIGURAÇÕES PARA DECISION TREE
        # ==========================================
        
        # 1. Árvore rasa (boa para evitar overfitting)
        {'tipo': 'decision_tree', 'params': {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}},
        
        # 2. Árvore baseada em Entropia com profundidade moderada
        {'tipo': 'decision_tree', 'params': {'criterion': 'entropy', 'max_depth': 8, 'random_state': 0}},
        
        # 3. Árvore profunda, mas exigindo mínimo de amostras nas folhas (traz estabilidade)
        {'tipo': 'decision_tree', 'params': {'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 5, 'random_state': 0}},
        
        # 4. Árvore sem limite de profundidade (vai crescer até classificar tudo ou atingir o limite lógico)
        {'tipo': 'decision_tree', 'params': {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 10, 'random_state': 0}},
        
        # 5. Usando log_loss (similar à entropia) com poda leve
        {'tipo': 'decision_tree', 'params': {'criterion': 'log_loss', 'max_depth': 6, 'min_samples_leaf': 2, 'random_state': 0}},

        # ==========================================
        # 5 CONFIGURAÇÕES PARA RANDOM FOREST
        # ==========================================
        
        # 1. Floresta pequena e rápida (poucas árvores, profundidade limitada)
        {'tipo': 'random_forest', 'params': {'n_estimators': 10, 'criterion': 'gini', 'max_depth': 5, 'random_state': 0}},
        
        # 2. Floresta padrão robusta (100 árvores é o padrão moderno do scikit-learn)
        {'tipo': 'random_forest', 'params': {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 10, 'random_state': 0}},
        
        # 3. Focada em Entropia com limite restrito de features por divisão (sqrt)
        {'tipo': 'random_forest', 'params': {'n_estimators': 50, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 'sqrt', 'random_state': 0}},
        
        # 4. Floresta grande (200 árvores) para máxima estabilidade das previsões
        {'tipo': 'random_forest', 'params': {'n_estimators': 200, 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 2, 'random_state': 0}},
        
        # 5. Floresta sem Bootstrap (pasting). Usa o dataset inteiro em todas as árvores em vez de amostras
        {'tipo': 'random_forest', 'params': {'n_estimators': 50, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 12, 'random_state': 0}}
    ]
    
    for cfg in configs:
        print(f"Executando {cfg['tipo'].upper()} com os parâmetros: {cfg['params']}")
        executar_arvore(cfg['params'], tipo=cfg['tipo'])
        print("\n" + "="*50 + "\n")