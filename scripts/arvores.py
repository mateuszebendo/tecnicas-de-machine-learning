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
    # Configurações extraídas do seu Arquivo de Investigação
    configs = [
        {'tipo': 'decision_tree', 'params': {'criterion': 'entropy', 'max_depth': 8, 'random_state': 0}},
        {'tipo': 'decision_tree', 'params': {'criterion': 'entropy', 'max_depth': 6, 'random_state': 0}},
        {'tipo': 'random_forest', 'params': {'criterion': 'gini', 'max_depth': 15, 'n_estimators': 9, 'max_features': 7, 'random_state': 0}}
    ]
    
    for cfg in configs:
        executar_arvore(cfg['params'], tipo=cfg['tipo'])
        print("\n" + "="*50 + "\n")