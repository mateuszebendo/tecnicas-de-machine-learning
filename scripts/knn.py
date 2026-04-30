from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pre_processamento import preparar_dados
from visualizacao import plotar_matriz_confusao, plotar_curva_roc, plotar_curva_pr, relatorio_metricas

def executar_knn(k=5):
    # 1. Importação dos dados modulares
    X, y = preparar_dados()
    
    # 2. Divisão de Treino e Teste (80% / 20%)
    # O uso do 'stratify=y' garante que o desbalanceamento não afete a amostragem
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Padronização (Z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Instanciação e Treinamento do Modelo
    modelo = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    modelo.fit(X_train_scaled, y_train)
    
    # 5. Previsões
    y_pred = modelo.predict(X_test_scaled)
    # Extrai as probabilidades da classe 1 (Venenoso) para as curvas
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    # 6. Extração de Métricas para o Arquivo de Investigação
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    nome_modelo = f"KNN (K={k})"
    print(f"--- Resultados {nome_modelo} ---")
    print(f"Acurácia: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    
    relatorio_metricas(y_test, y_pred, nome_modelo)
    plotar_matriz_confusao(cm, nome_modelo)
    plotar_curva_roc(y_test, y_proba, nome_modelo)
    plotar_curva_pr(y_test, y_proba, nome_modelo)

if __name__ == "__main__":
    valores_k = [23]
    
    for k in valores_k:
        executar_knn(k=k)
        print("\n" + "="*50 + "\n")