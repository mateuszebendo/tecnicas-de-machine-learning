from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from pre_processamento import preparar_dados

# Importando os modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def gerar_linha_e_matrizes_excel(modelo, nome_modelo, padronizar=False):
    # 1. Carrega os dados
    X, y = preparar_dados()
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # 2. Configura a Validação Cruzada (5 folds)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Dicionário para guardar as métricas e matrizes
    metricas = {
        'acc': [], 'prec_bad': [], 'prec_good': [],
        'rec_bad': [], 'rec_good': [], 'f1_bad': [], 'f1_good': [],
        'matrizes': [] # matrizes de confusão
    }
    
    print(f"\nCalculando {nome_modelo}...")
    
    for train_index, test_index in kf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        
        # Padronização isolada por fold
        if padronizar:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # --- CÁLCULO DAS MÉTRICAS PRINCIPAIS ---
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[1, 0], zero_division=0)
        
        # --- CÁLCULO DA MATRIZ DE CONFUSÃO DO FOLD ---
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        metricas['acc'].append(acc)
        metricas['prec_bad'].append(prec[0])
        metricas['prec_good'].append(prec[1])
        metricas['rec_bad'].append(rec[0])
        metricas['rec_good'].append(rec[1])
        metricas['f1_bad'].append(f1[0])
        metricas['f1_good'].append(f1[1])
        metricas['matrizes'].append(cm)

    # --- PROCESSANDO RESULTADOS PARA O EXCEL ---
    
    # 1. Métricas da Linha Principal
    lista_metricas = ['acc', 'prec_bad', 'prec_good', 'rec_bad', 'rec_good', 'f1_bad', 'f1_good']
    medias = [np.mean(metricas[m]) for m in lista_metricas]
    desvios = [np.std(metricas[m]) for m in lista_metricas]
    
    linha_medias = "\t".join([f"{v:.4f}".replace('.', ',') for v in medias])
    linha_desvios = "\t".join([f"{v:.4f}".replace('.', ',') for v in desvios])
    linha_excel = f"{nome_modelo}\t{linha_medias}\t\t\t{linha_desvios}"
    
    # 2. Processando as Matrizes
    # Empilha as 5 matrizes (formando um bloco 3D) e tira a média/desvio no eixo 0
    matrizes_array = np.array(metricas['matrizes'])
    cm_media = np.mean(matrizes_array, axis=0)
    cm_desvio = np.std(matrizes_array, axis=0)
    
    # --- IMPRESSÃO FORMATADA ---
    print("\n" + "="*80)
    print(linha_excel)
    
    print(f"-- Matriz Média: {nome_modelo} --")
    print(f"{cm_media[0,0]:.1f}\t{cm_media[0,1]:.1f}".replace('.', ','))
    print(f"{cm_media[1,0]:.1f}\t{cm_media[1,1]:.1f}".replace('.', ','))
    
    print(f"\n-- Matriz Desvio: {nome_modelo} --")
    print(f"{cm_desvio[0,0]:.5f}\t{cm_desvio[0,1]:.5f}".replace('.', ','))
    print(f"{cm_desvio[1,0]:.5f}\t{cm_desvio[1,1]:.5f}".replace('.', ','))
    print("="*80)

if __name__ == "__main__":
    
    # 1. KNN (K=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    gerar_linha_e_matrizes_excel(knn, "KNN", padronizar=True)

    # 2. Naive Bayes (Gaussian)
    nb = GaussianNB()
    gerar_linha_e_matrizes_excel(nb, "Naive Bayes", padronizar=False)
    
    # 3. Árvore de Decisão (depth=8)
    arvore = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0)
    gerar_linha_e_matrizes_excel(arvore, "Árvores de decisão", padronizar=False)
    
    # 4. Random Forest (100 árvores)
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, random_state=0)
    gerar_linha_e_matrizes_excel(rf, "Random Forest", padronizar=False)
    
    # 5. SVM (RBF C=10)
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=1)
    gerar_linha_e_matrizes_excel(svm, "SVM", padronizar=True)
    
    # 6. Rede Neural (L-BFGS)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='lbfgs', max_iter=2000, random_state=1)
    gerar_linha_e_matrizes_excel(mlp, "Redes Neurais", padronizar=True)