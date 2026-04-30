from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Cria a pasta 'resultados' automaticamente se ela não existir
if not os.path.exists('resultados'):
    os.makedirs('resultados')

def plotar_matriz_confusao(cm, nome_modelo):
    """
    Plota e salva a matriz de confusão como um mapa de calor.
    """
    plt.figure(figsize=(6, 4))
    # O cmap='Reds' ajuda na semântica visual do perigo (venenoso)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Comestível (0)', 'Venenoso (1)'],
                yticklabels=['Comestível (0)', 'Venenoso (1)'])
    
    plt.title(f'Matriz de Confusão - {nome_modelo}', pad=15)
    plt.ylabel('Rótulo Verdadeiro (Realidade)')
    plt.xlabel('Rótulo Previsto (Modelo)')
    plt.tight_layout()
    
    # Salva a imagem na pasta resultados para você usar no artigo Word/PDF
    nome_arquivo = f"resultados/cm_{nome_modelo.replace(' ', '_').lower()}.png"
    plt.savefig(nome_arquivo, dpi=300)
    plt.show()
    
def plotar_curva_roc(y_test, y_proba, nome_modelo):
    """ Plota a Curva ROC e calcula a AUC. """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {nome_modelo}', pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"resultados/roc_{nome_modelo.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()

def plotar_curva_pr(y_test, y_proba, nome_modelo):
    """ Plota a Curva Precision-Recall. """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall (Revocação)')
    plt.ylabel('Precision (Precisão)')
    plt.title(f'Curva Precision-Recall - {nome_modelo}', pad=15)
    plt.tight_layout()
    plt.savefig(f"resultados/pr_{nome_modelo.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()

def relatorio_metricas(y_test, y_pred, nome_modelo):
    """ Imprime o relatório detalhado de Precision, Recall e F1-Score. """
    print(f"\n--- Relatório de Classificação: {nome_modelo} ---")
    print(classification_report(y_test, y_pred, target_names=['Comestível (0)', 'Venenoso (1)']))
    
def plotar_importancia_variaveis(importancias, nomes_features, nome_modelo):
    """ Plota as 10 variáveis mais importantes do modelo de árvore. """
    # Cria um DataFrame para facilitar a ordenação
    df_imp = pd.DataFrame({'Feature': nomes_features, 'Importancia': importancias})
    df_imp = df_imp.sort_values(by='Importancia', ascending=False).head(10)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importancia', y='Feature', data=df_imp, palette='viridis')
    plt.title(f'Top 10 Variáveis Mais Importantes - {nome_modelo}', pad=15)
    plt.xlabel('Importância (Redução de Impureza)')
    plt.ylabel('Variáveis')
    plt.tight_layout()
    plt.savefig(f"resultados/importancia_{nome_modelo.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()    
    
    
    
    