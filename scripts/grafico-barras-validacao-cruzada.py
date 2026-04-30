import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def gerar_grafico_importancia_exato():
    print("Gerando Gráfico de Importância de Variáveis (Sincronizado com a Tabela)...")
    
    # Usando os mesmos dados exatos da sua Tabela 5 (LaTeX) para manter coerência visual
    dados = {
        'Atributo Biológico': ['Odor', 'Cor Impressão Esporos', 'Tamanho Lamela', 'Espaçamento Lamela', 'Formato Estipe'],
        'Importância (%)': [62.38, 21.25, 11.21, 3.02, 2.14]
    }
    df = pd.DataFrame(dados)
    
    plt.figure(figsize=(10, 5))
    barras = sns.barplot(x='Importância (%)', y='Atributo Biológico', data=df, palette='magma')
    
    plt.title('Importância Relativa na Árvore de Decisão (max_depth=8)', fontsize=14, pad=15)
    plt.xlabel('Importância Relativa (%)', fontsize=12)
    plt.ylabel('')
    
    # Adicionando a % na frente de cada barra
    for p in barras.patches:
        largura = p.get_width()
        plt.text(largura + 0.5, p.get_y() + p.get_height()/2 + 0.1, 
                 f'{largura:.2f}%', va='center', fontsize=11, fontweight='bold')
                 
    plt.xlim(0, 70)
    plt.tight_layout()
    plt.savefig('resultados/grafico_importancia_arvore8.png', dpi=300)
    plt.close()


def gerar_grafico_validacao_cruzada():
    print("Gerando Gráfico da Validação Cruzada...")
    
    # Valores de Recall extraídos da sua Tabela de Validação Cruzada (LaTeX)
    modelos = ['KNN', 'Naive Bayes', 'SVM', 'Árvores de Decisão', 'Random Forest', 'Redes Neurais']
    recall_cv = [0.9992, 0.9992, 0.9995, 1.0000, 1.0000, 1.0000]
    
    plt.figure(figsize=(10, 6))
    
    # Cores: Verde se atingiu 100%, vermelho caso contrário
    cores = ['#2ecc71' if r == 1.0 else '#e74c3c' for r in recall_cv]
    
    barras = plt.barh(modelos, recall_cv, color=cores)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlim(0.998, 1.0002) # Zoom para destacar os empates técnicos em 100%
    
    plt.title('Estabilidade da Revocação (Recall) na Validação Cruzada (5 Folds)', fontsize=14, pad=15)
    plt.xlabel('Recall Médio (Classe Venenosa)', fontsize=12)
    plt.gca().invert_yaxis()
    
    for barra in barras:
        largura = barra.get_width()
        plt.text(largura + 0.00005, barra.get_y() + barra.get_height()/2, 
                 f'{largura:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('resultados/grafico-barras-validacao-cruzada.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    gerar_grafico_importancia_exato()
    gerar_grafico_validacao_cruzada()
    print("\nGráficos prontos!")