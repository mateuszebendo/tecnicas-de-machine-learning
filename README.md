# 🍄 Classificação de Cogumelos: Avaliação de Modelos de Machine Learning

Este repositório contém a pesquisa, o código-fonte e a análise estatística desenvolvida para a classificação de risco biológico de cogumelos (**comestíveis vs. venenosos**).

O projeto aplica e compara seis diferentes algoritmos de Aprendizado de Máquina Supervisionado, com foco estrito na métrica de **Revocação (Recall)** para a classe tóxica, garantindo que falsos negativos (classificar um cogumelo letal como seguro) sejam minimizados ou erradicados.

---

## 🗂️ Estrutura do Repositório

O projeto adota uma arquitetura modular, separando processamento, modelagem exploratória e validação estatística:

```text
.
├── dados/                  # Dataset original (mushrooms.csv) da UCI Machine Learning
├── docs/                   # Documentação, análise de variáveis, relatórios e validação (Excel/Word)
├── scripts/                # Códigos-fonte em Python
│   ├── pre_processamento.py  # Pipeline central: limpeza e One-Hot Encoding (drop_first=True)
│   ├── visualizacao.py       # Módulo gerador de Matrizes de Confusão, Curvas ROC/PR e Importância
│   ├── knn.py                # K-Nearest Neighbors (Investigação de hiperparâmetro K)
│   ├── naive_bayes.py        # Modelos Probabilísticos (Gaussian, Bernoulli, Multinomial, etc.)
│   ├── arvores.py            # Árvores de Decisão e Random Forest (Critérios de Entropia e Gini)
│   ├── svm.py                # Support Vector Machines (Kernels Linear, RBF, Poly e Sigmoid)
│   ├── redes_neurais.py      # Multi-Layer Perceptron (Otimizadores SGD, Adam e L-BFGS)
│   ├── preencher_planilha.py # Script de Validação Cruzada Estratificada (5-Fold)
│   ├── grafico-barras-validacao-cruzada.png # Visualização da estabilidade do Recall (5-Fold) entre os algoritmos
│   └── resultados/           # Imagens geradas automaticamente (.png) de todos os testes
```

---

## 🧠 Modelos Avaliados

* K-Nearest Neighbours (KNN)
* Naive Bayes
* Árvores de Decisão (Decision Trees)
* Random Forest
* Support Vector Machines (SVM)
* Redes Neurais Artificiais (MLP)

---

## 🔬 Metodologia

**Pré-processamento:**
A base de dados é estritamente categórica. O tratamento consistiu na binarização da variável alvo e na conversão das características fenotípicas através de One-Hot Encoding.

**Prevenção de Vazamento de Dados (Data Leakage):**
Para modelos sensíveis a distâncias geométricas (SVM, KNN e MLP), a padronização (StandardScaler) foi isolada estritamente aos conjuntos de treinamento, simulando o ambiente real de produção.

**Avaliação Assimétrica:**
A acurácia global não foi considerada suficiente para este problema. Os modelos foram julgados pelo **Recall da classe venenosa**, penalizando configurações que permitiam a passagem de espécimes letais.

**Validação Cruzada:**
As melhores configurações de cada algoritmo foram submetidas ao método **Stratified K-Fold (k=5)** para atestar a estabilidade e extrair as médias e desvios padrões de performance.

---

## 🚀 Como Executar

Certifique-se de ter o Python 3.x instalado.

### 1. Instale as dependências:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 2. Executando testes isolados (Fase Exploratória):

Para rodar a investigação de hiperparâmetros de um modelo específico e gerar seus respectivos gráficos na pasta `scripts/resultados/`:

```bash
cd scripts
python arvores.py
```

### 3. Executando a Validação Cruzada:

Para executar a bateria de testes definitivos com os modelos campeões e gerar as métricas consolidadas:

```bash
cd scripts
python preencher_planilha.py
```

---

## 📊 Principais Resultados

* A **Árvore de Decisão** provou ser a arquitetura ideal pelo *Princípio da Parcimônia*: alcançou taxa de erro nula (**100% de Recall e Precisão**) com baixíssimo custo computacional.
* O desempenho superou o Naive Bayes e se equiparou ao Random Forest e às Redes Neurais.
* A análise de **Feature Importance** demonstrou que o **odor do espécime** (especialmente ausência de odor ou odor fétido) é a característica isolada mais preditiva para detectar toxicidade na família *Agaricus* e *Lepiota*.

---
