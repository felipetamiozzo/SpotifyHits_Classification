# Projeto de Classificação Spotify Hits & Flops

Este projeto utiliza técnicas de Machine Learning para prever se uma música, com base em suas características técnicas da API do Spotify, será um **"Hit" (1)** ou **"Flop" (0)**.

O objetivo é construir um modelo de classificação robusto que possa auxiliar na tomada de decisões na indústria musical, identificando quais *features* (como dançabilidade, energia, loudness) são os preditores mais fortes de sucesso.

## 📈 Metodologia e Pipeline do Projeto

O projeto foi estruturado seguindo as seguintes etapas:

1.  **Coleta e Limpeza de Dados**:
    * Os dados brutos (dos anos 90, 00 e 10) foram carregados da pasta `Data/Raw/`.
    * Os conjuntos de dados foram concatenados.
    * Foram removidas 173 duplicatas com base no nome da música (`track`) e artista (`artist`).
    * O *dataset* final apresentou um excelente balanceamento de classes (50.2% Hits / 49.8% Flops).

2.  **Engenharia e Seleção de Features**:
    * Uma análise de correlação foi realizada para identificar *features* com baixo poder preditivo.
    * **Features Removidas (Ruído):** `liveness`, `tempo` e `speechiness` foram removidas por terem correlação próxima de zero com a variável alvo.
    * **Feature Criada:** Com base na forte correlação negativa de `instrumentalness` (-0.47), foi criada a *feature* `is_vocal_track` (1 se `instrumentalness` < 0.01, 0 caso contrário) para ajudar o modelo a identificar explicitamente músicas cantadas.

3.  **Pré-processamento**:
    * Os dados foram divididos em conjuntos de treino (80%) e teste (20%).
    * As *features* de treino e teste foram normalizadas usando `MinMaxScaler`.

4.  **Benchmark de Modelos (Baseline)**:
    * Sete algoritmos de classificação foram treinados e avaliados usando Validação Cruzada (5-folds) para garantir robustez.
    * O **Gradient Boosting** foi selecionado como o modelo campeão, apresentando a maior acurácia média (83.16%) e a maior estabilidade (menor desvio padrão).

| Modelo | Acurácia Média (CV) | Desvio Padrão (CV) |
| :--- | :---: | :---: |
| **Gradient Boosting** | **0.8316** | 0.0076 |
| XGBoost | 0.8234 | 0.0084 |
| MLP Classifier | 0.8224 | 0.0056 |
| Random Forest | 0.8132 | 0.0086 |
| Decision Tree | 0.8025 | 0.0090 |
| Logistic Regression | 0.7968 | 0.0060 |
| KNN | 0.7904 | 0.0079 |

5.  **Otimização de Hiperparâmetros (Tuning)**:
    * O modelo campeão (Gradient Boosting) foi otimizado usando `RandomizedSearchCV` com foco em maximizar o `f1-score`.
    * **Melhores Hiperparâmetros:** `{'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 3, 'learning_rate': 0.05}`.

## Resultados Finais

O modelo final (Gradient Boosting Otimizado) foi avaliado no conjunto de teste, alcançando os seguintes resultados:

* **Acurácia Geral**: 83%
* **AUC (ROC Score)**: 0.9037 (Excelente capacidade de distinção)

### Métricas para "Hit" (Classe 1):
* **Recall: 89%**
    * *Interpretação*: O modelo é excelente em "encontrar" os Hits. Ele identificou corretamente 89% de todas as músicas que foram de fato um sucesso.
* **Precisão: 80%**
    * *Interpretação*: Quando o modelo prevê que uma música será um "Hit", ele está correto 80% das vezes.
* **F1-Score: 0.84**

### Matriz de Confusão Final (Dados de Teste)

| | Previsão: Flop (0) | Previsão: Hit (1) |
| :--- | :---: | :---: |
| **Real: Flop (0)** | **1321 (VN)** | 398 (FP) |
| **Real: Hit (1)** | 185 (FN) | **1569 (VP)** |

## 📂 Estrutura do Projeto

```
├── Data/
│   ├── Raw/                # Dados brutos .csv
│   └── Processed/          # Dados limpos .parquet
├── Models/
│   └── gradient_boosting.pkl # Modelo treinado (não versionado)
├── src/
│   └── spotifyhits_classification.ipynb  # Notebook principal
├── app.py                  # (Opcional: Aplicação Streamlit/Flask)
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Como Executar

1.  Clone este repositório:
    ```bash
    git clone [URL_DO_REPOSITORIO]
    ```
2.  (Recomendado) Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  Abra e execute o notebook Jupyter localizado em:
    `src/spotifyhits_classification.ipynb`
```

