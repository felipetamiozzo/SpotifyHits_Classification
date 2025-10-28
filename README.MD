


# Projeto de Classificação de Sucesso Musical (Spotify Hits & Flops)

## 🎯 Visão Geral do Projeto

Este projeto foca no desenvolvimento de **modelos de classificação** de Machine Learning para prever a probabilidade de uma faixa musical se tornar um sucesso (**"Hit"**) ou um não-sucesso generalizado (**"Flop"**) no *mainstream*, com base em suas características técnicas.

Busca-se não apenas criar um modelo preditivo robusto, mas também fornecer *insights* valiosos sobre o "DNA" técnico que impulsiona o sucesso musical na plataforma Spotify.


## Objetivo e Contexto

O cerne deste trabalho é a utilização de um *dataset* exclusivo, obtido a partir da **Web API do Spotify**, contendo diversas *features* técnicas por faixa.

### Variável Alvo (`Target`):

  * **1 (Hit):** Faixa com alta probabilidade de sucesso generalizado ou popularidade no *mainstream*.
  * **0 (Flop):** Faixa que, embora possa ser de alta qualidade, **provavelmente não alcançará popularidade generalizada** (não um sucesso de nicho ou não-popular).

### Features Principais (Exemplos):

`danceability`, `energy`, `valence`, `tempo`, `loudness`, `instrumentalness`, etc.



##  Metodologia e Modelos

A análise segue um fluxo de trabalho padronizado, garantindo a reprodutibilidade e a robustez dos modelos:

1.  **Análise Exploratória de Dados (EDA):** Investigação de correlações (ex: `instrumentalness` vs. `target`), distribuições e *outliers* para guiar o *feature engineering*.
2.  **Pré-processamento de Dados:** Normalização (`MinMaxScaler`) via **Pipeline** para garantir que a escala das *features* não influencie indevidamente modelos sensíveis à distância (KNN, Regressão Logística).
3.  **Modelagem e Avaliação:** Treinamento e comparação dos seguintes algoritmos de classificação, com foco nas métricas de **Acurácia**, **Precisão** e **Recall** (sensibilidade):
      * Regressão Logística
      * K-Nearest Neighbors (KNN)
      * Árvore de Decisão
      * Random Forest
      * Gradient Boosting



##  Como Executar o Projeto Localmente

Para replicar esta análise e treinar os modelos em sua máquina, siga os passos abaixo:

### 1\. Requisitos

Certifique-se de ter o Python 3.8+ instalado.

Instale as bibliotecas necessárias usando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2\. Estrutura do Repositório

| Pasta/Arquivo | Conteúdo |
| :--- | :--- |
| `data/raw/` | Dataset original do Spotify (ex: `spotify_tracks.csv`). |
| `notebooks/` | Documentação completa do projeto, EDA e Modelagem. |
| `src/` | Funções auxiliares e Pipelines de pré-processamento (limpeza/transformação). |
| `README.md` | Este arquivo. |

### 3\. Ordem de Execução

1.  Abra o ambiente Jupyter (Lab/Notebook).
2.  Execute os notebooks na pasta `notebooks/` na ordem numérica (ex: `1.0-EDA_Limpeza.ipynb` seguido de `2.0-Modelagem.ipynb`).

-----

## 🌟 Relevância para a Indústria

Este projeto demonstra a aplicação prática de *data science* para:

  * **Decisão de Produção:** Artistas e produtores podem otimizar *features* como `loudness` e `danceability` para aumentar o potencial *mainstream*.
  * **Marketing Estratégico:** Entender quais atributos técnicos de uma faixa devem ser priorizados em campanhas de *playlisting*.
  * **Validação de Negócios:** Mensurar o valor preditivo das *features* fornecidas pela API do Spotify.