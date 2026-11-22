# Comparação controlada de algoritmos de aprendizado de máquina para análise de sentimentos no IMDb

Código, dados processados e resultados do artigo *“Comparação controlada de algoritmos de
aprendizado de máquina para análise de sentimentos no conjunto IMDb”*, desenvolvido para a
disciplina de Metodologia de Pesquisa do IFSULDEMINAS – Campus Muzambinho (2026).

O experimento compara quatro classificadores — **Naive Bayes Multinomial**, **SVM linear**,
**regressão logística** e **perceptron multicamadas (MLP)** — sobre as 50 mil críticas de
filmes do conjunto IMDb, mantendo constantes o corpus, o pré-processamento, a divisão dos
dados, as sementes aleatórias e o ambiente de execução. O pré-processamento é aplicado de
forma idêntica a todos os modelos, de modo que a comparação se concentra no classificador.

## Principais resultados

Validação cruzada estratificada de 5 partições (média ± desvio-padrão) e avaliação única no
conjunto de teste de 10 mil críticas:

| Algoritmo           | F1 (CV, %)     | F1 (teste, %) | ROC AUC (teste, %) | Treino do classificador (s) | Modelo (MB) |
|---------------------|----------------|---------------|--------------------|-----------------------------|-------------|
| Naive Bayes         | 87,91 ± 0,48   | 88,40         | 95,22              | 0,02                        | 3,2         |
| SVM linear          | 90,32 ± 0,19   | 90,40         | 96,60              | 0,63                        | 2,1         |
| Regressão logística | 90,33 ± 0,30   | 90,68         | 96,68              | 0,77                        | 2,1         |
| MLP                 | 88,27 ± 0,39   | 88,26         | 95,26              | 1,53                        | 59,3        |

A regressão logística obteve o maior F1-score no teste; a diferença em relação à SVM linear
não é significativa (teste de McNemar, p = 0,247), mas é significativa em relação ao Naive
Bayes e ao MLP (p < 0,001). Naive Bayes e MLP empatam entre si (p = 0,722). Os valores
completos estão em [`results/metrics_test.csv`](results/metrics_test.csv) e
[`results/metrics_cv.csv`](results/metrics_cv.csv); as figuras do artigo estão em
[`results/figures`](results/figures) e a análise de erros em
[`results/error_analysis.json`](results/error_analysis.json).

## Estrutura do repositório

```
.
├── src/
│   ├── config.py          # sementes, caminhos, grades de hiperparâmetros, TF-IDF
│   ├── data.py            # download, extração, carga e divisão do corpus
│   ├── preprocessing.py   # limpeza, tokenização, negação, stemming
│   ├── models.py          # pipelines de cada classificador
│   ├── evaluation.py      # métricas, bootstrap, McNemar
│   └── experiment.py      # busca em grade, validação cruzada, medição de custo
├── scripts/
│   ├── run_experiments.py # executa o experimento completo
│   ├── error_analysis.py  # análise de erros por tipo de texto
│   └── make_figures.py    # gera as figuras do artigo
├── tests/                 # testes unitários do pré-processamento
├── results/               # tabelas, figuras e previsões (versionadas)
└── data/                  # corpus bruto e cache (não versionado)
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reprodução

```bash
# 1. Experimento completo (baixa o corpus automaticamente, ~84 MB)
python scripts/run_experiments.py

# 2. Figuras do artigo
python scripts/make_figures.py

# 3. Análise de erros
python scripts/error_analysis.py

# Teste rápido, com 300 críticas por classe (~10 s)
python scripts/run_experiments.py --sample 300 --results-dir results/smoke
```

Todos os resultados são gravados em `results/`, e o log completo da execução fica em
`results/run.log`. O corpus é extraído para `data/raw/` e o texto pré-processado é
armazenado em cache em `data/cache/`.

### Protocolo

* Divisão estratificada 80/20 (40 mil treino / 10 mil teste), semente 42.
* Validação cruzada estratificada de 5 partições sobre o conjunto de treinamento.
* Seleção de hiperparâmetros por busca exaustiva (`GridSearchCV`), critério F1-score.
* Reajuste no conjunto de treinamento completo e avaliação única no teste.
* Métricas: acurácia, precisão, recall, F1-score e ROC AUC, além de tempo de treinamento,
  tempo de inferência, tamanho do modelo e número de parâmetros.

## Testes

```bash
python -m pytest tests -q
```

## Como citar

```
OLIVEIRA, Guilherme Avelar de; LABANCA, Thiago. Comparação controlada de algoritmos de
aprendizado de máquina para análise de sentimentos no conjunto IMDb. 2026. Artigo
(Metodologia de Pesquisa) – IFSULDEMINAS, Campus Muzambinho, 2026.
```

## Licença

Distribuído sob a licença MIT. O conjunto IMDb é disponibilizado por Maas et al. (2011)
para uso em pesquisa e não é redistribuído neste repositório.
