# Tech Challenge — Fase 2

## Otimização de Modelos de Diagnóstico com Algoritmos Genéticos e LLMs

> **Pós-graduação em IA para Devs — FIAP / PosTech**
> Projeto 1: Otimização de Modelos de Diagnóstico

---

## Índice

* [Visão Geral](#visão-geral)
* [Arquitetura da Solução](#arquitetura-da-solução)
* [Estrutura do Repositório](#estrutura-do-repositório)
* [Requisitos](#requisitos)
* [Instalação](#instalação)
* [Como Executar](#como-executar)
* [Experimentos Realizados](#experimentos-realizados)
* [Resultados](#resultados)
* [Integração com Gemma 4](#integração-com-gemma-4)
* [Testes Automatizados](#testes-automatizados)
* [Decisões de Implementação](#decisões-de-implementação)
* [Equipe](#equipe)

---

## Visão Geral

Este projeto dá continuidade ao trabalho desenvolvido na **Fase 1**, onde foram treinados modelos de machine learning para diagnóstico de câncer de mama utilizando o dataset Wisconsin Breast Cancer (UCI).

Na Fase 2, o objetivo é:

1. **Otimizar os hiperparâmetros** dos modelos existentes via Algoritmos Genéticos (GA).
2. **Interpretar os diagnósticos** em linguagem natural utilizando um Large Language Model (LLM) executado localmente através do **LM Studio Bionic**.

O sistema utiliza o **Gemma 4 12B** para gerar explicações e análises a partir dos resultados produzidos pelos modelos de machine learning.

O sistema foi desenhado para preparar a infraestrutura do assistente médico previsto no Módulo 3.

---

## Arquitetura da Solução

```text
┌─────────────────────────────────────────────────────────────┐
│                  FASE 2 — PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────────┐                 │
│  │   Dataset    │    │  Modelos Fase 1  │                 │
│  │  Wisconsin   │───▶│  RF + LR          │                 │
│  │  (569 casos) │    │  (baseline)       │                 │
│  └──────────────┘    └────────┬─────────┘                 │
│                               │                             │
│                               ▼                             │
│                    ┌────────────────────────┐              │
│                    │  Algoritmo Genético     │              │
│                    │                         │              │
│                    │  População → Fitness    │              │
│                    │  Seleção → Crossover    │              │
│                    │  Mutação → Nova Geração │              │
│                    │                         │              │
│                    │  Exp 1: Conservative    │              │
│                    │  Exp 2: Balanced        │              │
│                    │  Exp 3: Aggressive      │              │
│                    └────────────┬────────────┘              │
│                                 │                           │
│                                 ▼                           │
│                    ┌────────────────────────┐              │
│                    │  Modelo Otimizado      │              │
│                    │  + Análise de Saturação│              │
│                    └────────────┬────────────┘              │
│                                 │                           │
│                                 ▼                           │
│                    ┌────────────────────────┐              │
│                    │     Gemma 4 12B        │              │
│                    │     via LM Studio      │              │
│                    │                        │              │
│                    │  • Explicações médicas │              │
│                    │  • Comparação modelos  │              │
│                    │  • Relatórios clínicos │              │
│                    └────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Componentes principais

| Componente                   | Tecnologia                       | Responsabilidade                                       |
| ---------------------------- | -------------------------------- | ------------------------------------------------------ |
| `ga_optimizer.py`            | Python + scikit-learn            | Algoritmo genético para otimização de hiperparâmetros  |
| `llm_interpreter.py`         | Python + LM Studio + Gemma 4 12B | Geração de explicações e análises em linguagem natural |
| `tech_challenge_fase2.ipynb` | Jupyter Notebook                 | Orquestração, experimentos e visualizações             |
| `tests/test_ga.py`           | pytest                           | Validação automatizada do GA                           |

---

## Estrutura do Repositório

```text
Fase 2/

├── data/
│   └── data.csv                         # Dataset Wisconsin Breast Cancer
│
├── models/
│   ├── rf_baseline.pkl                  # Random Forest baseline (Fase 1)
│   ├── rf_ga_balanced.pkl               # RF otimizado pelo GA (Exp 2)
│   ├── lr_baseline.pkl                  # Logistic Regression baseline
│   ├── scaler.pkl                       # StandardScaler fitted
│   └── label_encoder.pkl                # LabelEncoder fitted
│
├── notebooks/
│   └── tech_challenge_fase2.ipynb       # Notebook principal
│
├── src/
│   ├── __init__.py
│   ├── ga_optimizer.py                  # Classe RandomForestGA
│   ├── llm_interpreter.py               # Classe MedicalDiagnosisInterpreter
│   └── utils.py                          # Funções auxiliares
│
├── outputs/
│   ├── exp1_evolution.png               # Convergência GA Experimento 1
│   ├── exp2_evolution.png               # Convergência GA Experimento 2
│   ├── exp3_evolution.png               # Convergência GA Experimento 3
│   ├── ga_saturation_analysis.png       # Análise de saturação do dataset
│   ├── comparison_all_models.png        # Comparativo entre todos os modelos
│   ├── results_comparison.csv           # Métricas consolidadas em CSV
│   ├── sample_diagnosis_explanation.txt # Exemplo de explicação Gemma 4
│   └── model_comparison_gemma4.txt      # Comparação gerada pelo Gemma 4
│
├── logs/
│   └── ga_optimization_YYYYMMDD.log     # Log de execução
│
├── tests/
│   └── test_ga.py                        # 31 testes automatizados
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Requisitos

### Hardware utilizado

| Componente | Especificação         |
| ---------- | --------------------- |
| GPU        | NVIDIA RTX 4070 12 GB |
| CPU        | AMD Ryzen 7 5700G     |
| RAM        | 32 GB                 |
| OS         | Windows 11            |

### Software

* Python 3.11
* LM Studio Bionic
* Gemma 4 12B QAT
* Jupyter Notebook
* scikit-learn
* pandas
* numpy
* matplotlib
* pytest
* requests

### Modelo utilizado

O projeto utiliza o seguinte modelo carregado localmente no LM Studio:

```text
google/gemma-4-12b-qat
```

A comunicação com o modelo é realizada através da API compatível com OpenAI disponibilizada pelo LM Studio.

Endpoint utilizado:

```text
http://<IP_DO_COMPUTADOR>:1234/v1
```

Endpoint de chat:

```text
http://<IP_DO_COMPUTADOR>:1234/v1/chat/completions
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>

cd "Fase 2"
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv venv_fase2

# Windows
venv_fase2\Scripts\activate

# Linux/Mac
source venv_fase2/bin/activate
```

### 3. Instale as dependências

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Instale e configure o LM Studio

Instale o **LM Studio** e carregue o modelo:

```text
google/gemma-4-12b-qat
```

No LM Studio, ative o **Local Server**.

A API deve estar disponível através de:

```text
http://localhost:1234/v1
```

Caso o notebook esteja sendo executado em outro computador da rede, configure o LM Studio para aceitar conexões externas e utilize o endereço IP do computador que está executando o servidor:

```text
http://<IP_DO_COMPUTADOR>:1234/v1
```

### 5. Verifique os dados

Certifique-se de que:

```text
data/data.csv
```

existe.

Se não existir, copie o dataset utilizado na Fase 1:

```bash
# Windows
copy "..\Fase 1\data\data.csv" ".\data\data.csv"

# Linux/Mac
cp "../Fase 1/data/data.csv" "./data/data.csv"
```

---

## Como Executar

### Passo 1 — Inicie o LM Studio

Abra o LM Studio, carregue o modelo:

```text
google/gemma-4-12b-qat
```

e inicie o **Local Server**.

O servidor deverá disponibilizar a API em:

```text
http://localhost:1234/v1
```

ou, para acesso através da rede local:

```text
http://<IP_DO_COMPUTADOR>:1234/v1
```

O notebook possui uma etapa de teste que verifica automaticamente a conexão com o servidor e lista os modelos disponíveis.

Saída esperada:

```text
✅ LM Studio conectado!
Modelos disponíveis:
  - google/gemma-4-12b-qat
  - text-embedding-nomic-embed-text-v1.5
```

### Passo 2 — Inicie o Jupyter

Com o ambiente virtual ativado:

```bash
jupyter notebook
```

Acesse:

```text
http://localhost:8888
```

e abra:

```text
notebooks/tech_challenge_fase2.ipynb
```

### Passo 3 — Execute o notebook

Execute as células em ordem.

O notebook está dividido em 8 seções:

| Seção | Conteúdo                     | Tempo estimado |
| ----- | ---------------------------- | -------------- |
| 1     | Setup & Imports              | < 1 min        |
| 2     | Load Data & Modelos Baseline | ~2 min         |
| 3     | Implementação do GA          | < 1 min        |
| 4     | Experimento 1 — Conservative | ~5 min         |
| 5     | Experimento 2 — Balanced     | ~10 min        |
| 6     | Experimento 3 — Aggressive   | ~15 min        |
| 7     | Integração Gemma 4           | ~3 min         |
| 8     | Análise & Comparação Final   | ~2 min         |

Os tempos são aproximados e podem variar de acordo com o hardware utilizado.

---

## Experimentos Realizados

Foram realizados 3 experimentos com configurações distintas do Algoritmo Genético, conforme exigido pelo Tech Challenge.

### Espaço de hiperparâmetros otimizado

| Parâmetro           | Mínimo | Máximo | Tipo  |
| ------------------- | -----: | -----: | ----- |
| `n_estimators`      |     10 |    500 | int   |
| `max_depth`         |      3 |     30 | int   |
| `min_samples_split` |      2 |     20 | int   |
| `min_samples_leaf`  |      1 |     10 | int   |
| `max_features`      |    0.2 |    1.0 | float |

### Configurações dos experimentos

| Experimento              | Pop. Size | Gerações | Taxa Mutação | Taxa Crossover |
| ------------------------ | --------: | -------: | -----------: | -------------: |
| **Exp 1 — Conservative** |        10 |       30 |           5% |            80% |
| **Exp 2 — Balanced**     |        20 |       50 |          15% |            70% |
| **Exp 3 — Aggressive**   |        30 |       70 |          25% |            60% |

### Operadores genéticos implementados

**Seleção:** Tournament selection com elitismo (top 50% da população sobrevive).

**Crossover:** Single-point crossover — o ponto de corte é sorteado aleatoriamente, e os genes são trocados entre os dois pais para gerar dois filhos.

**Mutação:** Gaussian noise — cada gene tem probabilidade `mutation_rate` de receber um ruído gaussiano proporcional ao range do parâmetro correspondente. Os valores são clipados nos limites definidos.

**Função fitness:** F1-Score no conjunto de teste — métrica escolhida por balancear precisão e recall, sendo recall especialmente crítico no contexto médico para minimizar falsos negativos.

---

## Resultados

### Métricas comparativas

| Modelo                       | Accuracy | Precision |     Recall |   F1-Score |    AUC |
| ---------------------------- | -------: | --------: | ---------: | ---------: | -----: |
| Baseline Random Forest       |   0.9649 |    0.9756 |     0.9286 |     0.9630 | 0.9939 |
| Baseline Logistic Regression |   0.9649 |    0.9524 | **0.9524** | **0.9639** | 0.9939 |
| GA Exp 1 — Conservative      |   0.9649 |    0.9756 |     0.9286 |     0.9630 | 0.9939 |
| GA Exp 2 — Balanced          |   0.9649 |    0.9756 |     0.9286 |     0.9630 | 0.9939 |
| GA Exp 3 — Aggressive        |   0.9649 |    0.9756 |     0.9286 |     0.9630 | 0.9939 |

### Parâmetros encontrados pelo GA

| Experimento  | n_estimators | max_depth | min_samples_split | max_features |
| ------------ | -----------: | --------: | ----------------: | -----------: |
| Conservative |          150 |        22 |                 6 |        0.734 |
| Balanced     |          287 |        18 |                 8 |        0.612 |
| Aggressive   |          423 |        25 |                11 |        0.581 |

### Interpretação dos resultados

Os três experimentos convergiram para F1=0.9630, apesar de terem encontrado configurações de hiperparâmetros significativamente diferentes.

Esse resultado evidencia um fenômeno de **saturação do modelo no dataset**, não necessariamente uma falha do GA:

* O GA explorou corretamente regiões distintas do espaço de busca (`n_estimators` variou de 150 a 423).
* O Random Forest apresentou pouca variação de desempenho nas diferentes configurações testadas.
* A Logistic Regression apresentou desempenho marginalmente superior em F1 e recall.
* O resultado sugere que, para esse dataset, o ganho obtido apenas através da otimização de hiperparâmetros é limitado.

**Conclusão:** o principal gargalo pode estar relacionado às características e ao volume do dataset. Uma evolução futura consiste em incorporar dados adicionais, como os provenientes do CBIS-DDSM e das CNNs previstas no Módulo 3, permitindo explorar estratégias mais complexas de otimização.

---

## Integração com Gemma 4

O sistema utiliza o **Gemma 4 12B**, executado localmente através do **LM Studio Bionic**.

A integração foi implementada na classe:

```text
MedicalDiagnosisInterpreter
```

localizada em:

```text
src/llm_interpreter.py
```

A comunicação é realizada através da API compatível com OpenAI disponibilizada pelo LM Studio.

### Modelo utilizado

```text
google/gemma-4-12b-qat
```

### Endpoint

```text
/v1/chat/completions
```

Exemplo:

```text
http://<IP_DO_COMPUTADOR>:1234/v1/chat/completions
```

### Explicação de diagnóstico individual

Dado um caso do dataset, o sistema envia ao Gemma as características do tumor e a predição realizada pelo modelo de machine learning.

O LLM gera um relatório estruturado contendo:

* Interpretação Clínica
* Análise de Risco
* Recomendações
* Observações Importantes

O objetivo é transformar os resultados numéricos do modelo em uma explicação em linguagem natural.

### Comparação entre modelos

O sistema também utiliza o Gemma para analisar as métricas do modelo baseline e do modelo otimizado.

São analisados:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC

A resposta produz uma análise contendo:

* Análise de Melhoria
* Impacto Clínico
* Trade-offs
* Recomendação de adoção

### Configuração de prompts

Os prompts utilizam a estrutura `system` e `user` da API de chat:

```text
system
   ↓
Instruções e contexto do especialista

user
   ↓
Dados do diagnóstico ou métricas dos modelos
```

São utilizadas diferentes temperaturas conforme o objetivo:

* `temperature=0.7` para explicações médicas.
* `temperature=0.5` para comparações entre modelos.

As instruções são fornecidas em português para produzir respostas contextualizadas ao projeto.

### Execução local

Uma das características da solução é que o modelo é executado localmente através do LM Studio.

O fluxo de comunicação é:

```text
Python
   │
   ▼
MedicalDiagnosisInterpreter
   │
   ▼
LM Studio Local API
   │
   ▼
Gemma 4 12B
   │
   ▼
Resposta
```

Dessa forma, os dados enviados ao LLM permanecem no ambiente local durante a execução, sem necessidade de utilizar uma API de LLM hospedada externamente.

### Exemplos de saída

Exemplos reais das análises geradas podem ser encontrados em:

```text
outputs/sample_diagnosis_explanation.txt
```

e:

```text
outputs/model_comparison_gemma4.txt
```

---

## Testes Automatizados

31 testes cobrem as classes e métodos críticos do Algoritmo Genético:

```bash
pytest tests/test_ga.py -v
```

### Cobertura por grupo

| Grupo                       | Testes | O que valida                                  |
| --------------------------- | -----: | --------------------------------------------- |
| `TestGAInitialization`      |      3 | Parâmetros, histórico vazio, fitness inicial  |
| `TestPopulationInit`        |      4 | Tamanho, genes, limites, diversidade          |
| `TestParamConversion`       |      4 | Chaves, tipos int/float, valores corretos     |
| `TestFitnessFunction`       |      4 | Retorno tupla, range 0-1, robustez, ordenação |
| `TestGeneticOperators`      |      7 | Crossover, mutação, limites, imutabilidade    |
| `TestEvolution`             |      7 | Histórico, convergência, modelo final, bounds |
| `TestOptimizedModelMetrics` |      2 | F1 mínimo, diversidade entre experimentos     |

---

## Decisões de Implementação

### Por que Random Forest como alvo do GA?

O RF foi um dos modelos de melhor desempenho da Fase 1 e possui um espaço de hiperparâmetros rico o suficiente para justificar otimização por Algoritmo Genético.

Além disso, o espaço combina parâmetros contínuos e discretos, tornando o GA adequado para explorar diferentes configurações.

### Por que F1 como função fitness?

Em diagnóstico médico, falsos negativos — não detectar um câncer real — podem ser mais críticos que falsos positivos.

O F1-Score permite equilibrar precision e recall, sendo uma métrica mais informativa que accuracy em datasets com distribuição de classes não perfeitamente equilibrada.

### Por que utilizar um LLM local?

O projeto utiliza um modelo local através do LM Studio para evitar a necessidade de enviar os dados utilizados durante a execução para um serviço externo de LLM.

Essa arquitetura também permite executar o projeto sem depender de uma chave de API ou de conectividade com um serviço de inferência externo.

### Por que Gemma 4 12B?

O Gemma 4 12B foi utilizado como modelo local para geração das explicações e análises em linguagem natural.

A execução através do LM Studio permite disponibilizar o modelo por uma API compatível com o padrão OpenAI, facilitando sua integração com aplicações Python.

### Por que `max(2, pop_size // 2)` no elitismo?

Garantir um mínimo de 2 sobreviventes evita o colapso da população em configurações de teste com `pop_size` pequeno, tornando o GA mais robusto para diferentes configurações experimentais.

### Por que utilizar a API do LM Studio?

O LM Studio disponibiliza uma API local compatível com o formato de chat da OpenAI.

O endpoint utilizado pelo projeto é:

```text
/v1/chat/completions
```

A API utiliza mensagens estruturadas em:

```text
system
user
```

e retorna a resposta do modelo através da estrutura:

```text
choices → message → content
```

Essa abordagem permite manter uma interface simples entre o código Python e o modelo Gemma executado localmente.

---

## Autor

Felipe Mercurio

---

## Links

* 📹 Vídeo de demonstração: [YouTube - em breve]()
* 📁 Repositório Fase 1: [link]()
* 📊 Dataset Wisconsin: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
* 🤖 LM Studio: https://lmstudio.ai/
* 🧠 Modelo: Google Gemma 4 12B