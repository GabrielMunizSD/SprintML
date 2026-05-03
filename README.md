# SprintML

# Sprint 3 — Modelagem Avançada, Tuning e Interpretabilidade

Case **iFood** — Previsão de Resposta a Campanha de Marketing.
Turma 1TIAPR-2025 (2º semestre).

## Sobre o projeto

Este projeto evolui a modelagem da Sprint 2 para um nível mais profissional, incluindo otimização de hiperparâmetros com múltiplas abordagens, validação cruzada robusta, comparação entre modelos e interpretabilidade com SHAP.

O objetivo de negócio é identificar quais clientes têm maior probabilidade de responder positivamente a uma campanha de marketing, permitindo direcionar recursos para o público certo e aumentar o ROI das ações.

## Mapeamento Sprint 3 → Notebook

| Item da Sprint 3 | Implementação |
|---|---|
| 1. Feature Engineering (incl. sazonalidade) | 11 features novas: Age, Days_Customer, Mes_Cadastro, Trimestre_Cadastro, DiaSemana_Cadastro, Ano_Cadastro, TotalMnt, TotalPurchases, TotalCampaigns, HasChildren, IsAlone |
| 2. Mínimo 2 modelos | Logistic Regression, Decision Tree, Random Forest |
| 3. Tuning — 2 abordagens | GridSearchCV (todos) + RandomizedSearchCV (RF) com comparação de tempo e qualidade |
| 4. Validação robusta | K-Fold estratificado (5 folds) com média ± desvio padrão |
| 5. Seleção do melhor modelo | Justificada por F1 médio + variância no CV |
| 6. Avaliação final (holdout) | Precision, Recall, F1, AUC-ROC, Accuracy + Matriz de confusão + curva PR |
| 7. Interpretabilidade SHAP | Summary plot + waterfall individual de 2 clientes + análise dos drivers |
| 8. Visualizações | Tuning curve, comparação Grid vs Random, CV bars, SHAP plots, feature importance |

## Dataset

Dataset do iFood com 2.240 clientes e 29 colunas, contendo informações sobre:

- **Perfil demográfico**: ano de nascimento, escolaridade, estado civil, renda, filhos
- **Comportamento de compra**: gastos por categoria (vinhos, frutas, carnes, peixes, doces, ouro), número de compras por canal (web, catálogo, loja)
- **Histórico de campanhas**: aceitou ou não as 5 campanhas anteriores
- **Variável-alvo**: `Response` (1 = aceitou a campanha, 0 = não aceitou)

**Distribuição do target:** ~85% Response=0 vs ~15% Response=1 (classes desbalanceadas).

## Estrutura do notebook

O notebook está organizado em quatro partes:

- **Parte 0** — Leitura, exploração e Feature Engineering (incluindo features sazonais)
- **Parte 1** — Statistical Computing: Teste Z para comparação de médias entre grupos
- **Parte 2** — Machine Learning & Modelling: baseline, tuning com 2 abordagens, validação cruzada e avaliação no holdout
- **Parte 3** — Interpretabilidade com SHAP: summary plot, waterfall individual e análise dos drivers
- **Parte 4** — Comparação entre os métodos de identificação de features importantes (Teste Z vs feature_importances_ vs SHAP)

## Tecnologias utilizadas

- **Python 3.10+**
- **pandas** e **numpy** — manipulação de dados
- **scikit-learn** — modelagem, tuning e validação cruzada
- **scipy** — Teste Z (Statistical Computing)
- **matplotlib** — visualizações
- **SHAP** — interpretabilidade

## Como executar

### Opção 1 — Google Colab (recomendado)

1. Abra o arquivo `sprint3-ifood-colab.ipynb` no Google Colab
2. Execute a primeira célula para instalar as dependências
3. Quando solicitado, faça upload do arquivo `data.csv`
4. Execute as demais células sequencialmente (Runtime → Run all)

### Opção 2 — Ambiente local

```bash
# Clonar o repositório
git clone <url-do-repo>
cd <pasta-do-repo>

# Instalar dependências
pip install pandas numpy scipy scikit-learn matplotlib shap pyarrow jupyter

# Abrir o notebook
jupyter notebook sprint3-ifood-colab.ipynb
```

Para rodar localmente, substitua a célula de upload do Colab por:

```python
df_raw = pd.read_csv('data.csv')
```

## Principais resultados

### Comparação entre abordagens de tuning (Random Forest)

| Abordagem | Combinações | Melhor F1 (CV) | Tempo (s) |
|---|---|---|---|
| GridSearchCV | 24 | 0.486 | ~40s |
| RandomizedSearchCV | 30 | 0.510 | ~118s |

RandomizedSearchCV explorou intervalos contínuos e encontrou melhor F1, mas com maior tempo de execução.

### Cross-Validation (5-fold estratificado)

| Modelo | F1 médio | F1 std | AUC médio |
|---|---|---|---|
| LogReg Tunado | 0.544 | 0.033 | 0.894 |
| DTree Tunado | 0.464 | 0.049 | 0.789 |
| RForest Tunado | 0.464 | 0.029 | 0.883 |

A Regressão Logística tunada se destacou pela combinação de F1 mais alto e baixa variância entre folds — sinal de boa generalização.

### Top drivers identificados pelo SHAP

1. **Recency** — quanto tempo desde a última compra
2. **IsAlone** — cliente solteiro/separado/viúvo
3. **NumStorePurchases** — número de compras em loja física
4. **Days_Customer** — tempo como cliente
5. **Mes_Cadastro** — mês de cadastro (efeito sazonal)

## Insights de negócio

- O **comportamento individual** (Recency, gastos, número de compras) pesa mais que **variáveis demográficas puras** (idade, estado civil)
- Clientes com histórico de aceitação de campanhas anteriores têm probabilidade muito maior de aceitar a próxima
- Features sazonais (`Mes_Cadastro`, `Trimestre_Cadastro`) tiveram impacto secundário, mas não desprezível
- Para próximas campanhas, recomenda-se priorizar clientes com baixo Recency, alto TotalMnt e histórico de aceitação prévia

## Estrutura do repositório

```
.
├── sprint3-ifood-colab.ipynb   # Notebook principal
├── data.csv                    # Dataset (não versionar se sensível)
└── README.md                   # Este arquivo
```

## Próximas iterações

- Testar **Bayesian Optimization** (Optuna) como terceira abordagem de tuning
- Avaliar **balanceamento de classes** com SMOTE ou `class_weight`
- Calibrar threshold de decisão para maximizar utilidade de negócio (custo de falso positivo vs falso negativo)
- Testar **gradient boosting** (XGBoost, LightGBM) como modelo adicional

## Autores

Rafael Tavares - 567357
Gabriel Muniz - 568237
Yuri Quirino - 568512
Leonardo Barros - 566788
Marcelo Augusto - 567176v
