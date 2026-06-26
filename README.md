Este trabalho utilizou Claude (Anthropic, Opus 4.8) para apoio em criação e depuração dos códigos .py e elaborações visuais conforme ppt anexado na pasta docs. Todo o conteúdo técnico e associações lógicas foram desenvolvidas e validadas pelos autores com este apoio. 

# Predição de Compatibilidade Auditor–Projeto com Modelos Ensemble

Framework de aprendizado de máquina para prever a compatibilidade (*fit score*) entre auditores e projetos de auditoria, com suporte à decisão de alocação via matriz de scores e heurísticas de atribuição.

Código e dados de reprodução do artigo *"Modelos de Aprendizado de Máquina Ensemble para Predição de Compatibilidade entre Auditores e Projetos: Uma Abordagem por Score de Fit com Dados Sintéticos"*.

## Visão geral

O framework opera em três etapas:

1. **Dataset sintético** — perfis de auditores e requisitos de projetos gerados com base no IIA Internal Auditing Competency Framework
2. **Modelos ensemble** — cinco algoritmos comparados para prever o *fit score*
3. **Matriz de scores** — tradução das previsões em suporte à decisão de alocação, com heurísticas Greedy e Húngaro

## Estrutura do repositório

```
.
├── data/                       # CSVs de entrada (artefato canônico)
│   ├── auditores.csv           #   200 auditores × 47 atributos
│   ├── projetos.csv            #   300 projetos × 48 atributos
│   └── alocacoes.csv           #   1.500 pares (treino do ML)
├── src/
│   ├── config.py               # Caminhos e parâmetros centralizados
│   ├── fit_score.py            # Cálculo do score de compatibilidade
│   ├── 00_generate_dataset.py  # Gerador dos dados sintéticos (metodológico)
│   ├── 01_train_models.py      # Treina e compara os 5 modelos
│   ├── 02_heuristics.py        # Matriz de scores + Greedy vs. Húngaro
│   └── 03_figures.py           # Gera as figuras do artigo
├── notebooks/
│   └── pipeline_explicado.ipynb  # Notebook didático célula a célula
├── outputs/                    # Artefatos gerados (figuras, métricas)
├── docs/                       # Figura do framework (editável)
├── run_all.py                  # Executa o pipeline completo
├── requirements.txt
├── LICENSE
└── README.md
```

## Instalação

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

## Reprodução

Para rodar o pipeline completo de uma vez:

```bash
python run_all.py
```

Ou cada etapa individualmente:

```bash
python src/01_train_models.py   # treina modelos -> outputs/ml_resultados.pkl
python src/02_heuristics.py     # matriz + heurísticas -> outputs/matriz_5x5.npy
python src/03_figures.py        # figuras -> outputs/figures/
```

Todos os resultados são determinísticos (`random_state=42`); rodar em qualquer máquina reproduz exatamente os números do artigo.

### Geração dos dados

Os três CSVs em `data/` são o **artefato canônico** do repositório — são exatamente os dados usados no artigo. O script `src/00_generate_dataset.py` documenta a metodologia de geração (distribuições e regras de domínio calibradas ao IIA Framework) e permite gerar novos conjuntos sintéticos da mesma natureza:

```bash
python src/00_generate_dataset.py   # regenera data/*.csv
```

> **Nota:** regenerar os dados produz um conjunto estatisticamente equivalente, mas não idêntico byte-a-byte aos CSVs versionados. Para reproduzir exatamente os resultados publicados no artigo, use os CSVs já presentes em `data/`.

### Caminhos configuráveis

Os diretórios de dados e saída podem ser redefinidos por variáveis de ambiente, sem editar código:

```bash
export AUDITORES_PROJECT_DATA=/caminho/para/dados
export AUDITORES_PROJECT_OUTPUT=/caminho/para/saida
python run_all.py
```

## Resultados

| Modelo | R² (teste) | CV R² | MAE | RMSE |
|---|---|---|---|---|
| Random Forest | 0,8902 | 0,8861 | 0,0515 | 0,0661 |
| XGBoost | 0,9673 | 0,9684 | 0,0287 | 0,0361 |
| LightGBM | 0,9662 | 0,9659 | 0,0288 | 0,0367 |
| **CatBoost** | **0,9737** | **0,9719** | **0,0256** | **0,0324** |
| Extra Trees | 0,9533 | 0,9576 | 0,0322 | 0,0431 |

O **CatBoost** obteve o melhor desempenho. As heurísticas de alocação seguem o padrão: **Greedy** vence sob abundância de recursos, **Húngaro** sob escassez.

## Como aplicar a dados reais

O framework é diretamente aplicável a dados de uma firma real: basta substituir os CSVs em `data/` mantendo a mesma estrutura de colunas. O `fit_score` pode ser recalibrado ajustando os pesos em `src/config.py` (`FIT_WEIGHTS`).

## Citação

```bibtex
@article{motta_rego_2026_auditor_matching,
  title  = {Modelos de Aprendizado de Máquina Ensemble para Predição de
            Compatibilidade entre Auditores e Projetos},
  author = {Motta, Gustavo Souza da and Rego, Vinicius},
  year   = {2026}
}
```

## Licença

MIT — veja [LICENSE](LICENSE).
