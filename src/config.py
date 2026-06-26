"""
config.py — Configuração central do projeto.

Todos os caminhos são definidos em relação à raiz do repositório,
detectada automaticamente. Nenhum caminho absoluto é fixado no código,
o que torna o projeto 100% portável: basta clonar e rodar.

Também é possível sobrescrever os diretórios via variáveis de ambiente:
    AUDITORES_PROJECT_DATA   -> diretório dos CSVs de entrada
    AUDITORES_PROJECT_OUTPUT -> diretório de saída (figuras, modelos)
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Raiz do projeto: a pasta que contém este arquivo (src/) é src,
#    então a raiz é o diretório-pai de src/.
SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

# ── Diretórios principais (sobrescrevíveis por variável de ambiente)
DATA_DIR    = Path(os.environ.get("AUDITORES_PROJECT_DATA",   ROOT_DIR / "data"))
OUTPUT_DIR  = Path(os.environ.get("AUDITORES_PROJECT_OUTPUT", ROOT_DIR / "outputs"))
FIGURES_DIR = OUTPUT_DIR / "figures"

# ── Garante que os diretórios de saída existam
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Arquivos de dados de entrada
AUDITORES_CSV = DATA_DIR / "auditores.csv"
PROJETOS_CSV  = DATA_DIR / "projetos.csv"
ALOCACOES_CSV = DATA_DIR / "alocacoes.csv"

# ── Artefatos de saída
RESULTS_PKL   = OUTPUT_DIR / "ml_resultados.pkl"
MATRIZ_NPY    = OUTPUT_DIR / "matriz_5x5.npy"

# ── Reprodutibilidade
RANDOM_STATE = 42

# ── Hiperparâmetros compartilhados (centralizados para fácil ajuste)
N_ESTIMATORS  = 300
LEARNING_RATE = 0.05
TEST_SIZE     = 0.20
CV_FOLDS      = 5

# ── Definição da variável-alvo e colunas a excluir das features
TARGET          = "fit_score"
ID_COLUMNS      = ["alocacao_id", "auditor_id", "projeto_id"]

# ── Pesos do fit score (Seção 3.2 do artigo)
FIT_WEIGHTS = {
    "skills":         0.40,
    "nivel":          0.20,
    "setor":          0.15,
    "cert":           0.15,
    "idioma":         0.05,
    "disponibilidade":0.05,
}


def require_data() -> None:
    """Verifica se os CSVs de entrada existem; orienta caso faltem."""
    faltando = [p.name for p in (AUDITORES_CSV, PROJETOS_CSV, ALOCACOES_CSV) if not p.exists()]
    if faltando:
        raise FileNotFoundError(
            f"Arquivos de dados não encontrados em {DATA_DIR}: {', '.join(faltando)}.\n"
            f"Coloque os CSVs em {DATA_DIR} ou defina AUDITORES_PROJECT_DATA."
        )


if __name__ == "__main__":
    print("Raiz do projeto :", ROOT_DIR)
    print("Dados           :", DATA_DIR)
    print("Saída           :", OUTPUT_DIR)
    print("Figuras         :", FIGURES_DIR)
    print("random_state    :", RANDOM_STATE)
