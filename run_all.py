"""
run_all.py — Executa o pipeline completo de ponta a ponta.

Roda, na ordem correta:
    1. Treino e comparação dos modelos
    2. Heurísticas de alocação
    3. Geração das figuras

Uso (a partir da raiz do repositório):
    python run_all.py
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

ETAPAS = [
    ("Treinando e comparando modelos",   "01_train_models.py"),
    ("Heurísticas de alocação",          "02_heuristics.py"),
    ("Gerando figuras",                  "03_figures.py"),
]

for titulo, script in ETAPAS:
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)
    runpy.run_path(str(SRC / script), run_name="__main__")

print("\n" + "=" * 60)
print("  Pipeline completo finalizado com sucesso.")
print("=" * 60)
