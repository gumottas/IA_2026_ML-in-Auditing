"""
fit_score.py — Cálculo do score de compatibilidade auditor×projeto.

Implementa a fórmula ponderada descrita na Seção 3.2 do artigo. É usado
tanto para reconstruir a matriz de scores quanto como referência para
o alvo que os modelos de ML aprendem a prever.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from config import FIT_WEIGHTS


def _to_set(value: str) -> set:
    """Converte uma célula JSON (lista) em conjunto Python."""
    return set(json.loads(value))


def fit_score(aud: pd.Series, proj: pd.Series) -> float:
    """
    Calcula o fit score entre um auditor e um projeto.

    Componentes (todos em [0, 1]):
      - skills:          fração das skills requeridas que o auditor possui
      - nivel:           compatibilidade hierárquica (penalidade assimétrica)
      - setor:           1 se o auditor tem experiência no setor do cliente
      - cert:            fração das certificações requeridas que o auditor possui
      - idioma:          1 se atende ao requisito de inglês do projeto
      - disponibilidade: percentual de disponibilidade do auditor

    Retorna um valor arredondado em [0, 1].
    """
    skills_aud = _to_set(aud["skills"])
    skills_req = _to_set(proj["skills_requeridas"])
    certs_aud  = _to_set(aud["certificacoes"])
    certs_req  = _to_set(proj["certs_requeridas"])
    setores    = _to_set(aud["setores_experiencia"])

    skills_match = len(skills_aud & skills_req) / max(len(skills_req), 1)

    gap = aud["nivel_num"] - proj["nivel_min_num"]
    if gap < 0:                       # abaixo do mínimo: penalidade forte
        nivel_match = max(0.0, 1.0 + gap * 0.4)
    else:                             # acima do mínimo: desconto leve
        nivel_match = min(1.0, 1.0 - gap * 0.05)

    setor_match = 1.0 if proj["setor_cliente"] in setores else 0.0
    cert_match  = len(certs_aud & certs_req) / max(len(certs_req), 1)
    idioma_match = float(aud["ingles"]) if proj["ingles_requerido"] else 1.0
    disponibilidade = aud["disponibilidade_pct"] / 100.0

    score = (
        FIT_WEIGHTS["skills"]          * skills_match +
        FIT_WEIGHTS["nivel"]           * nivel_match +
        FIT_WEIGHTS["setor"]           * setor_match +
        FIT_WEIGHTS["cert"]            * cert_match +
        FIT_WEIGHTS["idioma"]          * idioma_match +
        FIT_WEIGHTS["disponibilidade"] * disponibilidade
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def build_score_matrix(auditores: pd.DataFrame, projetos: pd.DataFrame) -> np.ndarray:
    """Constrói a matriz de scores (linhas = auditores, colunas = projetos)."""
    matrix = np.zeros((len(auditores), len(projetos)))
    for j, (_, proj) in enumerate(projetos.iterrows()):
        for i, (_, aud) in enumerate(auditores.iterrows()):
            matrix[i, j] = fit_score(aud, proj)
    return matrix
