"""
02_heuristics.py — Matriz de scores e heurísticas de alocação.

Constrói a matriz de scores a partir do fit_score e compara as heurísticas
Greedy (ótimo local) e Húngaro (ótimo global) em diferentes razões entre
número de auditores disponíveis e número de projetos.

Referências:
    Algoritmo Húngaro — Kuhn (1955), The Hungarian method for the
        assignment problem, Naval Research Logistics Quarterly, 2, 83-97.
    Implementação via scipy.optimize.linear_sum_assignment.

Uso:
    python src/02_heuristics.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config as C
from fit_score import fit_score, build_score_matrix


def alocar_greedy(M: np.ndarray) -> dict:
    """Para cada projeto (coluna), aloca o melhor auditor ainda disponível."""
    alocados, resultado = set(), {}
    for j in range(M.shape[1]):
        col = M[:, j].copy()
        if alocados:
            col[list(alocados)] = -1.0
        best = int(np.argmax(col))
        resultado[j] = (best, float(M[best, j]))
        alocados.add(best)
    return resultado


def alocar_hungaro(M: np.ndarray) -> dict:
    """Maximiza a soma total de scores (resolve via minimização de -M)."""
    n = min(M.shape)
    sub = M[:n, :n]
    rows, cols = linear_sum_assignment(-sub)
    return {int(c): (int(r), float(sub[r, c])) for r, c in zip(rows, cols)}


def soma(aloc: dict) -> float:
    return sum(v[1] for v in aloc.values())


def main() -> None:
    C.require_data()
    auditores = pd.read_csv(C.AUDITORES_CSV)
    projetos  = pd.read_csv(C.PROJETOS_CSV)

    # ── Demonstração 5×5 (usada na Figura 4 do artigo)
    aud5, proj5 = auditores.head(5), projetos.head(5)
    M5 = build_score_matrix(aud5, proj5)
    np.save(C.MATRIZ_NPY, M5)

    g, h = alocar_greedy(M5), alocar_hungaro(M5)
    print("=== Matriz 5×5 ===")
    print(pd.DataFrame(M5,
          index=aud5["auditor_id"].values,
          columns=proj5["projeto_id"].values).to_string())
    print(f"\nGreedy  — soma: {soma(g):.4f}")
    print(f"Húngaro — soma: {soma(h):.4f}")

    # ── Análise de sensibilidade: abundância vs. escassez
    print("\n=== Greedy vs. Húngaro por razão auditor/projeto ===")
    print(f"{'N_aud':>6}{'N_proj':>8}{'Greedy':>12}{'Hungaro':>12}{'Vencedor':>12}")
    cenarios = [(200, 5), (200, 10), (60, 50), (55, 50), (50, 50)]
    for n_aud, n_proj in cenarios:
        aus = auditores.sample(n_aud, random_state=C.RANDOM_STATE).reset_index(drop=True)
        ps  = projetos.sample(n_proj, random_state=C.RANDOM_STATE).reset_index(drop=True)
        M = build_score_matrix(aus, ps)
        sg = soma(alocar_greedy(M))
        sh = soma(alocar_hungaro(M))
        venc = "Húngaro" if sh > sg + 1e-9 else ("Empate" if abs(sh - sg) < 1e-9 else "Greedy")
        print(f"{n_aud:>6}{n_proj:>8}{sg:>12.4f}{sh:>12.4f}{venc:>12}")

    print(f"\nMatriz 5×5 salva em {C.MATRIZ_NPY.relative_to(C.ROOT_DIR)}")


if __name__ == "__main__":
    main()
