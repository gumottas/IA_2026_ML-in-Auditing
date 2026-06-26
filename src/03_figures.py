"""
03_figures.py — Gera as figuras do artigo a partir dos resultados reais.

Lê outputs/ml_resultados.pkl (gerado por 01_train_models.py) e
outputs/matriz_5x5.npy (gerado por 02_heuristics.py) e produz:
    - fig2_metricas.png        : comparação de R² e MAE dos modelos
    - fig3_features_pred.png   : importância de features + previsto vs. real
    - fig4_matriz.png          : matriz de scores com Greedy e Húngaro

Uso:
    python src/03_figures.py
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment

import config as C

# ── Paleta (journal de gestão, fundo claro)
BG, C1, C2, C3, C4 = "#FFFFFF", "#2563EB", "#DC2626", "#16A34A", "#9333EA"
TEXT, MUTED, RULE = "#1E293B", "#64748B", "#CBD5E1"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": RULE,
    "axes.labelcolor": TEXT, "axes.titlecolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": RULE, "text.color": TEXT,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Extra Trees"]


def load_results() -> dict:
    if not C.RESULTS_PKL.exists():
        raise FileNotFoundError(
            f"{C.RESULTS_PKL} não encontrado. Rode 01_train_models.py primeiro.")
    with open(C.RESULTS_PKL, "rb") as f:
        return pickle.load(f)


def fig_metricas(R: dict) -> None:
    res, melhor = R["resultados"], R["melhor"]
    labels = [m.replace(" ", "\n") for m in ORDER]
    r2s = [res[m]["r2"] for m in ORDER]
    cvs = [res[m]["cv_mean"] for m in ORDER]
    stds = [res[m]["cv_std"] for m in ORDER]
    maes = [res[m]["mae"] for m in ORDER]
    cores = [MUTED, C1, C3, C2, C4]
    best_i = ORDER.index(melhor)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Figura 2 — Desempenho comparativo dos cinco modelos ensemble",
                 fontsize=13, fontweight="bold")

    ax = axes[0]; x = np.arange(len(ORDER)); w = 0.38
    ax.bar(x - w/2, r2s, w, label="R² (teste)", color=cores, alpha=0.9)
    ax.bar(x + w/2, cvs, w, label="CV R² (5-fold)", color=cores, alpha=0.45,
           yerr=stds, capsize=4, error_kw={"color": MUTED, "lw": 1.2})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel("R²")
    ax.set_title("R² — teste e validação cruzada", fontsize=11)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y", alpha=0.4)
    for i, v in enumerate(r2s):
        ax.text(i - w/2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.get_xticklabels()[best_i].set_fontweight("bold")
    ax.get_xticklabels()[best_i].set_color(C2)

    ax = axes[1]
    bars = ax.bar(labels, maes, color=cores, alpha=0.9)
    ax.set_ylabel("MAE"); ax.set_title("MAE — erro médio absoluto", fontsize=11)
    ax.set_ylim(0, max(maes) * 1.25); ax.grid(axis="y", alpha=0.4)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width()/2, v + max(maes)*0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.get_xticklabels()[best_i].set_fontweight("bold")
    ax.get_xticklabels()[best_i].set_color(C2)

    plt.tight_layout()
    out = C.FIGURES_DIR / "fig2_metricas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"✓ {out.relative_to(C.ROOT_DIR)}")


def fig_features_pred(R: dict) -> None:
    res, melhor, fi = R["resultados"], R["melhor"], R["fi"]
    y_test, y_pred = R["y_test"], R["y_pred"]
    labels_fi = {
        "skills_match_pct": "Match de skills (%)", "cert_match": "Match de certificações",
        "setor_match": "Match de setor", "nivel_match": "Compatib. de nível",
        "skills_overlap": "Skills em comum", "nivel_gap": "Gap de nível",
        "ingles": "Inglês (auditor)", "disponibilidade_pct": "Disponibilidade",
        "anos_experiencia": "Anos de experiência", "ingles_requerido": "Inglês (requerido)",
    }
    top = fi.head(10).copy()
    top["lbl"] = top["feature"].map(lambda x: labels_fi.get(x, x))
    cores = [C1 if any(k in f for k in ["match", "overlap", "gap"]) else C3
             for f in top["feature"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle(f"Figura 3 — Importância de features e qualidade preditiva ({melhor})",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.barh(top["lbl"][::-1], top["importance"][::-1], color=cores[::-1], alpha=0.88)
    ax.set_xlabel("Importância relativa"); ax.set_title("Top 10 features", fontsize=11)
    ax.grid(axis="x", alpha=0.4)
    ax.legend(handles=[mpatches.Patch(color=C1, label="Dimensões de compatibilidade"),
                       mpatches.Patch(color=C3, label="Atributos do auditor")],
              fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.scatter(y_test, y_pred, alpha=0.3, s=14, color=C1, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 0.03, max(y_test.max(), y_pred.max()) + 0.03]
    ax.plot(lims, lims, "--", color=C2, lw=1.8, label="Linha ideal (y=x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Fit Score Real"); ax.set_ylabel("Fit Score Previsto")
    ax.set_title("Previsto vs. Real — teste", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.05, 0.93, f"R² = {res[melhor]['r2']:.4f}", transform=ax.transAxes,
            fontsize=11, color=C2, fontweight="bold")
    ax.text(0.05, 0.86, f"MAE = {res[melhor]['mae']:.4f}", transform=ax.transAxes,
            fontsize=11, color=C1)

    plt.tight_layout()
    out = C.FIGURES_DIR / "fig3_features_pred.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"✓ {out.relative_to(C.ROOT_DIR)}")


def fig_matriz() -> None:
    if not C.MATRIZ_NPY.exists():
        raise FileNotFoundError(
            f"{C.MATRIZ_NPY} não encontrado. Rode 02_heuristics.py primeiro.")
    M = np.load(C.MATRIZ_NPY)
    aud_labels = ["AUD-0001\n(Supervisor)", "AUD-0002\n(Júnior)", "AUD-0003\n(Júnior)",
                  "AUD-0004\n(Júnior)", "AUD-0005\n(Sênior)"]
    proj_labels = [f"PRJ-000{i+1}" for i in range(M.shape[1])]

    def greedy(M):
        al, res = set(), {}
        for j in range(M.shape[1]):
            c = M[:, j].copy()
            if al: c[list(al)] = -1
            b = int(np.argmax(c)); res[j] = b; al.add(b)
        return res

    def hungaro(M):
        r, c = linear_sum_assignment(-M)
        return {int(cc): int(rr) for rr, cc in zip(r, c)}

    g, h = greedy(M), hungaro(M)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)
    fig.suptitle("Figura 4 — Matriz de scores (5×5) e alocações por heurística",
                 fontsize=13, fontweight="bold")
    for ax, aloc, titulo in [
        (axes[0], g, "Greedy (ótimo local)"),
        (axes[1], h, "Húngaro (ótimo global)"),
    ]:
        soma = sum(M[aloc[j], j] for j in range(M.shape[1]))
        ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=0.7)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(proj_labels, fontsize=8.5)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(aud_labels, fontsize=8)
        ax.set_title(f"{titulo}  —  soma = {soma:.3f}", fontsize=11)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                sel = (aloc[j] == i)
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.45 else TEXT,
                        fontweight="bold" if sel else "normal")
                if sel:
                    ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                 fill=False, edgecolor=C2, linewidth=2.5))
    plt.tight_layout()
    out = C.FIGURES_DIR / "fig4_matriz.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"✓ {out.relative_to(C.ROOT_DIR)}")


def main() -> None:
    R = load_results()
    fig_metricas(R)
    fig_features_pred(R)
    fig_matriz()
    print("\nFiguras geradas em", C.FIGURES_DIR.relative_to(C.ROOT_DIR))


if __name__ == "__main__":
    main()
