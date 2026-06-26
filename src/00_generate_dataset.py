"""
00_generate_dataset.py — Geração dos datasets sintéticos.

Reconstrói os três CSVs (auditores, projetos, alocações) a partir de
distribuições estatísticas calibradas ao IIA Internal Auditing Competency
Framework. Com random_state fixo, a geração é determinística.

Uso:
    python src/00_generate_dataset.py

IMPORTANTE: os CSVs distribuídos no repositório foram gerados por este
script. Caso queira regenerá-los do zero, execute-o; o resultado será
idêntico desde que a semente (config.RANDOM_STATE) não seja alterada.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import config as C

# ── Domínio (Seção 3.1 do artigo) ────────────────────────────────────
NIVEIS = ["Junior", "Senior", "Supervisor", "Gerente", "Socio"]
NIVEL_PESOS = [0.25, 0.30, 0.20, 0.15, 0.10]
NIVEL_ANOS_MEAN = [1, 4, 8, 14, 22]
NIVEL_PERF_MEAN = [3.0, 3.4, 3.7, 4.1, 4.5]

SKILLS = [
    "IFRS", "USGAAP", "BRGAAP", "SOX", "COSO", "CISA", "ISO27001",
    "Auditoria_Financeira", "Auditoria_Operacional", "Auditoria_TI",
    "Auditoria_Fiscal", "Auditoria_Compliance", "Forensics_Antilavagem",
    "ESG_Sustentabilidade", "Gestao_Riscos",
]
CERTIFICACOES = ["CPA", "CIA", "CISA", "CFE", "CRC", "ACCA", "CFA"]
SETORES = [
    "Financeiro_Bancario", "Seguros", "Energia_Utilities", "Saude_Farma",
    "Varejo_Consumo", "Industria_Manufacturing", "Tecnologia",
    "Governo_PublicSector", "Agronegocio", "Telecomunicacoes",
]
TIPOS_PROJETO = [
    "Auditoria_Financeira_Anual", "Due_Diligence", "Auditoria_TI_Sistemas",
    "Forensics_Investigacao", "Auditoria_Compliance_Regulatoria", "ESG_Assurance",
    "IPO_Readiness", "Revisao_Controles_Internos", "Auditoria_Fiscal_Tributaria",
    "Auditoria_Interna_Operacional",
]
COMPLEXIDADES = ["Baixa", "Media", "Alta", "Critica"]

CORE_SKILLS = {
    "Auditoria_Financeira_Anual":      ["Auditoria_Financeira", "IFRS"],
    "Due_Diligence":                   ["Auditoria_Financeira", "Gestao_Riscos"],
    "Auditoria_TI_Sistemas":           ["Auditoria_TI", "CISA", "ISO27001"],
    "Forensics_Investigacao":          ["Forensics_Antilavagem", "Auditoria_Financeira"],
    "Auditoria_Compliance_Regulatoria":["Auditoria_Compliance", "Gestao_Riscos"],
    "ESG_Assurance":                   ["ESG_Sustentabilidade", "Auditoria_Operacional"],
    "IPO_Readiness":                   ["Auditoria_Financeira", "IFRS", "SOX"],
    "Revisao_Controles_Internos":      ["COSO", "SOX"],
    "Auditoria_Fiscal_Tributaria":     ["Auditoria_Fiscal", "BRGAAP"],
    "Auditoria_Interna_Operacional":   ["Auditoria_Operacional", "COSO"],
}

HOJE = datetime(2026, 5, 10)


def gerar_auditores(n: int) -> pd.DataFrame:
    linhas = []
    for i in range(n):
        nivel_idx = np.random.choice(len(NIVEIS), p=NIVEL_PESOS)
        nivel = NIVEIS[nivel_idx]
        anos = max(0, int(np.random.normal(NIVEL_ANOS_MEAN[nivel_idx], 1.5)))

        n_sk = max(1, int(np.random.normal(3.5 + nivel_idx * 2, 1.0)))
        n_sk = min(n_sk, len(SKILLS))
        skills = sorted(np.random.choice(SKILLS, size=n_sk, replace=False).tolist())

        n_ce = min(max(0, int(np.random.normal(nivel_idx, 0.6))), len(CERTIFICACOES))
        certs = sorted(np.random.choice(CERTIFICACOES, size=n_ce, replace=False).tolist())

        n_se = min(max(1, int(np.random.normal(2 + nivel_idx * 0.3, 0.8))), 4)
        setores = sorted(np.random.choice(SETORES, size=n_se, replace=False).tolist())

        disp = round(float(np.random.beta(2, 2) * 100), 1)
        ingles = int(np.random.random() < 0.70)
        espanhol = int(np.random.random() < 0.33)
        perf = round(float(np.clip(np.random.normal(NIVEL_PERF_MEAN[nivel_idx], 0.3), 1, 5)), 2)
        data_disp = (HOJE + timedelta(days=int(np.random.exponential(30)))).strftime("%Y-%m-%d")

        linha = {
            "auditor_id": f"AUD-{i+1:04d}", "nivel": nivel, "nivel_num": nivel_idx,
            "anos_experiencia": anos, "skills": json.dumps(skills), "n_skills": len(skills),
            "setores_experiencia": json.dumps(setores), "n_setores_exp": len(setores),
            "certificacoes": json.dumps(certs), "n_certificacoes": len(certs),
            "disponibilidade_pct": disp, "ingles": ingles, "espanhol": espanhol,
            "performance_media": perf, "data_disponivel": data_disp,
        }
        for s in SKILLS:        linha[f"skill_{s}"] = int(s in skills)
        for c in CERTIFICACOES: linha[f"cert_{c}"] = int(c in certs)
        for se in SETORES:      linha[f"setor_exp_{se}"] = int(se in setores)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def gerar_projetos(n: int) -> pd.DataFrame:
    linhas = []
    for i in range(n):
        tipo = np.random.choice(TIPOS_PROJETO)
        compl_idx = np.random.choice(len(COMPLEXIDADES), p=[0.30, 0.35, 0.25, 0.10])
        compl = COMPLEXIDADES[compl_idx]
        setor = np.random.choice(SETORES)
        nivel_min_idx = min(4, max(0, compl_idx + np.random.choice([0, 1])))

        core = CORE_SKILLS.get(tipo, [])
        outros = [s for s in SKILLS if s not in core]
        n_extra = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        extras = np.random.choice(outros, size=n_extra, replace=False).tolist() if n_extra else []
        skills_req = sorted(set(core + extras))

        n_ce = [0, 1, 2, 3][compl_idx]
        certs_req = sorted(np.random.choice(CERTIFICACOES, size=n_ce, replace=False).tolist()) if n_ce else []

        duracao = int(np.random.lognormal(2.5 + compl_idx * 0.3, 0.4))
        duracao = max(3, duracao)
        valor = round(duracao * np.random.uniform(1000, 3000), 2)
        data_ini = (HOJE + timedelta(days=int(np.random.uniform(3, 180)))).strftime("%Y-%m-%d")
        ingles_req = int(setor in ["Financeiro_Bancario", "Tecnologia"] or compl_idx >= 2)
        tamanho = [2, 3, 5, 8][compl_idx]

        linha = {
            "projeto_id": f"PRJ-{i+1:04d}", "tipo_projeto": tipo, "complexidade": compl,
            "complexidade_num": compl_idx, "setor_cliente": setor,
            "nivel_min_requerido": NIVEIS[nivel_min_idx], "nivel_min_num": nivel_min_idx,
            "skills_requeridas": json.dumps(skills_req), "n_skills_requeridas": len(skills_req),
            "certs_requeridas": json.dumps(certs_req), "n_certs_requeridas": len(certs_req),
            "duracao_dias": duracao, "valor_honorarios": valor,
            "data_inicio_prevista": data_ini, "ingles_requerido": ingles_req,
            "tamanho_equipe": tamanho,
        }
        for s in SKILLS:        linha[f"req_skill_{s}"] = int(s in skills_req)
        for c in CERTIFICACOES: linha[f"req_cert_{c}"] = int(c in certs_req)
        for se in SETORES:      linha[f"req_setor_{se}"] = int(se == setor)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def gerar_alocacoes(auditores: pd.DataFrame, projetos: pd.DataFrame, n: int) -> pd.DataFrame:
    from fit_score import fit_score
    linhas = []
    for i in range(n):
        aud = auditores.sample(1).iloc[0]
        proj = projetos.sample(1).iloc[0]

        skills_aud = set(json.loads(aud["skills"]))
        skills_req = set(json.loads(proj["skills_requeridas"]))
        certs_aud  = set(json.loads(aud["certificacoes"]))
        certs_req  = set(json.loads(proj["certs_requeridas"]))
        setores    = set(json.loads(aud["setores_experiencia"]))

        nivel_gap = int(aud["nivel_num"] - proj["nivel_min_num"])
        skills_overlap = len(skills_aud & skills_req)
        skills_match_pct = round(skills_overlap / max(len(skills_req), 1), 4)
        gap = aud["nivel_num"] - proj["nivel_min_num"]
        nivel_match = round(max(0.0, 1.0 + gap * 0.4) if gap < 0 else min(1.0, 1.0 - gap * 0.05), 4)
        setor_match = 1.0 if proj["setor_cliente"] in setores else 0.0
        cert_match = round(len(certs_aud & certs_req) / max(len(certs_req), 1), 4)

        score = fit_score(aud, proj)
        score = round(float(np.clip(score + np.random.normal(0, 0.03), 0, 1)), 4)

        linha = {
            "alocacao_id": f"ALOC-{i+1:05d}",
            "auditor_id": aud["auditor_id"], "projeto_id": proj["projeto_id"],
            "nivel_num": aud["nivel_num"], "anos_experiencia": aud["anos_experiencia"],
            "n_skills_auditor": aud["n_skills"], "n_setores_exp": aud["n_setores_exp"],
            "n_certificacoes": aud["n_certificacoes"], "disponibilidade_pct": aud["disponibilidade_pct"],
            "ingles": aud["ingles"], "espanhol": aud["espanhol"], "performance_media": aud["performance_media"],
            "complexidade_num": proj["complexidade_num"], "nivel_min_num": proj["nivel_min_num"],
            "n_skills_requeridas": proj["n_skills_requeridas"], "n_certs_requeridas": proj["n_certs_requeridas"],
            "duracao_dias": proj["duracao_dias"], "ingles_requerido": proj["ingles_requerido"],
            "nivel_gap": nivel_gap, "skills_overlap": skills_overlap,
            "skills_match_pct": skills_match_pct, "nivel_match": nivel_match,
            "setor_match": setor_match, "cert_match": cert_match, "fit_score": score,
        }
        for s in SKILLS:
            a, r = int(s in skills_aud), int(s in skills_req)
            linha[f"aud_skill_{s}"] = a; linha[f"req_skill_{s}"] = r; linha[f"match_skill_{s}"] = int(a and r)
        for c in CERTIFICACOES:
            linha[f"aud_cert_{c}"] = int(c in certs_aud); linha[f"req_cert_{c}"] = int(c in certs_req)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def main() -> None:
    np.random.seed(C.RANDOM_STATE)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Gerando auditores...")
    auditores = gerar_auditores(200)
    auditores.to_csv(C.AUDITORES_CSV, index=False)

    print("Gerando projetos...")
    projetos = gerar_projetos(300)
    projetos.to_csv(C.PROJETOS_CSV, index=False)

    print("Gerando alocações...")
    alocacoes = gerar_alocacoes(auditores, projetos, 1500)
    alocacoes.to_csv(C.ALOCACOES_CSV, index=False)

    print(f"\n✓ {len(auditores)} auditores, {len(projetos)} projetos, "
          f"{len(alocacoes)} alocações gerados em {C.DATA_DIR.relative_to(C.ROOT_DIR)}")


if __name__ == "__main__":
    main()
