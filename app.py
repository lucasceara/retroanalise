# ============================================================
# GMM — Pavimentos Aeroportuários v2
# Suporte a 3 camadas (Rev + Base + Subleito)
#         e 4 camadas (Rev + Base + Sub-base + Subleito)
# Detecção automática pelo número de colunas do arquivo
# ============================================================

# ── CÉLULA 1: Instalar dependências ─────────────────────────
!pip install scikit-learn scipy matplotlib pandas openpyxl -q
print("✓ Dependências instaladas.")


# ============================================================
# ── CÉLULA 2: Funções do pipeline ───────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, norm
import io
import warnings
warnings.filterwarnings("ignore")

CORES = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def detectar_camadas(df_raw):
    """
    Detecta automaticamente o número de camadas pelo total de colunas.
    4 colunas → 3 camadas (Rev, Base, Subleito) + RMSE
    5 colunas → 4 camadas (Rev, Base, Sub-base, Subleito) + RMSE
    """
    n = df_raw.shape[1]
    if n == 4:
        return ["Revestimento", "Base", "Subleito"]
    elif n == 5:
        return ["Revestimento", "Base", "Sub-base", "Subleito"]
    else:
        raise ValueError(
            f"Arquivo com {n} colunas não suportado. "
            "Esperado: 4 colunas (3 camadas + RMSE) ou "
            "5 colunas (4 camadas + RMSE)."
        )


def carregar_arquivo(caminho_ou_buffer, nome_arquivo=""):
    """Carrega CSV ou XLSX, detecta camadas, padroniza colunas."""
    nome = nome_arquivo.lower()
    if nome.endswith(".xlsx"):
        df = pd.read_excel(caminho_ou_buffer)
    else:
        try:
            df = pd.read_csv(caminho_ou_buffer, sep=";")
            if df.shape[1] < 4:
                if hasattr(caminho_ou_buffer, "seek"):
                    caminho_ou_buffer.seek(0)
                df = pd.read_csv(caminho_ou_buffer, sep=",")
        except Exception:
            if hasattr(caminho_ou_buffer, "seek"):
                caminho_ou_buffer.seek(0)
            df = pd.read_csv(caminho_ou_buffer, sep=",")

    df.columns = [c.strip() for c in df.columns]
    camadas = detectar_camadas(df)
    df = df.iloc[:, :len(camadas) + 1]
    df.columns = camadas + ["RMSE"]
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df, camadas


def rodar_pipeline(df, camadas, percentil=15):
    """Pipeline GMM completo. Funciona com 3 ou 4 camadas."""

    # Pesos pelo RMSE
    pesos = 1.0 / df["RMSE"].values
    pesos = pesos / pesos.sum()

    # Reamostragem ponderada por Monte Carlo
    np.random.seed(42)
    idx_resamp = np.random.choice(len(df), size=500, replace=True, p=pesos)

    # Padronização Z-score
    X = df[camadas].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_resamp = X_scaled[idx_resamp]

    # Seleção de K via BIC
    bic, aic = [], []
    for k in range(1, 6):
        g = GaussianMixture(n_components=k, covariance_type="diag",
                            random_state=42, n_init=10)
        g.fit(X_resamp)
        bic.append(g.bic(X_scaled))   # BIC nos dados originais (n real)
        aic.append(g.aic(X_scaled))   # penalização adequada ao tamanho da amostra
    k_otimo = int(np.argmin(bic)) + 1

    # GMM final
    gmm = GaussianMixture(n_components=k_otimo, covariance_type="diag",
                          random_state=42, n_init=20)
    gmm.fit(X_resamp)
    labels = gmm.predict(X_scaled)

    # Parâmetros na escala original
    means = scaler.inverse_transform(gmm.means_)
    stds  = np.zeros_like(means)
    for k in range(k_otimo):
        stds[k] = np.sqrt(gmm.covariances_[k]) * scaler.scale_

    # Componente dominante — sempre pelo subleito
    idx_sub  = camadas.index("Subleito")
    comp_dom = int(np.argmin(means[:, idx_sub]))

    # z pelo percentil
    z = norm.ppf(1 - percentil / 100)

    # E_rep por camada
    resultados = {}
    for i, cam in enumerate(camadas):
        mu    = means[comp_dom, i]
        sigma = stds[comp_dom, i]
        cv    = sigma / mu
        E_rep = max(mu - z * sigma, 0)
        resultados[cam] = {
            "mu": mu, "sigma": sigma, "cv": cv,
            "E_rep": E_rep, "percentil": percentil
        }

    return {
        "df": df, "pesos": pesos, "camadas": camadas,
        "k_otimo": k_otimo, "bic": bic, "aic": aic,
        "gmm": gmm, "labels": labels,
        "means": means, "stds": stds,
        "comp_dom": comp_dom, "resultados": resultados,
        "z": z, "percentil": percentil
    }


def plotar_segmento(res, nome_segmento):
    """Gera e exibe figura adaptada para 3 ou 4 camadas."""
    df         = res["df"]
    camadas    = res["camadas"]
    n_camadas  = len(camadas)
    k_otimo    = res["k_otimo"]
    gmm        = res["gmm"]
    means      = res["means"]
    stds       = res["stds"]
    labels     = res["labels"]
    pesos      = res["pesos"]
    resultados = res["resultados"]
    bic        = res["bic"]
    aic        = res["aic"]
    comp_dom   = res["comp_dom"]
    pesos_comp = gmm.weights_

    n_cols = n_camadas + 1
    fig = plt.figure(figsize=(5 * n_cols, 12))
    fig.suptitle(
        f"Segmento: {nome_segmento}  |  {n_camadas} camadas  |  "
        f"K = {k_otimo}  |  P{res['percentil']}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.48, wspace=0.38)

    # Distribuições por camada
    for i, cam in enumerate(camadas):
        ax    = fig.add_subplot(gs[0, i])
        dados = df[cam].values
        kde   = gaussian_kde(dados, weights=pesos, bw_method="silverman")
        x_r   = np.linspace(dados.min() * 0.85, dados.max() * 1.10, 500)
        ax.plot(x_r, kde(x_r), "k-", lw=2, label="KDE ponderado", zorder=5)
        for k in range(k_otimo):
            y_k = pesos_comp[k] * norm.pdf(x_r, means[k, i], stds[k, i])
            lw  = 2.5 if k == comp_dom else 1.2
            ax.fill_between(x_r, y_k, alpha=0.25, color=CORES[k])
            ax.plot(x_r, y_k, color=CORES[k], lw=lw,
                    label=f"C{k+1}{'★' if k==comp_dom else ''} "
                          f"(w={pesos_comp[k]:.2f})")
        r = resultados[cam]
        ax.axvline(r["E_rep"], color="red",  lw=2,   ls="--",
                   label=f"E_rep={r['E_rep']:.0f}")
        ax.axvline(r["mu"],   color="navy", lw=1.5, ls=":",
                   label=f"μ={r['mu']:.0f}")
        ax.set_title(cam, fontweight="bold")
        ax.set_xlabel("Módulo (MPa)")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=6.5)

    # BIC/AIC
    ax_bic = fig.add_subplot(gs[0, n_camadas])
    ax_bic.plot(range(1, 6), bic, "bo-",  lw=2, label="BIC")
    ax_bic.plot(range(1, 6), aic, "rs--", lw=2, label="AIC")
    ax_bic.axvline(k_otimo, color="green", lw=2, ls=":", label=f"K={k_otimo}")
    ax_bic.set_xlabel("K")
    ax_bic.set_ylabel("Score")
    ax_bic.set_title("Seleção de K — BIC/AIC", fontweight="bold")
    ax_bic.legend()

    # Scatters: pares de camadas adjacentes
    pares = [(camadas[i], camadas[i + 1]) for i in range(n_camadas - 1)]
    for p_idx, (cam_x, cam_y) in enumerate(pares):
        ax_s = fig.add_subplot(gs[1, p_idx])
        for k in range(k_otimo):
            m = labels == k
            ax_s.scatter(df.loc[m, cam_x], df.loc[m, cam_y],
                         c=CORES[k], label=f"C{k+1}", s=60,
                         alpha=0.8, edgecolors="k", lw=0.5)
        ax_s.set_xlabel(f"{cam_x} (MPa)")
        ax_s.set_ylabel(f"{cam_y} (MPa)")
        ax_s.set_title(f"{cam_x} vs {cam_y}", fontweight="bold")
        ax_s.legend(fontsize=7)

    # Subleito × RMSE
    ax_rmse = fig.add_subplot(gs[1, n_camadas - 1])
    sc = ax_rmse.scatter(range(len(df)), df["Subleito"],
                         c=df["RMSE"], cmap="RdYlGn_r",
                         s=60, edgecolors="k", lw=0.5, zorder=3)
    plt.colorbar(sc, ax=ax_rmse, label="RMSE")
    ax_rmse.axhline(resultados["Subleito"]["E_rep"], color="red",
                    lw=2, ls="--",
                    label=f"E_rep={resultados['Subleito']['E_rep']:.0f}")
    ax_rmse.set_xlabel("Ponto de medição")
    ax_rmse.set_ylabel("Subleito (MPa)")
    ax_rmse.set_title("Subleito × RMSE", fontweight="bold")
    ax_rmse.legend(fontsize=8)

    # Tabela resumo
    ax_tab = fig.add_subplot(gs[1, n_camadas])
    ax_tab.axis("off")
    td = [[cam,
           f"{resultados[cam]['mu']:.0f}",
           f"{resultados[cam]['sigma']:.0f}",
           f"{resultados[cam]['cv']:.3f}",
           f"{resultados[cam]['E_rep']:.0f}"]
          for cam in camadas]
    tbl = ax_tab.table(
        cellText=td,
        colLabels=["Camada", "μ", "σ", "CV", "E_rep"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.9)
    idx_sub_row = camadas.index("Subleito") + 1
    for col in range(5):
        tbl[idx_sub_row, col].set_facecolor("#FFCDD2")
    ax_tab.set_title(f"E_rep — P{res['percentil']}", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print()


def imprimir_relatorio(nome, res):
    """Imprime relatório textual de um segmento."""
    r   = res["resultados"]
    sep = "=" * 60
    print(sep)
    print(f"  SEGMENTO : {nome}")
    print(f"  Camadas  : {len(res['camadas'])} — {', '.join(res['camadas'])}")
    print(sep)
    print(f"  Pontos analisados   : {len(res['df'])}")
    print(f"  K ótimo (BIC)       : {res['k_otimo']}")
    print(f"  Componente dominante: C{res['comp_dom']+1} "
          f"(π = {res['gmm'].weights_[res['comp_dom']]:.3f})")
    print(f"  Percentil adotado   : P{res['percentil']}  (z = {res['z']:.3f})")
    print()
    print(f"  {'Camada':<15} {'μ (MPa)':>9} {'σ (MPa)':>9} "
          f"{'CV':>7} {'E_rep (MPa)':>12}")
    print("  " + "-" * 56)
    for cam in res["camadas"]:
        rv = r[cam]
        print(f"  {cam:<15} {rv['mu']:>9.1f} {rv['sigma']:>9.1f} "
              f"{rv['cv']:>7.3f} {rv['E_rep']:>12.1f}")
    print()


print("✓ Funções do pipeline v2 carregadas.")
print("  Suporte: 3 camadas (4 colunas) e 4 camadas (5 colunas).")


# ============================================================
# ── CÉLULA 3: Upload e processamento ────────────────────────

from google.colab import files

PERCENTIL = 15   # ← altere para 10 ou 20 se desejar

print("─" * 60)
print("Faça o upload dos arquivos de cada segmento.")
print("Pode misturar arquivos com 3 e 4 camadas.")
print("─" * 60)

uploaded = files.upload()

if not uploaded:
    print("Nenhum arquivo enviado.")
else:
    resultados_todos = {}

    for nome_arquivo, conteudo in uploaded.items():
        print(f"\nProcessando: {nome_arquivo} ...")
        try:
            buf = io.BytesIO(conteudo)
            df, camadas = carregar_arquivo(buf, nome_arquivo)
            res = rodar_pipeline(df, camadas, percentil=PERCENTIL)
            nome_seg = nome_arquivo.rsplit(".", 1)[0]
            resultados_todos[nome_seg] = res
            print(f"  ✓ OK — {len(df)} pontos — "
                  f"{len(camadas)} camadas — K = {res['k_otimo']}")
        except Exception as e:
            print(f"  ✗ Erro: {e}")

    print(f"\n✓ {len(resultados_todos)} segmento(s) processado(s).")


# ============================================================
# ── CÉLULA 4: Resultados por segmento ───────────────────────

for nome, res in resultados_todos.items():
    imprimir_relatorio(nome, res)
    plotar_segmento(res, nome)


# ============================================================
# ── CÉLULA 5: Comparativo e segmento crítico ────────────────

print("\n" + "=" * 70)
print("  COMPARATIVO ENTRE SEGMENTOS")
print("=" * 70)

linhas        = []
erep_subleito = []

for nome, res in resultados_todos.items():
    r        = res["resultados"]
    erep_sub = r["Subleito"]["E_rep"]
    erep_subleito.append(erep_sub)
    linha = {
        "Segmento"         : nome,
        "Camadas"          : len(res["camadas"]),
        "K"                : res["k_otimo"],
        "C. Dominante"     : f"C{res['comp_dom']+1}",
        "μ Sub. (MPa)"     : f"{r['Subleito']['mu']:.1f}",
        "CV Sub."          : f"{r['Subleito']['cv']:.3f}",
        "E_rep Sub. (MPa)" : f"{erep_sub:.1f}",
        "E_rep Base (MPa)" : f"{r['Base']['E_rep']:.1f}",
        "E_rep Rev. (MPa)" : f"{r['Revestimento']['E_rep']:.1f}",
    }
    if "Sub-base" in r:
        linha["E_rep Sub-base (MPa)"] = f"{r['Sub-base']['E_rep']:.1f}"
    linhas.append(linha)

df_comp = pd.DataFrame(linhas)
print(df_comp.to_string(index=False))

idx_critico  = int(np.argmin(erep_subleito))
nome_critico = list(resultados_todos.keys())[idx_critico]
erep_critico = erep_subleito[idx_critico]

print("\n" + "★" * 70)
print(f"  SEGMENTO CRÍTICO : {nome_critico}")
print(f"  E_rep subleito   : {erep_critico:.1f} MPa  (P{PERCENTIL})")
print(f"  → Este é o módulo governante para entrada no FAARFIELD.")
print("★" * 70)
