import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, norm
from matplotlib.backends.backend_pdf import PdfPages
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GMM — Pavimentos Aeroportuários",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Módulo Representativo — Pavimentos Aeroportuários")
st.markdown(
    "Metodologia baseada em **Mistura de Gaussianas (GMM)** para determinação "
    "do módulo de resiliência representativo a partir de dados retroanalisados (FWD)."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────────────────────
def detectar_camadas(df_raw):
    """
    Detecta automaticamente se o arquivo tem 3 ou 4 camadas.
    Última coluna = sempre RMSE.
    Colunas intermediárias = camadas na ordem correta.
    Aceita 4 colunas (3 camadas) ou 5 colunas (4 camadas).
    """
    n_cols = df_raw.shape[1]
    if n_cols == 4:
        camadas = ["Revestimento", "Base", "Subleito"]
    elif n_cols == 5:
        camadas = ["Revestimento", "Base", "Sub-base", "Subleito"]
    else:
        raise ValueError(
            f"Arquivo com {n_cols} colunas não é suportado. "
            "Esperado: 4 colunas (3 camadas + RMSE) ou 5 colunas (4 camadas + RMSE)."
        )
    return camadas


def carregar_arquivo(uploaded_file):
    """
    Carrega CSV ou XLSX.
    Sempre usa as primeiras 4 ou 5 colunas, ignorando nomes originais.
    Ordem obrigatória: Revestimento, Base, [Sub-base,] Subleito, RMSE.
    Nomes das colunas e textos extras são ignorados — renomeação na força.
    """
    nome = uploaded_file.name.lower()
    if nome.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, header=0)
    else:
        try:
            df = pd.read_csv(uploaded_file, sep=";", header=0)
            if df.shape[1] < 4:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=",", header=0)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", header=0)

    # Pega apenas as primeiras 5 colunas e converte tudo para numérico
    df = df.iloc[:, :5].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove linhas onde TODAS as colunas são NaN (linhas de texto/cabeçalho extra)
    df = df.dropna(how="all")

    # Detecta se tem 3 ou 4 camadas pelo número de colunas com dados
    camadas = detectar_camadas(df)

    # Mantém apenas as colunas necessárias e renomeia na força
    df = df.iloc[:, :len(camadas) + 1].copy()
    df.columns = camadas + ["RMSE"]

    # Remove linhas com qualquer NaN restante
    df = df.dropna()

    return df, camadas


def rodar_pipeline(df, camadas, percentil=15):
    """Pipeline GMM completo. Funciona com 3 ou 4 camadas."""

    # Pesos pelo RMSE
    pesos = 1.0 / df["RMSE"].values
    pesos = pesos / pesos.sum()

    # Reamostragem ponderada
    np.random.seed(42)
    idx = np.random.choice(len(df), size=500, replace=True, p=pesos)

    # Padronização
    X = df[camadas].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_resamp = X_scaled[idx]

    # Seleção de K via BIC
    bic, aic = [], []
    for k in range(1, 6):
        g = GaussianMixture(n_components=k, covariance_type="diag",
                            random_state=42, n_init=10)
        g.fit(X_resamp)
        bic.append(g.bic(X_scaled))   # BIC nos dados originais (n real)
        aic.append(g.aic(X_scaled))   # garante penalização adequada
    k_otimo = int(np.argmin(bic)) + 1

    # GMM final
    gmm = GaussianMixture(n_components=k_otimo, covariance_type="diag",
                          random_state=42, n_init=20)
    gmm.fit(X_resamp)
    labels = gmm.predict(X_scaled)

    # Parâmetros na escala original
    # covariance_type=diag: covariances_ shape (K, n_features), sem termos cruzados
    means = scaler.inverse_transform(gmm.means_)
    stds  = np.zeros_like(means)
    for k in range(k_otimo):
        stds[k] = np.sqrt(gmm.covariances_[k]) * scaler.scale_

    # Componente dominante — sempre pelo subleito (última camada)
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


def gerar_figura(res, nome_segmento):
    """Gera figura adaptada para 3 ou 4 camadas."""
    CORES      = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800"]
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

    # Layout: linha 0 = distribuições + BIC | linha 1 = scatters + RMSE + tabela
    n_cols = n_camadas + 1   # camadas + BIC
    fig = plt.figure(figsize=(5 * n_cols, 12))
    fig.suptitle(
        f"Segmento: {nome_segmento}  |  {n_camadas} camadas  |  "
        f"K = {k_otimo}  |  P{res['percentil']}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.48, wspace=0.38)

    # ── Linha 0: distribuições ──
    for i, cam in enumerate(camadas):
        ax  = fig.add_subplot(gs[0, i])
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

    # ── Linha 1: scatters, RMSE, tabela ──
    # Scatter: pares de camadas adjacentes
    pares = [(camadas[i], camadas[i+1]) for i in range(n_camadas - 1)]
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
    return fig


def gerar_pdf(resultados_todos, percentil):
    """Gera PDF consolidado com todos os segmentos."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # Capa
        fig_capa, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65,
                "Relatório de Análise GMM\nPavimentos Aeroportuários",
                ha="center", va="center", fontsize=22,
                fontweight="bold", color="#1F497D",
                transform=ax.transAxes)
        ax.text(0.5, 0.50,
                f"Módulo Representativo — Percentil {percentil}",
                ha="center", va="center", fontsize=14,
                color="#555555", transform=ax.transAxes)
        ax.text(0.5, 0.40,
                f"Total de segmentos analisados: {len(resultados_todos)}",
                ha="center", va="center", fontsize=12,
                color="#333333", transform=ax.transAxes)
        pdf.savefig(fig_capa, bbox_inches="tight")
        plt.close(fig_capa)

        # Figura de cada segmento
        for nome, res in resultados_todos.items():
            fig = gerar_figura(res, nome)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Tabela comparativa
        erep_vals    = [res["resultados"]["Subleito"]["E_rep"]
                        for res in resultados_todos.values()]
        idx_critico  = int(np.argmin(erep_vals))
        nome_critico = list(resultados_todos.keys())[idx_critico]

        fig_tab, ax_tab = plt.subplots(
            figsize=(14, 3 + len(resultados_todos) * 0.7))
        ax_tab.axis("off")
        ax_tab.set_title(
            "Comparativo — Módulos Representativos por Segmento",
            fontsize=13, fontweight="bold", color="#1F497D", pad=20)

        headers = ["Segmento", "Camadas", "K", "C.Dom.",
                   "μ Sub.", "CV Sub.",
                   "E_rep Sub.", "E_rep Base", "E_rep Rev."]
        rows = []
        for nome, res in resultados_todos.items():
            r = res["resultados"]
            rows.append([
                nome,
                str(len(res["camadas"])),
                str(res["k_otimo"]),
                f"C{res['comp_dom']+1}",
                f"{r['Subleito']['mu']:.1f}",
                f"{r['Subleito']['cv']:.3f}",
                f"{r['Subleito']['E_rep']:.1f}",
                f"{r['Base']['E_rep']:.1f}",
                f"{r['Revestimento']['E_rep']:.1f}",
            ])

        tbl = ax_tab.table(
            cellText=rows, colLabels=headers,
            cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 2.2)

        for col in range(len(headers)):
            tbl[0, col].set_facecolor("#1F497D")
            tbl[0, col].get_text().set_color("white")
            tbl[0, col].get_text().set_fontweight("bold")
            tbl[idx_critico + 1, col].set_facecolor("#FFCDD2")
            tbl[idx_critico + 1, col].get_text().set_fontweight("bold")

        for row in range(1, len(rows) + 1):
            if row != idx_critico + 1:
                fill = "#EBF3FB" if row % 2 == 0 else "white"
                for col in range(len(headers)):
                    tbl[row, col].set_facecolor(fill)

        ax_tab.text(
            0.5, 0.02,
            f"★ Segmento crítico: {nome_critico}  |  "
            f"E_rep subleito = {erep_vals[idx_critico]:.1f} MPa  |  P{percentil}",
            ha="center", fontsize=10, fontweight="bold",
            color="#C0392B", transform=ax_tab.transAxes)

        pdf.savefig(fig_tab, bbox_inches="tight")
        plt.close(fig_tab)

    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")
    percentil = st.selectbox(
        "Percentil de projeto",
        options=[10, 15, 20],
        index=1,
        help="P15 é o mais comum para pistas aeroportuárias principais."
    )
    st.markdown("---")
    st.markdown(
        "**Formato dos arquivos:**\n\n"
        "**3 camadas** (4 colunas):\n"
        "`Revestimento | Base | Subleito | RMSE`\n\n"
        "**4 camadas** (5 colunas):\n"
        "`Revestimento | Base | Sub-base | Subleito | RMSE`\n\n"
        "Aceita `.csv` (`;` ou `,`) e `.xlsx`\n\n"
        "Pode misturar arquivos de 3 e 4 camadas."
    )
    st.markdown("---")
    st.caption("O componente dominante é sempre definido pelo menor μ de subleito.")

# Upload
st.subheader("📂 Upload dos Segmentos Homogêneos")
st.markdown(
    "Faça o upload de **um arquivo por segmento**. "
    "Arquivos com 3 ou 4 camadas são aceitos simultaneamente."
)

uploaded_files = st.file_uploader(
    "Selecione os arquivos (CSV ou XLSX)",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown("---")
    st.subheader("✏️ Confirme ou edite os nomes dos segmentos")

    nomes_segmentos = {}
    cols = st.columns(min(len(uploaded_files), 3))
    for i, f in enumerate(uploaded_files):
        nome_padrao = f.name.rsplit(".", 1)[0]
        with cols[i % 3]:
            nomes_segmentos[i] = st.text_input(
                f"Arquivo: `{f.name}`",
                value=nome_padrao,
                key=f"nome_{i}"
            )

    st.markdown("---")

    if st.button("🚀 Processar segmentos", type="primary",
                 use_container_width=True):

        resultados_todos = {}
        erros = []
        barra = st.progress(0, text="Iniciando processamento...")

        for i, f in enumerate(uploaded_files):
            nome = nomes_segmentos[i]
            barra.progress(
                int((i / len(uploaded_files)) * 100),
                text=f"Processando: {nome}..."
            )
            try:
                f.seek(0)
                df, camadas = carregar_arquivo(f)
                res = rodar_pipeline(df, camadas, percentil=percentil)
                resultados_todos[nome] = res
            except Exception as e:
                erros.append(f"**{nome}**: {str(e)}")

        barra.progress(100, text="Concluído!")

        if erros:
            st.error("Erros nos seguintes arquivos:")
            for e in erros:
                st.markdown(f"- {e}")

        if resultados_todos:

            # ── Resultados por segmento ──
            st.markdown("---")
            st.subheader("📊 Resultados por Segmento")

            for nome, res in resultados_todos.items():
                with st.expander(
                    f"📍 **{nome}** — "
                    f"{len(res['camadas'])} camadas  |  K = {res['k_otimo']}",
                    expanded=True
                ):
                    r = res["resultados"]
                    camadas = res["camadas"]

                    # Métricas rápidas
                    cols_met = st.columns(len(camadas) + 1)
                    cols_met[0].metric("K ótimo (BIC)", res["k_otimo"])
                    for j, cam in enumerate(camadas):
                        label = (f"E_rep Subleito (P{percentil})"
                                 if cam == "Subleito" else f"E_rep {cam}")
                        cols_met[j + 1].metric(
                            label, f"{r[cam]['E_rep']:.0f} MPa",
                            delta=f"μ = {r[cam]['mu']:.0f} MPa",
                            delta_color="off"
                        )

                    # Tabela detalhada
                    tabela = pd.DataFrame([
                        {
                            "Camada": cam,
                            "μ (MPa)": f"{r[cam]['mu']:.1f}",
                            "σ (MPa)": f"{r[cam]['sigma']:.1f}",
                            "CV": f"{r[cam]['cv']:.3f}",
                            f"E_rep P{percentil} (MPa)": f"{r[cam]['E_rep']:.1f}"
                        }
                        for cam in camadas
                    ])
                    st.dataframe(tabela, use_container_width=True,
                                 hide_index=True)

                    # Figura
                    fig = gerar_figura(res, nome)
                    st.pyplot(fig)
                    plt.close(fig)

            # ── Comparativo ──
            st.markdown("---")
            st.subheader("🏆 Comparativo entre Segmentos")

            dados_comp = []
            erep_vals  = []
            for nome, res in resultados_todos.items():
                r = res["resultados"]
                erep_sub = r["Subleito"]["E_rep"]
                erep_vals.append(erep_sub)
                linha = {
                    "Segmento"    : nome,
                    "Camadas"     : len(res["camadas"]),
                    "K"           : res["k_otimo"],
                    "C. Dom."     : f"C{res['comp_dom']+1}",
                    "μ Sub. (MPa)": f"{r['Subleito']['mu']:.1f}",
                    "CV Sub."     : f"{r['Subleito']['cv']:.3f}",
                    f"E_rep Sub. P{percentil} (MPa)": f"{erep_sub:.1f}",
                    f"E_rep Base P{percentil} (MPa)": f"{r['Base']['E_rep']:.1f}",
                    f"E_rep Rev. P{percentil} (MPa)": f"{r['Revestimento']['E_rep']:.1f}",
                }
                # Sub-base se existir
                if "Sub-base" in r:
                    linha[f"E_rep Sub-base P{percentil} (MPa)"] = \
                        f"{r['Sub-base']['E_rep']:.1f}"
                dados_comp.append(linha)

            df_comp      = pd.DataFrame(dados_comp)
            idx_critico  = int(np.argmin(erep_vals))
            nome_critico = list(resultados_todos.keys())[idx_critico]

            def highlight_critico(row):
                if row["Segmento"] == nome_critico:
                    return ["background-color: #FFCDD2; font-weight: bold"] \
                           * len(row)
                return [""] * len(row)

            st.dataframe(
                df_comp.style.apply(highlight_critico, axis=1),
                use_container_width=True,
                hide_index=True
            )

            st.error(
                f"🔴 **Segmento crítico: {nome_critico}** — "
                f"E_rep subleito = **{erep_vals[idx_critico]:.1f} MPa** "
                f"(P{percentil})\n\n"
                "Este é o módulo governante para entrada no FAARFIELD."
            )

            # ── Download PDF ──
            st.markdown("---")
            st.subheader("📥 Download do Relatório")

            with st.spinner("Gerando PDF..."):
                pdf_buf = gerar_pdf(resultados_todos, percentil)

            st.download_button(
                label="⬇️ Baixar Relatório PDF",
                data=pdf_buf,
                file_name="relatorio_gmm_pavimento.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
