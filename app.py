import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KDE — Pavimentos Aeroportuários",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Módulo Representativo — Pavimentos Aeroportuários")
st.markdown(
    "Metodologia baseada em **Kernel Density Estimation (KDE)** ponderado pelo RMSE "
    "para determinação do módulo de resiliência representativo a partir de dados "
    "retroanalisados (FWD/HWD)."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
N_RESAMPLE = 10000


# ─────────────────────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────────────────────
def detectar_camadas(df_raw):
    n_cols = df_raw.shape[1]
    if n_cols == 4:
        return ["Revestimento", "Base", "Subleito"]
    elif n_cols == 5:
        return ["Revestimento", "Base", "Sub-base", "Subleito"]
    else:
        raise ValueError(
            f"Arquivo com {n_cols} colunas não suportado. "
            "Esperado: 4 colunas (3 camadas + RMSE) ou 5 colunas (4 camadas + RMSE)."
        )


def carregar_arquivo(uploaded_file):
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

    df = df.iloc[:, :5].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    camadas = detectar_camadas(df)
    df = df.iloc[:, :len(camadas) + 1].copy()
    df.columns = camadas + ["RMSE"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df, camadas


def calcular_kde_camada(serie, rmse_serie, percentil):
    temp = pd.DataFrame({
        "val": pd.to_numeric(serie, errors="coerce"),
        "err": pd.to_numeric(rmse_serie, errors="coerce")
    }).dropna()

    # Todos os pontos — ponderação pelo RMSE garante influência proporcional à qualidade
    pesos = 1.0 / temp["err"].values.astype(float)
    pesos = pesos / pesos.sum()
    vals  = temp["val"].values.astype(float)

    # Estatísticas sem ponderação (μ−σ clássico)
    mu    = vals.mean()
    sigma = vals.std(ddof=1)
    cv    = sigma / mu if mu != 0 else 0

    # KDE ponderado
    kde     = gaussian_kde(vals, weights=pesos, bw_method="silverman")
    v_min   = vals.min() * 0.5
    v_max   = vals.max() * 1.5
    amostra = kde.resample(N_RESAMPLE).flatten()
    amostra = amostra[(amostra >= v_min) & (amostra <= v_max)]
    if len(amostra) == 0:
        amostra = vals
    e_rep_p = np.percentile(amostra, percentil)

    # μ−σ clássico (sem ponderação)
    e_rep_ms = max(mu - sigma, 0)

    return {
        "kde": kde, "vals": vals, "pesos": pesos,
        "mu": mu, "sigma": sigma, "cv": cv,
        "E_rep": e_rep_p,
        "e_rep_ms": e_rep_ms,
        "percentil": percentil,
        "n_total": len(temp)
    }


def rodar_pipeline(df, camadas, percentil=15):
    resultados = {}
    for cam in camadas:
        resultados[cam] = calcular_kde_camada(df[cam], df["RMSE"], percentil)
    return {
        "df": df,
        "camadas": camadas,
        "resultados": resultados,
        "percentil": percentil
    }


def gerar_figura(res, nome_segmento):
    COR_KDE = "#0077B6"
    COR_P   = "#E74C3C"
    COR_MS  = "#E07A00"
    COR_MU  = "#1A1A2E"

    df         = res["df"]
    camadas    = res["camadas"]
    resultados = res["resultados"]
    percentil  = res["percentil"]
    n_camadas  = len(camadas)
    n_cols     = n_camadas + 1

    fig = plt.figure(figsize=(5 * n_cols, 10))
    fig.suptitle(
        f"Segmento: {nome_segmento}  |  {n_camadas} camadas  |  "
        f"KDE ponderado RMSE  |  P{percentil} vs μ−σ",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.48, wspace=0.38)

    # ── Linha 0: distribuições KDE ──
    for i, cam in enumerate(camadas):
        ax = fig.add_subplot(gs[0, i])
        r  = resultados[cam]
        x_r = np.linspace(r["vals"].min() * 0.80, r["vals"].max() * 1.15, 500)
        y_r = r["kde"](x_r)

        ax.fill_between(x_r, y_r, alpha=0.15, color=COR_KDE)
        ax.plot(x_r, y_r, color=COR_KDE, lw=2.5, label="KDE ponderado")
        ax.axvline(r["E_rep"], color=COR_P, lw=2, ls="--",
                   label=f"P{percentil} = {r['E_rep']:.0f}")
        ax.axvline(r["e_rep_ms"], color=COR_MS, lw=2, ls="-.",
                   label=f"μ−σ = {r['e_rep_ms']:.0f}")
        ax.axvline(r["mu"], color=COR_MU, lw=1.2, ls=":",
                   label=f"μ = {r['mu']:.0f}")
        ax.set_title(cam, fontweight="bold")
        ax.set_xlabel("Módulo (MPa)")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7)



    # ── Subleito × RMSE ──
    ax_rmse = fig.add_subplot(gs[0, n_camadas])
    sc = ax_rmse.scatter(
        range(len(df)), df["Subleito"],
        c=df["RMSE"], cmap="RdYlGn_r",
        s=65, edgecolors="k", lw=0.5, zorder=3
    )
    plt.colorbar(sc, ax=ax_rmse, label="RMSE")
    ax_rmse.axhline(resultados["Subleito"]["E_rep"], color=COR_P,
                    lw=2, ls="--",
                    label=f"P{percentil}={resultados['Subleito']['E_rep']:.0f}")
    ax_rmse.axhline(resultados["Subleito"]["e_rep_ms"], color=COR_MS,
                    lw=2, ls="-.",
                    label=f"μ−σ={resultados['Subleito']['e_rep_ms']:.0f}")
    ax_rmse.set_xlabel("Ponto de medição")
    ax_rmse.set_ylabel("Subleito (MPa)")
    ax_rmse.set_title("Subleito × RMSE", fontweight="bold")
    ax_rmse.legend(fontsize=8)

    # ── Tabela resumo ──
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")

    td = [[cam,
           str(resultados[cam]["n_total"]),
           f"{resultados[cam]['mu']:.0f}",
           f"{resultados[cam]['sigma']:.0f}",
           f"{resultados[cam]['cv']:.3f}",
           f"{resultados[cam]['E_rep']:.0f}",
           f"{resultados[cam]['e_rep_ms']:.0f}",
           f"{abs(resultados[cam]['E_rep'] - resultados[cam]['e_rep_ms']):.0f}"]
          for cam in camadas]

    tbl = ax_tab.table(
        cellText=td,
        colLabels=["Camada", "n",
                   "μ", "σ", "CV",
                   f"E_rep P{percentil}", "E_rep μ−σ", "Δ"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 2.0)

    idx_sub_row = camadas.index("Subleito") + 1
    for col in range(8):
        tbl[idx_sub_row, col].set_facecolor("#DBEAFE")

    for i, cam in enumerate(camadas):
        p = resultados[cam]["E_rep"]
        m = resultados[cam]["e_rep_ms"]
        if p > 0 and abs(p - m) / p > 0.10:
            tbl[i + 1, 9].set_facecolor("#FEE2E2")

    ax_tab.set_title(
        f"Comparativo P{percentil} vs μ−σ  (vermelho = diferença > 10%)",
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gerar_pdf(resultados_todos, percentil):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # Capa
        fig_capa, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65,
                "Relatório de Análise KDE\nPavimentos Aeroportuários",
                ha="center", va="center", fontsize=22,
                fontweight="bold", color="#0077B6",
                transform=ax.transAxes)
        ax.text(0.5, 0.50,
                f"Módulo Representativo — P{percentil} vs μ−σ\n"
                "KDE ponderado pelo RMSE  |  Silverman (1986)",
                ha="center", va="center", fontsize=13,
                color="#555555", transform=ax.transAxes)
        ax.text(0.5, 0.38,
                f"Total de segmentos analisados: {len(resultados_todos)}",
                ha="center", va="center", fontsize=12,
                color="#333333", transform=ax.transAxes)
        pdf.savefig(fig_capa, bbox_inches="tight")
        plt.close(fig_capa)

        # Figura por segmento
        for nome, res in resultados_todos.items():
            fig = gerar_figura(res, nome)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Tabela comparativa
        nomes       = list(resultados_todos.keys())
        erep_vals   = [res["resultados"]["Subleito"]["E_rep"]
                       for res in resultados_todos.values()]
        idx_critico = int(np.argmin(erep_vals))
        nome_critico = nomes[idx_critico]

        fig_tab, ax_tab = plt.subplots(
            figsize=(16, 3 + len(resultados_todos) * 0.8))
        ax_tab.axis("off")
        ax_tab.set_title(
            "Comparativo — Módulos Representativos por Segmento (KDE)",
            fontsize=13, fontweight="bold", color="#0077B6", pad=20)

        headers = ["Segmento",
                   f"P{percentil} Rev.", "μ−σ Rev.",
                   f"P{percentil} Base", "μ−σ Base",
                   f"P{percentil} Sub.", "μ−σ Sub.", "Δ Sub."]
        rows = []
        for nome, res in resultados_todos.items():
            r = res["resultados"]
            rows.append([
                nome,
                f"{r['Revestimento']['E_rep']:.1f}",
                f"{r['Revestimento']['e_rep_ms']:.1f}",
                f"{r['Base']['E_rep']:.1f}",
                f"{r['Base']['e_rep_ms']:.1f}",
                f"{r['Subleito']['E_rep']:.1f}",
                f"{r['Subleito']['e_rep_ms']:.1f}",
                f"{abs(r['Subleito']['E_rep'] - r['Subleito']['e_rep_ms']):.1f}",
            ])

        tbl = ax_tab.table(
            cellText=rows, colLabels=headers,
            cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 2.2)

        for col in range(len(headers)):
            tbl[0, col].set_facecolor("#0077B6")
            tbl[0, col].get_text().set_color("white")
            tbl[0, col].get_text().set_fontweight("bold")
            tbl[idx_critico + 1, col].set_facecolor("#DBEAFE")
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
            color="#0077B6", transform=ax_tab.transAxes)

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
    st.markdown(
        f"**Pipeline KDE:**\n\n"
        f"1. Pesos = 1/RMSE — todos os pontos\n"
        f"2. KDE com bandwidth de Silverman\n"
        f"3. Reamostragem N = {N_RESAMPLE:,} pontos\n"
        f"4. E_rep = P{{% }} da distribuição estimada"
    )
    st.markdown("---")
    st.caption("Silverman (1986) · FAA AC 150/5370-11B")

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
                    f"📍 **{nome}** — {len(res['camadas'])} camadas",
                    expanded=True
                ):
                    r       = res["resultados"]
                    camadas = res["camadas"]

                    # Métricas rápidas
                    cols_met = st.columns(len(camadas))
                    for j, cam in enumerate(camadas):
                        label = (f"E_rep Subleito (P{percentil})"
                                 if cam == "Subleito" else f"E_rep {cam}")
                        cols_met[j].metric(
                            label,
                            f"{r[cam]['E_rep']:.0f} MPa",
                            delta=f"μ−σ = {r[cam]['e_rep_ms']:.0f} MPa",
                            delta_color="off"
                        )

                    # Tabela detalhada
                    tabela = pd.DataFrame([
                        {
                            "Camada"          : cam,
                            "n"               : r[cam]["n_total"],
                            "μ (MPa)"         : f"{r[cam]['mu']:.1f}",
                            "σ (MPa)"         : f"{r[cam]['sigma']:.1f}",
                            "CV"              : f"{r[cam]['cv']:.3f}",
                            f"E_rep P{percentil} (MPa)": f"{r[cam]['E_rep']:.1f}",
                            "E_rep μ−σ (MPa)" : f"{r[cam]['e_rep_ms']:.1f}",
                            "Δ (MPa)"         : f"{abs(r[cam]['E_rep'] - r[cam]['e_rep_ms']):.1f}"
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
            st.subheader("🏆 Comparativo entre Segmentos — Todas as Camadas")

            nomes       = list(resultados_todos.keys())
            erep_p_sub  = [resultados_todos[n]["resultados"]["Subleito"]["E_rep"]
                           for n in nomes]
            erep_ms_sub = [resultados_todos[n]["resultados"]["Subleito"]["e_rep_ms"]
                           for n in nomes]
            idx_crit_p  = int(np.argmin(erep_p_sub))
            idx_crit_ms = int(np.argmin(erep_ms_sub))

            dados_comp = []
            for nome, res in resultados_todos.items():
                r   = res["resultados"]
                cam = res["camadas"]
                linha = {"Segmento": nome}
                for c in cam:
                    linha[f"P{percentil} {c}"] = f"{r[c]['E_rep']:.1f}"
                    linha[f"μ−σ {c}"]          = f"{r[c]['e_rep_ms']:.1f}"
                    linha[f"Δ {c}"]            = f"{abs(r[c]['E_rep'] - r[c]['e_rep_ms']):.1f}"
                dados_comp.append(linha)

            df_comp = pd.DataFrame(dados_comp)

            def highlight_critico(row):
                nome  = row["Segmento"]
                cores = [""] * len(row)
                if nome == nomes[idx_crit_p]:
                    cores = ["background-color: #DBEAFE; font-weight: bold"] \
                            * len(row)
                if nome == nomes[idx_crit_ms] and idx_crit_ms != idx_crit_p:
                    cores = ["background-color: #FEF3C7; font-weight: bold"] \
                            * len(row)
                return cores

            st.dataframe(
                df_comp.style.apply(highlight_critico, axis=1),
                use_container_width=True,
                hide_index=True
            )

            if idx_crit_p == idx_crit_ms:
                st.success(
                    f"✅ Ambos os critérios identificam o mesmo segmento crítico: "
                    f"**{nomes[idx_crit_p]}**\n\n"
                    f"E_rep subleito — P{percentil}: **{erep_p_sub[idx_crit_p]:.1f} MPa** | "
                    f"μ−σ: **{erep_ms_sub[idx_crit_ms]:.1f} MPa**"
                )
            else:
                st.warning(
                    f"⚠️ Os critérios divergem no segmento crítico.\n\n"
                    f"🔵 P{percentil}: **{nomes[idx_crit_p]}** — "
                    f"{erep_p_sub[idx_crit_p]:.1f} MPa\n\n"
                    f"🟡 μ−σ: **{nomes[idx_crit_ms]}** — "
                    f"{erep_ms_sub[idx_crit_ms]:.1f} MPa"
                )

            st.info(
                "💡 O módulo de subleito do segmento crítico é o valor "
                "governante para entrada no FAARFIELD."
            )

            # ── Download PDF ──
            st.markdown("---")
            st.subheader("📥 Download do Relatório")

            with st.spinner("Gerando PDF..."):
                pdf_buf = gerar_pdf(resultados_todos, percentil)

            st.download_button(
                label="⬇️ Baixar Relatório PDF",
                data=pdf_buf,
                file_name="relatorio_kde_pavimento.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
