import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, shapiro, anderson
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
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
# DICIONÁRIO DE MATERIAIS → INTERVALOS DE MR (MPa)
# Fontes: Bernucci et al. (2022), Tabela 11 (DNIT/IPR),
#         Soares et al. (2000), Possebon (2018)
# ─────────────────────────────────────────────────────────────
MATERIAIS_CAMADA = {
    "Revestimento": {
        "Revestimento Asfáltico (CA convencional ou modificado)": (1000, 10500),
        "Revestimento — Concreto de Cimento Portland":            (25000, 35000),
    },
    "Base": {
        "Base Granular (brita graduada, macadame, solo-brita)":    (100, 450),
        "Base Estabilizada Quimicamente (solo-cimento, BGTC, reciclado)": (400, 15000),
        "Base — Concreto Compactado com Rolo (CCR)":               (7000, 29500),
    },
    "Sub-base": {
        "Sub-base Granular (brita graduada, macadame, solo-brita)":    (100, 450),
        "Sub-base Estabilizada Quimicamente (solo-cimento, BGTC, reciclado)": (400, 15000),
        "Sub-base — Concreto Compactado com Rolo (CCR)":               (7000, 29500),
    },
    "Subleito": {
        "Subleito (solos lateríticos, não lateríticos e melhorados)": (25, 400),
    },
}

# ─────────────────────────────────────────────────────────────
# TESTES DE NORMALIDADE
# ─────────────────────────────────────────────────────────────
def testar_normalidade(df, camadas):
    registros = []
    for cam in camadas:
        vals = pd.to_numeric(df[cam], errors="coerce").dropna().values
        n = len(vals)
        if n < 3:
            registros.append({
                "Camada": cam, "n": n,
                "SW_stat": None, "SW_p": None, "SW_normal": None,
                "AD_stat": None, "AD_critico": None, "AD_normal": None,
                "Conclusão": "Insuficiente"
            })
            continue

        sw_stat, sw_p = shapiro(vals)
        sw_normal = sw_p >= 0.05

        ad_result  = anderson(vals, dist="norm")
        ad_stat    = ad_result.statistic
        ad_critico = ad_result.critical_values[2]
        ad_normal  = ad_stat < ad_critico

        if sw_normal and ad_normal:
            conclusao = "Normal"
        elif not sw_normal and not ad_normal:
            conclusao = "Não normal"
        else:
            conclusao = "Inconclusivo"

        registros.append({
            "Camada": cam, "n": n,
            "SW_stat": round(sw_stat, 4), "SW_p": round(sw_p, 4),
            "SW_normal": sw_normal,
            "AD_stat": round(ad_stat, 4), "AD_critico": round(ad_critico, 4),
            "AD_normal": ad_normal,
            "Conclusão": conclusao
        })
    return pd.DataFrame(registros)


def plotar_mapa_normalidade(df_norm, nome_segmento):
    cod  = {"Normal": 1, "Inconclusivo": 0, "Não normal": -1, "Insuficiente": 0}
    cmap = mcolors.ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    camadas    = df_norm["Camada"].tolist()
    conclusoes = df_norm["Conclusão"].tolist()
    valores    = np.array([[cod.get(c, 0) for c in conclusoes]])

    fig, ax = plt.subplots(figsize=(max(4, len(camadas) * 2), 1.6))
    ax.imshow(valores, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(camadas)))
    ax.set_xticklabels(camadas, fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.set_title(
        f"Testes de Normalidade — {nome_segmento}  (SW + AD, α = 5%)",
        fontsize=11, pad=8
    )

    for j, (cam, conc) in enumerate(zip(camadas, conclusoes)):
        ax.text(j, 0, conc, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor="#2ecc71", label="Normal"),
        Patch(facecolor="#f39c12", label="Inconclusivo"),
        Patch(facecolor="#e74c3c", label="Não normal"),
    ]
    ax.legend(handles=legenda, bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# DETECÇÃO DE CAMADAS E CARREGAMENTO
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


# ─────────────────────────────────────────────────────────────
# FILTRO DE BOUNDS FÍSICOS
# Aplicado ANTES de qualquer cálculo estatístico.
# Remove linhas onde qualquer camada está fora do intervalo
# fisicamente admissível definido pelo usuário.
# ─────────────────────────────────────────────────────────────
def aplicar_filtro_bounds(df, camadas, bounds_por_camada):
    """
    df                : DataFrame com colunas [camadas..., RMSE]
    camadas           : lista de nomes das camadas
    bounds_por_camada : dict {camada: (mr_min, mr_max)}

    Retorna (df_filtrado, n_removidos, detalhes_por_camada)
    """
    mascara = pd.Series([True] * len(df), index=df.index)
    detalhes = {}

    for cam in camadas:
        mr_min, mr_max = bounds_por_camada[cam]
        fora = (df[cam] < mr_min) | (df[cam] > mr_max)
        n_fora = fora.sum()
        detalhes[cam] = {
            "mr_min": mr_min,
            "mr_max": mr_max,
            "n_removidos": int(n_fora)
        }
        mascara = mascara & ~fora

    n_original  = len(df)
    df_filtrado = df[mascara].reset_index(drop=True)
    n_removidos = n_original - len(df_filtrado)

    return df_filtrado, n_removidos, detalhes


# ─────────────────────────────────────────────────────────────
# PIPELINE KDE
# ─────────────────────────────────────────────────────────────
def calcular_kde_camada(serie, rmse_serie, percentil):
    temp = pd.DataFrame({
        "val": pd.to_numeric(serie, errors="coerce"),
        "err": pd.to_numeric(rmse_serie, errors="coerce")
    }).dropna()

    pesos = 1.0 / temp["err"].values.astype(float)
    pesos = pesos / pesos.sum()
    vals  = temp["val"].values.astype(float)

    mu    = vals.mean()
    sigma = vals.std(ddof=1)
    cv    = sigma / mu if mu != 0 else 0

    mu_w    = np.sum(pesos * vals)
    sigma_w = np.sqrt(np.sum(pesos * (vals - mu_w) ** 2))
    cv_w    = sigma_w / mu_w if mu_w != 0 else 0
    e_rep_ms_w = max(mu_w - sigma_w, 0)

    kde     = gaussian_kde(vals, weights=pesos, bw_method="silverman")
    v_min   = vals.min() * 0.5
    v_max   = vals.max() * 1.5
    amostra = kde.resample(N_RESAMPLE).flatten()
    amostra = amostra[(amostra >= v_min) & (amostra <= v_max)]
    if len(amostra) == 0:
        amostra = vals
    e_rep_p = np.percentile(amostra, percentil)

    e_rep_ms = max(mu - sigma, 0)

    return {
        "kde": kde, "vals": vals, "pesos": pesos,
        "mu": mu, "sigma": sigma, "cv": cv,
        "mu_w": mu_w, "sigma_w": sigma_w, "cv_w": cv_w,
        "E_rep": e_rep_p,
        "e_rep_ms": e_rep_ms,
        "e_rep_ms_w": e_rep_ms_w,
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


# ─────────────────────────────────────────────────────────────
# FIGURA E PDF
# ─────────────────────────────────────────────────────────────
def gerar_figura(res, nome_segmento):
    COR_KDE  = "#0077B6"
    COR_P    = "#E74C3C"
    COR_MS   = "#E07A00"
    COR_MS_W = "#6A0DAD"
    COR_MU   = "#1A1A2E"

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

    for i, cam in enumerate(camadas):
        ax = fig.add_subplot(gs[0, i])
        r  = resultados[cam]
        x_r = np.linspace(r["vals"].min() * 0.80, r["vals"].max() * 1.15, 500)
        y_r = r["kde"](x_r)

        ax.fill_between(x_r, y_r, alpha=0.15, color=COR_KDE)
        ax.plot(x_r, y_r, color=COR_KDE, lw=2.5, label="KDE ponderado")
        ax.axvline(r["E_rep"], color=COR_P, lw=2, ls="--",
                   label=f"P{percentil} = {r['E_rep']:.0f}")
        ax.axvline(r["e_rep_ms_w"], color=COR_MS_W, lw=2, ls="-.",
                   label=f"μ_w−σ_w = {r['e_rep_ms_w']:.0f}")
        ax.axvline(r["e_rep_ms"], color=COR_MS, lw=2, ls="--",
                   label=f"μ−σ = {r['e_rep_ms']:.0f}")
        ax.axvline(r["mu"], color=COR_MU, lw=1.2, ls=":",
                   label=f"μ = {r['mu']:.0f}")
        ax.set_title(cam, fontweight="bold")
        ax.set_xlabel("Módulo (MPa)")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=7)

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
    ax_rmse.axhline(resultados["Subleito"]["e_rep_ms_w"], color=COR_MS_W,
                    lw=2, ls="-.",
                    label=f"μ_w−σ_w={resultados['Subleito']['e_rep_ms_w']:.0f}")
    ax_rmse.axhline(resultados["Subleito"]["e_rep_ms"], color=COR_MS,
                    lw=2, ls="--",
                    label=f"μ−σ={resultados['Subleito']['e_rep_ms']:.0f}")
    ax_rmse.set_xlabel("Ponto de medição")
    ax_rmse.set_ylabel("Subleito (MPa)")
    ax_rmse.set_title("Subleito × RMSE", fontweight="bold")
    ax_rmse.legend(fontsize=8)

    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")

    td = [[cam,
           str(resultados[cam]["n_total"]),
           f"{resultados[cam]['mu']:.0f}",
           f"{resultados[cam]['sigma']:.0f}",
           f"{resultados[cam]['cv']:.3f}",
           f"{resultados[cam]['E_rep']:.0f}",
           f"{resultados[cam]['e_rep_ms_w']:.0f}",
           f"{resultados[cam]['e_rep_ms']:.0f}"]
          for cam in camadas]

    tbl = ax_tab.table(
        cellText=td,
        colLabels=["Camada", "n",
                   "μ", "σ", "CV",
                   f"E_rep P{resultados[camadas[0]]['percentil']}", "E_rep μ_w−σ_w", "E_rep μ−σ"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 2.0)

    idx_sub_row = camadas.index("Subleito") + 1
    for col in range(8):
        tbl[idx_sub_row, col].set_facecolor("#DBEAFE")

    ax_tab.set_title(
        f"Módulos Representativos — P{resultados[camadas[0]]['percentil']} | μ_w−σ_w | μ−σ",
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gerar_pdf(resultados_todos, percentil):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig_capa, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.65,
                "Relatório de Análise KDE\nPavimentos Aeroportuários",
                ha="center", va="center", fontsize=22,
                fontweight="bold", color="#0077B6",
                transform=ax.transAxes)
        ax.text(0.5, 0.50,
                f"Módulo Representativo — P{percentil} | μ_w−σ_w | μ−σ\n"
                "KDE ponderado pelo RMSE  |  Silverman (1986)",
                ha="center", va="center", fontsize=13,
                color="#555555", transform=ax.transAxes)
        ax.text(0.5, 0.38,
                f"Total de segmentos analisados: {len(resultados_todos)}",
                ha="center", va="center", fontsize=12,
                color="#333333", transform=ax.transAxes)
        pdf.savefig(fig_capa, bbox_inches="tight")
        plt.close(fig_capa)

        for nome, res in resultados_todos.items():
            fig = gerar_figura(res, nome)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

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
        f"1. Filtro de bounds físicos\n"
        f"2. Pesos = 1/RMSE — todos os pontos\n"
        f"3. KDE com bandwidth de Silverman\n"
        f"4. Reamostragem N = {N_RESAMPLE:,} pontos\n"
        f"5. E_rep = P{{%}} da distribuição estimada"
    )
    st.markdown("---")
    st.caption("Silverman (1986) · FAA AC 150/5370-11B\nBernucci et al. (2022) · Tabela 11 IPR/DNIT")

# ─────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────
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

    # ── Pré-visualização para detectar camadas ──
    arquivos_info = {}
    for f in uploaded_files:
        try:
            f.seek(0)
            df_prev, camadas_prev = carregar_arquivo(f)
            arquivos_info[f.name] = {"camadas": camadas_prev, "file": f}
        except Exception as e:
            st.error(f"Erro ao ler `{f.name}`: {e}")

    if arquivos_info:
        # ── Detectar conjunto de camadas único entre todos os arquivos ──
        todas_camadas = []
        for info in arquivos_info.values():
            for cam in info["camadas"]:
                if cam not in todas_camadas:
                    todas_camadas.append(cam)
        # Ordenar canonicamente
        ordem = ["Revestimento", "Base", "Sub-base", "Subleito"]
        todas_camadas = [c for c in ordem if c in todas_camadas]

        # ── Seção: Filtro de Bounds Físicos ──
        st.subheader("🔍 Filtro de Bounds Físicos — Controle de Qualidade")
        st.markdown(
            "Defina os intervalos de MR admissíveis para cada camada. "
            "Linhas com qualquer valor fora do intervalo serão **removidas antes de qualquer cálculo** "
            "(μ−σ, μ_w−σ_w e KDE).\n\n"
            "Selecione o material para preencher automaticamente os limites — "
            "você pode ajustar os valores manualmente se necessário."
        )

        # ── Inicializar session_state para bounds ──
        for cam in todas_camadas:
            key_mat  = f"mat_{cam}"
            key_min  = f"min_{cam}"
            key_max  = f"max_{cam}"
            key_prev = f"mat_prev_{cam}"

            # Na primeira carga, inicializar com o primeiro material da lista
            if key_mat not in st.session_state:
                primeiro = list(MATERIAIS_CAMADA[cam].keys())[0]
                st.session_state[key_mat]  = primeiro
                mr_min0, mr_max0 = MATERIAIS_CAMADA[cam][primeiro]
                st.session_state[key_min]  = mr_min0
                st.session_state[key_max]  = mr_max0
                st.session_state[key_prev] = primeiro

            # Se o material mudou, atualizar os bounds automaticamente
            mat_atual = st.session_state[key_mat]
            if mat_atual != st.session_state.get(key_prev):
                mr_min_novo, mr_max_novo = MATERIAIS_CAMADA[cam][mat_atual]
                st.session_state[key_min]  = mr_min_novo
                st.session_state[key_max]  = mr_max_novo
                st.session_state[key_prev] = mat_atual

        bounds_interface = {}
        for cam in todas_camadas:
            st.markdown(f"**{cam}**")
            opcoes_material = list(MATERIAIS_CAMADA.get(cam, {}).keys())
            col_mat, col_min, col_max = st.columns([3, 1, 1])

            with col_mat:
                st.selectbox(
                    f"Material — {cam}",
                    options=opcoes_material,
                    key=f"mat_{cam}",
                    label_visibility="collapsed"
                )

            with col_min:
                st.number_input(
                    f"MR mín. {cam} (MPa)",
                    min_value=0,
                    step=50,
                    key=f"min_{cam}",
                    label_visibility="collapsed",
                    help=f"MR mínimo para {cam} — editável manualmente"
                )
            with col_max:
                st.number_input(
                    f"MR máx. {cam} (MPa)",
                    min_value=1,
                    step=50,
                    key=f"max_{cam}",
                    label_visibility="collapsed",
                    help=f"MR máximo para {cam} — editável manualmente"
                )

            bounds_interface[cam] = (
                st.session_state[f"min_{cam}"],
                st.session_state[f"max_{cam}"]
            )

        # Mostrar resumo dos bounds definidos
        with st.expander("📋 Resumo dos intervalos definidos", expanded=False):
            df_bounds = pd.DataFrame([
                {"Camada": cam, "MR mínimo (MPa)": v[0], "MR máximo (MPa)": v[1]}
                for cam, v in bounds_interface.items()
            ])
            st.dataframe(df_bounds, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Nomes dos segmentos ──
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

        # ── Botão processar ──
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
                    n_original = len(df)

                    # ── FILTRO DE BOUNDS — PASSO 1 DO PIPELINE ──
                    bounds_arquivo = {
                        cam: bounds_interface[cam]
                        for cam in camadas
                        if cam in bounds_interface
                    }
                    df_filtrado, n_removidos, detalhes_filtro = aplicar_filtro_bounds(
                        df, camadas, bounds_arquivo
                    )

                    if len(df_filtrado) < 3:
                        erros.append(
                            f"**{nome}**: após filtro de bounds restaram apenas "
                            f"{len(df_filtrado)} pontos — insuficiente para análise."
                        )
                        continue

                    res = rodar_pipeline(df_filtrado, camadas, percentil=percentil)
                    res["n_original"]     = n_original
                    res["n_removidos"]    = n_removidos
                    res["detalhes_filtro"] = detalhes_filtro
                    resultados_todos[nome] = res

                except Exception as e:
                    erros.append(f"**{nome}**: {str(e)}")

            barra.progress(100, text="Concluído!")

            if erros:
                st.error("Erros nos seguintes arquivos:")
                for e in erros:
                    st.markdown(f"- {e}")

            if resultados_todos:

                st.markdown("---")
                st.subheader("📊 Resultados por Segmento")

                for nome, res in resultados_todos.items():
                    with st.expander(
                        f"📍 **{nome}** — {len(res['camadas'])} camadas",
                        expanded=True
                    ):
                        r       = res["resultados"]
                        camadas = res["camadas"]

                        # ── Relatório do filtro de bounds ──
                        n_orig = res["n_original"]
                        n_rem  = res["n_removidos"]
                        n_util = n_orig - n_rem

                        if n_rem == 0:
                            st.success(
                                f"✅ Filtro de bounds: todos os {n_orig} pontos "
                                f"estão dentro dos intervalos físicos admissíveis."
                            )
                        else:
                            st.warning(
                                f"⚠️ Filtro de bounds: **{n_rem} ponto(s) removido(s)** "
                                f"de {n_orig} ({n_rem/n_orig*100:.1f}%). "
                                f"Análise realizada com **{n_util} pontos**."
                            )
                            det = res["detalhes_filtro"]
                            df_det = pd.DataFrame([
                                {
                                    "Camada": cam,
                                    "MR mín. (MPa)": det[cam]["mr_min"],
                                    "MR máx. (MPa)": det[cam]["mr_max"],
                                    "Pontos removidos": det[cam]["n_removidos"]
                                }
                                for cam in camadas
                            ])
                            st.dataframe(df_det, use_container_width=True,
                                         hide_index=True)

                        st.markdown("---")

                        # ── Testes de normalidade ──
                        df_norm  = testar_normalidade(res["df"], camadas)
                        fig_norm = plotar_mapa_normalidade(df_norm, nome)
                        st.pyplot(fig_norm)
                        plt.close(fig_norm)

                        nao_normais   = df_norm[df_norm["Conclusão"] == "Não normal"]["Camada"].tolist()
                        inconclusivos = df_norm[df_norm["Conclusão"] == "Inconclusivo"]["Camada"].tolist()
                        if nao_normais:
                            st.warning(
                                f"⚠️ Distribuição **não normal** detectada em: "
                                f"**{', '.join(nao_normais)}**. "
                                f"O critério μ−σ pode não ser adequado para essas camadas."
                            )
                        if inconclusivos:
                            st.info(
                                f"ℹ️ Resultado **inconclusivo** em: "
                                f"**{', '.join(inconclusivos)}**. "
                                f"Adotar abordagem conservadora (não paramétrica)."
                            )

                        st.markdown("---")

                        # ── Métricas rápidas ──
                        cols_met = st.columns(len(camadas))
                        for j, cam in enumerate(camadas):
                            label = (f"E_rep Subleito (P{percentil})"
                                     if cam == "Subleito" else f"E_rep {cam}")
                            cols_met[j].metric(label, f"{r[cam]['E_rep']:.0f} MPa")

                        # ── Tabela detalhada ──
                        tabela = pd.DataFrame([
                            {
                                "Camada"                    : cam,
                                "n"                         : r[cam]["n_total"],
                                "μ (MPa)"                   : f"{r[cam]['mu']:.1f}",
                                "σ (MPa)"                   : f"{r[cam]['sigma']:.1f}",
                                "CV"                        : f"{r[cam]['cv']:.3f}",
                                f"E_rep P{percentil} (MPa)" : f"{r[cam]['E_rep']:.1f}",
                                "E_rep μ_w−σ_w (MPa)"       : f"{r[cam]['e_rep_ms_w']:.1f}",
                                "E_rep μ−σ (MPa)"           : f"{r[cam]['e_rep_ms']:.1f}",
                            }
                            for cam in camadas
                        ])
                        st.dataframe(tabela, use_container_width=True,
                                     hide_index=True)

                        # ── Figura ──
                        fig = gerar_figura(res, nome)
                        st.pyplot(fig)
                        plt.close(fig)

                # ── Comparativo ──
                st.markdown("---")
                st.subheader("🏆 Comparativo entre Segmentos — Todas as Camadas")

                dados_comp = []
                for nome, res in resultados_todos.items():
                    r   = res["resultados"]
                    cam = res["camadas"]
                    linha = {
                        "Segmento": nome,
                        "n original": res["n_original"],
                        "n removidos": res["n_removidos"],
                        "n utilizado": res["n_original"] - res["n_removidos"],
                    }
                    for c in cam:
                        linha[f"P{percentil} {c}"]  = f"{r[c]['E_rep']:.1f}"
                        linha[f"μ_w−σ_w {c}"]       = f"{r[c]['e_rep_ms_w']:.1f}"
                        linha[f"μ−σ {c}"]           = f"{r[c]['e_rep_ms']:.1f}"
                    dados_comp.append(linha)

                df_comp = pd.DataFrame(dados_comp)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)

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
