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
warnings.filterwarnings('ignore')

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
# FUNÇÕES DO PIPELINE
# ─────────────────────────────────────────────────────────────
def carregar_arquivo(uploaded_file):
    """Carrega CSV ou XLSX e padroniza os nomes das colunas."""
    nome = uploaded_file.name.lower()
    if nome.endswith('.csv'):
        # tenta ponto e vírgula primeiro, depois vírgula
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            if df.shape[1] < 4:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',')
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [c.strip() for c in df.columns]

    # aceita qualquer ordem de colunas desde que tenha 4
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ['Revestimento', 'Base', 'Subleito', 'RMSE']
    else:
        raise ValueError("O arquivo deve ter pelo menos 4 colunas: Revestimento, Base, Subleito, RMSE.")

    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df


def rodar_pipeline(df, percentil=15):
    """Executa o pipeline GMM completo e retorna resultados."""
    camadas = ['Revestimento', 'Base', 'Subleito']

    # Pesos RMSE
    pesos = 1.0 / df['RMSE'].values
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
        g = GaussianMixture(n_components=k, covariance_type='full',
                            random_state=42, n_init=10)
        g.fit(X_resamp)
        bic.append(g.bic(X_resamp))
        aic.append(g.aic(X_resamp))
    k_otimo = int(np.argmin(bic)) + 1

    # Ajuste final
    gmm = GaussianMixture(n_components=k_otimo, covariance_type='full',
                          random_state=42, n_init=20)
    gmm.fit(X_resamp)
    labels = gmm.predict(X_scaled)

    # Parâmetros na escala original
    means = scaler.inverse_transform(gmm.means_)
    stds = np.zeros_like(means)
    for k in range(k_otimo):
        stds[k] = np.sqrt(np.diag(gmm.covariances_[k])) * scaler.scale_

    # Componente dominante (menor subleito)
    idx_sub = camadas.index('Subleito')
    comp_dom = int(np.argmin(means[:, idx_sub]))

    # z pelo percentil
    z = norm.ppf(1 - percentil / 100)

    # E_rep
    resultados = {}
    for i, cam in enumerate(camadas):
        mu = means[comp_dom, i]
        sigma = stds[comp_dom, i]
        cv = sigma / mu
        E_rep = max(mu - z * sigma, 0)
        resultados[cam] = {
            'mu': mu, 'sigma': sigma, 'cv': cv,
            'E_rep': E_rep, 'percentil': percentil
        }

    return {
        'df': df,
        'pesos': pesos,
        'camadas': camadas,
        'k_otimo': k_otimo,
        'bic': bic,
        'aic': aic,
        'gmm': gmm,
        'labels': labels,
        'means': means,
        'stds': stds,
        'comp_dom': comp_dom,
        'resultados': resultados,
        'z': z,
        'percentil': percentil,
        'scaler': scaler,
    }


def gerar_figura(res, nome_segmento):
    """Gera figura com 4 painéis: distribuições, BIC, scatter e scatter2."""
    cores = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    df = res['df']
    camadas = res['camadas']
    k_otimo = res['k_otimo']
    gmm = res['gmm']
    means = res['means']
    stds = res['stds']
    labels = res['labels']
    pesos = res['pesos']
    resultados = res['resultados']
    bic = res['bic']
    aic = res['aic']
    comp_dom = res['comp_dom']
    pesos_comp = gmm.weights_

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Segmento: {nome_segmento}  |  K={k_otimo}  |  P{res["percentil"]}',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # Painéis 0-2: distribuição por camada
    for i, cam in enumerate(camadas):
        ax = fig.add_subplot(gs[0, i])
        dados = df[cam].values
        kde = gaussian_kde(dados, weights=pesos, bw_method='silverman')
        x_r = np.linspace(dados.min() * 0.85, dados.max() * 1.10, 500)
        ax.plot(x_r, kde(x_r), 'k-', lw=2, label='KDE', zorder=5)
        for k in range(k_otimo):
            y_k = pesos_comp[k] * norm.pdf(x_r, means[k, i], stds[k, i])
            lw = 2.5 if k == comp_dom else 1.2
            ax.fill_between(x_r, y_k, alpha=0.25, color=cores[k])
            ax.plot(x_r, y_k, color=cores[k], lw=lw,
                    label=f'C{k+1}{"★" if k==comp_dom else ""} (w={pesos_comp[k]:.2f})')
        r = resultados[cam]
        ax.axvline(r['E_rep'], color='red', lw=2, ls='--',
                   label=f"E_rep={r['E_rep']:.0f}")
        ax.axvline(r['mu'], color='navy', lw=1.5, ls=':',
                   label=f"μ={r['mu']:.0f}")
        ax.set_title(cam, fontweight='bold')
        ax.set_xlabel('Módulo (MPa)')
        ax.set_ylabel('Densidade')
        ax.legend(fontsize=6.5)

    # Painel 3: BIC/AIC
    ax_bic = fig.add_subplot(gs[0, 3])
    k_range = list(range(1, 6))
    ax_bic.plot(k_range, bic, 'bo-', lw=2, label='BIC')
    ax_bic.plot(k_range, aic, 'rs--', lw=2, label='AIC')
    ax_bic.axvline(k_otimo, color='green', lw=2, ls=':', label=f'K={k_otimo}')
    ax_bic.set_xlabel('K')
    ax_bic.set_ylabel('Score')
    ax_bic.set_title('Seleção de K — BIC/AIC', fontweight='bold')
    ax_bic.legend()

    # Painel 4: scatter Base vs Subleito
    ax_s1 = fig.add_subplot(gs[1, 0])
    for k in range(k_otimo):
        m = labels == k
        ax_s1.scatter(df.loc[m, 'Base'], df.loc[m, 'Subleito'],
                      c=cores[k], label=f'C{k+1}', s=60,
                      alpha=0.8, edgecolors='k', lw=0.5,
                      zorder=3 if k == comp_dom else 2)
    ax_s1.set_xlabel('Base (MPa)')
    ax_s1.set_ylabel('Subleito (MPa)')
    ax_s1.set_title('Base vs Subleito', fontweight='bold')
    ax_s1.legend(fontsize=8)

    # Painel 5: scatter Revestimento vs Subleito
    ax_s2 = fig.add_subplot(gs[1, 1])
    for k in range(k_otimo):
        m = labels == k
        ax_s2.scatter(df.loc[m, 'Revestimento'], df.loc[m, 'Subleito'],
                      c=cores[k], label=f'C{k+1}', s=60,
                      alpha=0.8, edgecolors='k', lw=0.5)
    ax_s2.set_xlabel('Revestimento (MPa)')
    ax_s2.set_ylabel('Subleito (MPa)')
    ax_s2.set_title('Revestimento vs Subleito', fontweight='bold')
    ax_s2.legend(fontsize=8)

    # Painel 6: subleito ao longo dos pontos × RMSE
    ax_rmse = fig.add_subplot(gs[1, 2])
    sc = ax_rmse.scatter(range(len(df)), df['Subleito'],
                         c=df['RMSE'], cmap='RdYlGn_r',
                         s=60, edgecolors='k', lw=0.5)
    plt.colorbar(sc, ax=ax_rmse, label='RMSE')
    ax_rmse.axhline(resultados['Subleito']['E_rep'], color='red',
                    lw=2, ls='--',
                    label=f"E_rep={resultados['Subleito']['E_rep']:.0f}")
    ax_rmse.set_xlabel('Ponto')
    ax_rmse.set_ylabel('Subleito (MPa)')
    ax_rmse.set_title('Subleito × RMSE', fontweight='bold')
    ax_rmse.legend(fontsize=8)

    # Painel 7: tabela resumo
    ax_tab = fig.add_subplot(gs[1, 3])
    ax_tab.axis('off')
    td = [[cam,
           f"{resultados[cam]['mu']:.0f}",
           f"{resultados[cam]['sigma']:.0f}",
           f"{resultados[cam]['cv']:.3f}",
           f"{resultados[cam]['E_rep']:.0f}"]
          for cam in camadas]
    tbl = ax_tab.table(
        cellText=td,
        colLabels=['Camada', 'μ', 'σ', 'CV', 'E_rep'],
        cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.9)
    # destacar subleito
    for col in range(5):
        tbl[3, col].set_facecolor('#FFCDD2')
    ax_tab.set_title(f'E_rep por camada (P{res["percentil"]})', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def gerar_pdf(resultados_todos, percentil):
    """Gera PDF consolidado com todos os segmentos e tabela comparativa."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ── Capa ──
        fig_capa, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.65,
                'Relatório de Análise GMM\nPavimentos Aeroportuários',
                ha='center', va='center', fontsize=22,
                fontweight='bold', color='#1F497D',
                transform=ax.transAxes)
        ax.text(0.5, 0.50,
                f'Módulo Representativo — Percentil {percentil}',
                ha='center', va='center', fontsize=14,
                color='#555555', transform=ax.transAxes)
        ax.text(0.5, 0.40,
                f'Total de segmentos analisados: {len(resultados_todos)}',
                ha='center', va='center', fontsize=12,
                color='#333333', transform=ax.transAxes)
        pdf.savefig(fig_capa, bbox_inches='tight')
        plt.close(fig_capa)

        # ── Figura de cada segmento ──
        for nome, res in resultados_todos.items():
            fig = gerar_figura(res, nome)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── Tabela comparativa final ──
        fig_tab, ax_tab = plt.subplots(figsize=(11, 4 + len(resultados_todos) * 0.5))
        ax_tab.axis('off')
        ax_tab.set_title('Tabela Comparativa — Módulos Representativos por Segmento',
                         fontsize=13, fontweight='bold', color='#1F497D', pad=20)

        headers = ['Segmento', 'K', 'C. Dom.',
                   'μ Sub. (MPa)', 'σ Sub. (MPa)', 'CV Sub.',
                   'E_rep Sub. (MPa)', 'E_rep Base (MPa)', 'E_rep Rev. (MPa)']
        rows = []
        erep_subleito = []
        for nome, res in resultados_todos.items():
            r_sub = res['resultados']['Subleito']
            r_bas = res['resultados']['Base']
            r_rev = res['resultados']['Revestimento']
            rows.append([
                nome,
                str(res['k_otimo']),
                f"C{res['comp_dom']+1} (π={res['gmm'].weights_[res['comp_dom']]:.2f})",
                f"{r_sub['mu']:.1f}",
                f"{r_sub['sigma']:.1f}",
                f"{r_sub['cv']:.3f}",
                f"{r_sub['E_rep']:.1f}",
                f"{r_bas['E_rep']:.1f}",
                f"{r_rev['E_rep']:.1f}",
            ])
            erep_subleito.append(r_sub['E_rep'])

        idx_critico = int(np.argmin(erep_subleito))

        tbl = ax_tab.table(
            cellText=rows,
            colLabels=headers,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 2.0)

        # cabeçalho azul
        for col in range(len(headers)):
            tbl[0, col].set_facecolor('#1F497D')
            tbl[0, col].get_text().set_color('white')
            tbl[0, col].get_text().set_fontweight('bold')

        # segmento crítico em vermelho
        for col in range(len(headers)):
            tbl[idx_critico + 1, col].set_facecolor('#FFCDD2')
            tbl[idx_critico + 1, col].get_text().set_fontweight('bold')

        # linhas alternadas
        for row in range(1, len(rows) + 1):
            if row != idx_critico + 1:
                fill = '#EBF3FB' if row % 2 == 0 else 'white'
                for col in range(len(headers)):
                    tbl[row, col].set_facecolor(fill)

        nome_critico = list(resultados_todos.keys())[idx_critico]
        erep_critico = erep_subleito[idx_critico]
        ax_tab.text(0.5, 0.02,
                    f'★ Segmento crítico: {nome_critico}  |  '
                    f'E_rep subleito = {erep_critico:.1f} MPa  |  '
                    f'Percentil adotado: P{percentil}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#C0392B',
                    transform=ax_tab.transAxes)

        pdf.savefig(fig_tab, bbox_inches='tight')
        plt.close(fig_tab)

    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────────────────────

# Sidebar — configurações
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
        "**Formato esperado dos arquivos:**\n\n"
        "4 colunas na ordem:\n"
        "`Revestimento | Base | Subleito | RMSE`\n\n"
        "Aceita `.csv` (`;` ou `,`) e `.xlsx`"
    )
    st.markdown("---")
    st.caption("Metodologia: GMM multivariado + reamostragem ponderada pelo RMSE")

# Upload dos arquivos
st.subheader("📂 Upload dos Segmentos Homogêneos")
st.markdown(
    "Faça o upload de **um arquivo por segmento**. "
    "O nome do arquivo será usado como nome do segmento (editável abaixo)."
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
        nome_padrao = f.name.rsplit('.', 1)[0]  # remove extensão
        with cols[i % 3]:
            nome_editado = st.text_input(
                f"Arquivo: `{f.name}`",
                value=nome_padrao,
                key=f"nome_{i}"
            )
            nomes_segmentos[i] = nome_editado

    st.markdown("---")

    if st.button("🚀 Processar segmentos", type="primary", use_container_width=True):

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
                df = carregar_arquivo(f)
                res = rodar_pipeline(df, percentil=percentil)
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
                with st.expander(f"📍 Segmento: **{nome}**", expanded=True):

                    # métricas rápidas
                    r = res['resultados']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("K ótimo (BIC)", res['k_otimo'])
                    c2.metric("E_rep Revestimento", f"{r['Revestimento']['E_rep']:.0f} MPa")
                    c3.metric("E_rep Base", f"{r['Base']['E_rep']:.0f} MPa")
                    c4.metric(f"E_rep Subleito (P{percentil})",
                              f"{r['Subleito']['E_rep']:.0f} MPa",
                              delta=f"μ = {r['Subleito']['mu']:.0f} MPa",
                              delta_color="off")

                    # tabela detalhada
                    tabela = pd.DataFrame([
                        {
                            'Camada': cam,
                            'μ (MPa)': f"{r[cam]['mu']:.1f}",
                            'σ (MPa)': f"{r[cam]['sigma']:.1f}",
                            'CV': f"{r[cam]['cv']:.3f}",
                            f'E_rep P{percentil} (MPa)': f"{r[cam]['E_rep']:.1f}"
                        }
                        for cam in res['camadas']
                    ])
                    st.dataframe(tabela, use_container_width=True, hide_index=True)

                    # figura
                    fig = gerar_figura(res, nome)
                    st.pyplot(fig)
                    plt.close(fig)

            # ── Tabela comparativa ──
            st.markdown("---")
            st.subheader("🏆 Comparativo entre Segmentos")

            dados_comp = []
            for nome, res in resultados_todos.items():
                r = res['resultados']
                dados_comp.append({
                    'Segmento': nome,
                    'K': res['k_otimo'],
                    'Comp. dominante': f"C{res['comp_dom']+1}",
                    'μ Sub. (MPa)': f"{r['Subleito']['mu']:.1f}",
                    'CV Sub.': f"{r['Subleito']['cv']:.3f}",
                    f'E_rep Sub. P{percentil} (MPa)': f"{r['Subleito']['E_rep']:.1f}",
                    f'E_rep Base P{percentil} (MPa)': f"{r['Base']['E_rep']:.1f}",
                    f'E_rep Rev. P{percentil} (MPa)': f"{r['Revestimento']['E_rep']:.1f}",
                })

            df_comp = pd.DataFrame(dados_comp)
            erep_vals = [res['resultados']['Subleito']['E_rep']
                         for res in resultados_todos.values()]
            idx_critico = int(np.argmin(erep_vals))
            nome_critico = list(resultados_todos.keys())[idx_critico]

            # highlight do segmento crítico
            def highlight_critico(row):
                if row['Segmento'] == nome_critico:
                    return ['background-color: #FFCDD2; font-weight: bold'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_comp.style.apply(highlight_critico, axis=1),
                use_container_width=True,
                hide_index=True
            )

            st.error(
                f"🔴 **Segmento crítico: {nome_critico}** — "
                f"E_rep subleito = **{erep_vals[idx_critico]:.1f} MPa** (P{percentil})\n\n"
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
