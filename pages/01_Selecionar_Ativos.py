"""
Página 1: Seleção de Ativos
Permite filtrar e selecionar ativos da B3 para análise
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data, filters, ui, utils
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Selecionar Ativos - Portfolio B3",
    page_icon="🎯",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    utils.ensure_session_state_initialized()


def load_universe():
    """Carrega universo de ativos."""
    try:
        universe_df = data.load_ticker_universe()
        
        if universe_df.empty:
            st.error("❌ Erro ao carregar universo. Verifique b3_universe.csv")
            return pd.DataFrame()
        
        st.session_state.universe_df = universe_df
        logger.info(f"Universo carregado: {len(universe_df)} ativos")
        return universe_df
    
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        st.error(f"❌ Erro: {e}")
        return pd.DataFrame()


def apply_liquidity_filter():
    """Aplica filtro de liquidez."""
    
    st.markdown("### 💧 Filtro de Liquidez")
    st.markdown("Verificando ativos negociados nos últimos 30 dias via yfinance...")
    
    universe_df = st.session_state.universe_df
    
    if universe_df.empty:
        st.warning("⚠️ Carregue o universo primeiro")
        return
    
    # Explicação
    with st.expander("ℹ️ Como interpretar o volume?", expanded=False):
        st.markdown("""
        Volume = **número de ações negociadas por dia**
        
        **Referências:**
        - **Muito Baixa**: < 100.000 ações/dia
        - **Baixa**: 100.000 - 1.000.000 ações/dia
        - **Média**: 1.000.000 - 10.000.000 ações/dia
        - **Alta**: 10.000.000 - 50.000.000 ações/dia
        - **Muito Alta**: > 50.000.000 ações/dia
        
        **Exemplos:**
        - PETR4, VALE3: 100-500 milhões/dia
        - Médias: 1-10 milhões/dia
        - Small caps: < 1 milhão/dia
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_sessions = st.number_input(
            "Sessões mínimas (30d):",
            min_value=1,
            max_value=30,
            value=5,
            help="Mínimo de dias com negociação"
        )
    
    with col2:
        liquidity_level = st.selectbox(
            "Nível de liquidez:",
            [
                "Muito Baixa (> 10.000)",
                "Baixa (> 100.000)",
                "Média (> 1.000.000)",
                "Alta (> 10.000.000)",
                "Muito Alta (> 50.000.000)",
                "Personalizado"
            ],
            index=1
        )
        
        liquidity_map = {
            "Muito Baixa (> 10.000)": 10000,
            "Baixa (> 100.000)": 100000,
            "Média (> 1.000.000)": 1000000,
            "Alta (> 10.000.000)": 10000000,
            "Muito Alta (> 50.000.000)": 50000000,
        }
        
        if liquidity_level == "Personalizado":
            min_volume = st.number_input(
                "Volume mínimo (ações/dia):",
                min_value=1000,
                max_value=1000000000,
                value=100000,
                step=10000,
                format="%d"
            )
        else:
            min_volume = liquidity_map[liquidity_level]
            st.info(f"📊 Volume: **{min_volume:,.0f}** ações/dia")
    
    # Botões
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Usar Todos (Sem Filtro)", use_container_width=True):
            universe_df['is_traded_30d'] = True
            universe_df['avg_volume_30d'] = 1000000
            universe_df['sessions_traded_30d'] = 20
            
            st.session_state.filtered_universe_df = universe_df
            st.session_state.liquidity_applied = True
            
            st.success(f"✅ {len(universe_df)} ativos disponíveis")
            st.info("ℹ️ Liquidez não verificada")
            st.rerun()
    
    with col2:
        if st.button("🔍 Verificar Liquidez", use_container_width=True, type="primary"):
            
            filtered_df = data.filter_traded_last_30d(
                universe_df,
                min_sessions=min_sessions,
                min_avg_volume=min_volume,
                show_progress=True
            )
            
            traded_df = filtered_df[filtered_df['is_traded_30d'] == True].copy()
            
            st.session_state.filtered_universe_df = traded_df
            st.session_state.liquidity_applied = True
            
            # Estatísticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total", len(universe_df))
            
            with col2:
                st.metric("Líquidos", len(traded_df))
            
            with col3:
                pct = (len(traded_df) / len(universe_df) * 100) if len(universe_df) > 0 else 0
                st.metric("% Aprovado", f"{pct:.1f}%")
            
            with col4:
                if len(traded_df) > 0:
                    avg_vol = traded_df['avg_volume_30d'].mean()
                    st.metric("Vol. Médio", f"{avg_vol/1e6:.1f}M")
            
            if len(traded_df) > 0:
                st.success(f"✅ {len(traded_df)} ativos líquidos!")
                
                with st.expander("🔥 Top 10 Mais Líquidos"):
                    top10 = traded_df.nlargest(10, 'avg_volume_30d')[
                        ['ticker', 'nome', 'avg_volume_30d', 'sessions_traded_30d']
                    ].copy()
                    
                    top10['avg_volume_30d'] = top10['avg_volume_30d'].apply(
                        lambda x: f"{x/1e6:.2f}M"
                    )
                    
                    top10.columns = ['Ticker', 'Nome', 'Volume', 'Sessões']
                    st.dataframe(top10, use_container_width=True)
            else:
                st.warning("⚠️ Nenhum ativo atende aos critérios")


def show_simple_filters():
    """Interface de filtros."""
    
    st.markdown("### 🎯 Filtros")
    
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
        st.info(f"📊 {len(working_df)} ativos filtrados")
    elif not st.session_state.universe_df.empty:
        working_df = st.session_state.universe_df
        st.warning("⚠️ Usando universo completo")
    else:
        st.error("❌ Sem dados")
        return
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Por Setor**")
        all_sectors = sorted(working_df['setor'].unique().tolist())
        selected_sectors = st.multiselect(
            "Setores:",
            options=all_sectors,
            default=[]
        )
    
    with col2:
        st.markdown("**Por Tipo**")
        all_types = sorted(working_df['tipo'].unique().tolist())
        selected_types = st.multiselect(
            "Tipos:",
            options=all_types,
            default=[]
        )
    
    # Aplicar
    filtered = working_df.copy()
    
    if selected_sectors:
        filtered = filtered[filtered['setor'].isin(selected_sectors)]
    
    if selected_types:
        filtered = filtered[filtered['tipo'].isin(selected_types)]
    
    # Busca
    search = st.text_input(
        "🔍 Buscar:",
        placeholder="Ticker ou nome..."
    )
    
    if search:
        search_upper = search.upper()
        mask = (
            filtered['ticker'].str.contains(search_upper, case=False, na=False) |
            filtered['nome'].str.contains(search_upper, case=False, na=False)
        )
        filtered = filtered[mask]
    
    # Resultados
    st.markdown(f"### 📋 Disponíveis ({len(filtered)})")
    
    if filtered.empty:
        st.warning("⚠️ Nenhum ativo encontrado")
        return
    
    # Tabela
    display_df = filtered[['ticker', 'nome', 'setor', 'tipo']].copy()
    display_df.columns = ['Ticker', 'Nome', 'Setor', 'Tipo']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Ações
    st.markdown("### ✅ Ações")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Todos", use_container_width=True):
            st.session_state.selected_tickers = filtered['ticker'].tolist()
            st.success(f"✅ {len(st.session_state.selected_tickers)} selecionados!")
            st.rerun()
    
    with col2:
        if st.button("🔥 Top 20", use_container_width=True):
            if 'avg_volume_30d' in filtered.columns:
                top = filtered.nlargest(20, 'avg_volume_30d')
                st.session_state.selected_tickers = top['ticker'].tolist()
                st.success(f"✅ Top 20 selecionados!")
                st.rerun()
            else:
                st.warning("⚠️ Aplique filtro de liquidez")
    
    with col3:
        if st.button("🎲 10 Aleatórios", use_container_width=True):
            sample_size = min(10, len(filtered))
            random = filtered.sample(n=sample_size)
            st.session_state.selected_tickers = random['ticker'].tolist()
            st.success(f"✅ {sample_size} aleatórios!")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.selected_tickers = []
            st.success("✅ Limpo!")
            st.rerun()


def show_manual_selection():
    """Seleção manual."""
    
    st.markdown("### ✍️ Seleção Manual")
    
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
    elif not st.session_state.universe_df.empty:
        working_df = st.session_state.universe_df
    else:
        st.error("❌ Sem dados")
        return
    
    available = sorted(working_df['ticker'].tolist())
    
    # Criar opções
    options = []
    ticker_map = {}
    
    for ticker in available:
        nome = working_df[working_df['ticker'] == ticker]['nome'].iloc[0]
        option = f"{ticker} - {nome}"
        options.append(option)
        ticker_map[option] = ticker
    
    # Pré-selecionar
    defaults = []
    for ticker in st.session_state.selected_tickers:
        if ticker in working_df['ticker'].values:
            nome = working_df[working_df['ticker'] == ticker]['nome'].iloc[0]
            defaults.append(f"{ticker} - {nome}")
    
    selected_options = st.multiselect(
        "Selecione:",
        options=options,
        default=defaults
    )
    
    selected_tickers = [ticker_map[opt] for opt in selected_options]
    
    st.info(f"📊 {len(selected_tickers)} selecionados")
    
    if st.button("💾 Salvar", use_container_width=True, type="primary"):
        st.session_state.selected_tickers = selected_tickers
        st.success(f"✅ {len(selected_tickers)} salvos!")
        st.rerun()


def show_current_selection():
    """Mostra seleção atual."""
    
    st.markdown("### ✅ Seleção Atual")
    
    if not st.session_state.selected_tickers:
        st.info("ℹ️ Nenhum ativo selecionado")
        return
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total", len(st.session_state.selected_tickers))
    
    with col2:
        if not st.session_state.universe_df.empty:
            selected_df = st.session_state.universe_df[
                st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
            ]
            st.metric("Setores", selected_df['setor'].nunique())
    
    with col3:
        if not st.session_state.universe_df.empty:
            conc = filters.get_sector_concentration(
                st.session_state.selected_tickers,
                st.session_state.universe_df
            )
            if conc:
                st.metric("Maior Conc.", f"{max(conc.values()):.1f}%")
    
    # Validação
    if not st.session_state.universe_df.empty:
        is_valid, conc = filters.validate_sector_diversification(
            st.session_state.selected_tickers,
            st.session_state.universe_df,
            40.0
        )
        
        if not is_valid:
            max_sector = max(conc, key=conc.get)
            st.warning(f"⚠️ {max_sector}: {conc[max_sector]:.1f}% (> 40%)")
    
    # Lista
    st.markdown("**Ativos:**")
    
    if not st.session_state.universe_df.empty:
        selected_df = st.session_state.universe_df[
            st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
        ][['ticker', 'nome', 'setor', 'tipo']].copy()
        
        selected_df.columns = ['Ticker', 'Nome', 'Setor', 'Tipo']
        st.dataframe(selected_df, use_container_width=True, height=300)
    
    # Ações
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_text = "\n".join(st.session_state.selected_tickers)
        st.download_button(
            "📥 TXT",
            tickers_text,
            "tickers.txt",
            use_container_width=True
        )
    
    with col2:
        if not st.session_state.universe_df.empty:
            csv = st.session_state.universe_df[
                st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
            ].to_csv(index=False)
            
            st.download_button(
                "📥 CSV",
                csv,
                "tickers.csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("🗑️ Limpar", use_container_width=True):
            st.session_state.selected_tickers = []
            st.success("✅ Limpo!")
            st.rerun()


def show_statistics():
    """Estatísticas do universo."""
    
    st.markdown("### 📊 Estatísticas")
    
    universe_df = st.session_state.universe_df
    
    if universe_df.empty:
        st.info("ℹ️ Sem dados")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ativos", len(universe_df))
    
    with col2:
        st.metric("Setores", universe_df['setor'].nunique())
    
    with col3:
        st.metric("Subsetores", universe_df['subsetor'].nunique())
    
    with col4:
        st.metric("Segmentos", universe_df['segmento_listagem'].nunique())
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Por Setor")
        sector_counts = universe_df['setor'].value_counts()
        st.bar_chart(sector_counts)
    
    with col2:
        st.markdown("#### Por Tipo")
        type_counts = universe_df['tipo'].value_counts()
        st.bar_chart(type_counts)


def main():
    """Função principal."""
    
    initialize_session_state()
    
    st.markdown('<p class="gradient-title">🎯 Seleção de Ativos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Selecione os ativos da B3 para análise usando filtros ou seleção manual.
    """)
    
    # Carregar universo
    if st.session_state.universe_df.empty:
        with st.spinner("Carregando universo..."):
            universe_df = load_universe()
            if universe_df.empty:
                st.stop()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Filtros",
        "✍️ Manual",
        "✅ Seleção",
        "📊 Stats"
    ])
    
    with tab1:
        apply_liquidity_filter()
        st.markdown("---")
        
        if st.session_state.liquidity_applied or not st.session_state.universe_df.empty:
            show_simple_filters()
    
    with tab2:
        show_manual_selection()
    
    with tab3:
        show_current_selection()
    
    with tab4:
        show_statistics()
    
    # Próximos passos
    if st.session_state.selected_tickers:
        st.markdown("---")
        st.success(f"✅ {len(st.session_state.selected_tickers)} ativos prontos!")
        
        st.info("""
        **Continue:** Use o menu lateral (☰) para:
        - 💸 Análise de Dividendos
        - 📊 Portfólios Eficientes
        - 🎯 Sharpe e MinVol
        - 📋 Resumo Executivo
        """)
    
    st.markdown("---")
    st.caption("💡 Recomendado: 10-30 ativos para análise balanceada")


if __name__ == "__main__":
    main()
