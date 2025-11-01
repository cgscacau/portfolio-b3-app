"""
Página 1: Seleção de Ativos
Permite filtrar e selecionar ativos da B3 para análise
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from core import data, filters, ui
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Selecionar Ativos - Portfolio B3",
    page_icon="🎯",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão se não existirem."""
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'universe_df' not in st.session_state:
        st.session_state.universe_df = pd.DataFrame()
    
    if 'filtered_universe_df' not in st.session_state:
        st.session_state.filtered_universe_df = pd.DataFrame()
    
    if 'liquidity_applied' not in st.session_state:
        st.session_state.liquidity_applied = False


def load_universe():
    """Carrega universo de ativos."""
    try:
        universe_df = data.load_ticker_universe()
        
        if universe_df.empty:
            st.error("❌ Erro ao carregar universo de ativos. Verifique o arquivo b3_universe.csv")
            return pd.DataFrame()
        
        st.session_state.universe_df = universe_df
        logger.info(f"Universo carregado: {len(universe_df)} ativos")
        return universe_df
    
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        st.error(f"❌ Erro ao carregar universo: {e}")
        return pd.DataFrame()


def apply_liquidity_filter():
    """Aplica filtro de liquidez."""
    
    st.markdown("### 💧 Filtro de Liquidez")
    st.markdown("Verificando ativos negociados nos últimos 30 dias...")
    
    universe_df = st.session_state.universe_df
    
    if universe_df.empty:
        st.warning("⚠️ Carregue o universo de ativos primeiro")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_sessions = st.number_input(
            "Sessões mínimas negociadas (30d):",
            min_value=1,
            max_value=30,
            value=5,
            help="Número mínimo de dias com negociação nos últimos 30 dias",
            key="liquidity_min_sessions"
        )
    
    with col2:
        min_volume = st.number_input(
            "Volume médio mínimo:",
            min_value=0,
            max_value=10000000,
            value=10000,
            step=10000,
            help="Volume médio diário mínimo",
            key="liquidity_min_volume"
        )
    
    if st.button("🔍 Aplicar Filtro de Liquidez", use_container_width=True, type="primary", key="apply_liquidity"):
        
        with st.spinner("Verificando liquidez dos ativos... Isso pode levar alguns minutos."):
            
            filtered_df = data.filter_traded_last_30d(
                universe_df,
                min_sessions=min_sessions,
                min_avg_volume=min_volume,
                show_progress=True
            )
            
            # Filtrar apenas os negociados
            traded_df = filtered_df[filtered_df['is_traded_30d'] == True].copy()
            
            st.session_state.filtered_universe_df = traded_df
            st.session_state.liquidity_applied = True
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total no Universo", len(universe_df))
            
            with col2:
                st.metric("Ativos Líquidos", len(traded_df))
            
            with col3:
                pct = (len(traded_df) / len(universe_df) * 100) if len(universe_df) > 0 else 0
                st.metric("% Aprovado", f"{pct:.1f}%")
            
            if len(traded_df) > 0:
                st.success(f"✅ {len(traded_df)} ativos líquidos identificados!")
            else:
                st.warning("⚠️ Nenhum ativo atende aos critérios de liquidez. Tente reduzir os limites.")


def show_simple_filters():
    """Interface simplificada de filtros."""
    
    st.markdown("### 🎯 Filtros de Seleção")
    
    # Usar universo filtrado se disponível, senão usar completo
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
        st.info(f"📊 Trabalhando com {len(working_df)} ativos líquidos")
    elif not st.session_state.universe_df.empty:
        working_df = st.session_state.universe_df
        st.warning("⚠️ Usando universo completo. Recomendamos aplicar o filtro de liquidez primeiro.")
    else:
        st.error("❌ Nenhum dado disponível")
        return
    
    # Filtros básicos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Filtro por Setor**")
        all_sectors = sorted(working_df['setor'].unique().tolist())
        selected_sectors = st.multiselect(
            "Selecione setores:",
            options=all_sectors,
            default=[],
            key="filter_sectors"
        )
    
    with col2:
        st.markdown("**Filtro por Tipo**")
        all_types = sorted(working_df['tipo'].unique().tolist())
        selected_types = st.multiselect(
            "Selecione tipos:",
            options=all_types,
            default=[],
            key="filter_types"
        )
    
    # Aplicar filtros
    filtered = working_df.copy()
    
    if selected_sectors:
        filtered = filtered[filtered['setor'].isin(selected_sectors)]
    
    if selected_types:
        filtered = filtered[filtered['tipo'].isin(selected_types)]
    
    # Busca por texto
    search = st.text_input(
        "🔍 Buscar por ticker ou nome:",
        placeholder="Ex: PETR, Petrobras...",
        key="search_text"
    )
    
    if search:
        search_upper = search.upper()
        mask = (
            filtered['ticker'].str.contains(search_upper, case=False, na=False) |
            filtered['nome'].str.contains(search_upper, case=False, na=False)
        )
        filtered = filtered[mask]
    
    # Exibir resultados
    st.markdown(f"### 📋 Ativos Disponíveis ({len(filtered)})")
    
    if filtered.empty:
        st.warning("⚠️ Nenhum ativo encontrado com os filtros aplicados")
        return
    
    # Tabela interativa
    display_cols = ['ticker', 'nome', 'setor', 'tipo']
    display_df = filtered[display_cols].copy()
    
    # Renomear
    display_df.columns = ['Ticker', 'Nome', 'Setor', 'Tipo']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Botões de seleção
    st.markdown("### ✅ Ações de Seleção")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Selecionar Todos", use_container_width=True, key="select_all"):
            st.session_state.selected_tickers = filtered['ticker'].tolist()
            st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados!")
            st.rerun()
    
    with col2:
        if st.button("🔥 Top 20 Liquidez", use_container_width=True, key="select_top20"):
            if 'avg_volume_30d' in filtered.columns:
                top = filtered.nlargest(20, 'avg_volume_30d')
                st.session_state.selected_tickers = top['ticker'].tolist()
                st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados!")
                st.rerun()
            else:
                st.warning("⚠️ Aplique o filtro de liquidez primeiro")
    
    with col3:
        if st.button("🎲 10 Aleatórios", use_container_width=True, key="select_random"):
            sample_size = min(10, len(filtered))
            random_sample = filtered.sample(n=sample_size)
            st.session_state.selected_tickers = random_sample['ticker'].tolist()
            st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados!")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Limpar", use_container_width=True, key="clear_selection"):
            st.session_state.selected_tickers = []
            st.success("✅ Seleção limpa!")
            st.rerun()


def show_manual_selection():
    """Seleção manual com multiselect."""
    
    st.markdown("### ✍️ Seleção Manual")
    
    # Usar universo apropriado
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
    elif not st.session_state.universe_df.empty:
        working_df = st.session_state.universe_df
    else:
        st.error("❌ Nenhum dado disponível")
        return
    
    available_tickers = sorted(working_df['ticker'].tolist())
    
    # Criar opções com nome
    ticker_options = []
    for ticker in available_tickers:
        nome = working_df[working_df['ticker'] == ticker]['nome'].iloc[0]
        ticker_options.append(f"{ticker} - {nome}")
    
    selected_options = st.multiselect(
        "Digite ou selecione tickers:",
        options=ticker_options,
        default=[f"{t} - {working_df[working_df['ticker'] == t]['nome'].iloc[0]}" 
                for t in st.session_state.selected_tickers 
                if t in working_df['ticker'].values],
        key="manual_select"
    )
    
    # Extrair apenas os tickers
    selected_tickers = [opt.split(' - ')[0] for opt in selected_options]
    
    if st.button("💾 Salvar Seleção Manual", use_container_width=True, type="primary", key="save_manual"):
        st.session_state.selected_tickers = selected_tickers
        st.success(f"✅ {len(selected_tickers)} ativos selecionados!")
        st.rerun()


def show_current_selection():
    """Mostra seleção atual."""
    
    st.markdown("### ✅ Seleção Atual")
    
    if not st.session_state.selected_tickers:
        st.info("ℹ️ Nenhum ativo selecionado ainda")
        return
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Selecionado", len(st.session_state.selected_tickers))
    
    with col2:
        # Calcular setores únicos
        if not st.session_state.universe_df.empty:
            selected_df = st.session_state.universe_df[
                st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
            ]
            unique_sectors = selected_df['setor'].nunique()
            st.metric("Setores Únicos", unique_sectors)
        else:
            st.metric("Setores Únicos", "N/A")
    
    with col3:
        if not st.session_state.universe_df.empty:
            concentration = filters.get_sector_concentration(
                st.session_state.selected_tickers,
                st.session_state.universe_df
            )
            if concentration:
                max_conc = max(concentration.values())
                st.metric("Maior Concentração", f"{max_conc:.1f}%")
            else:
                st.metric("Maior Concentração", "N/A")
        else:
            st.metric("Maior Concentração", "N/A")
    
    # Lista
    st.markdown("**Tickers Selecionados:**")
    
    if not st.session_state.universe_df.empty:
        selected_df = st.session_state.universe_df[
            st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
        ][['ticker', 'nome', 'setor', 'tipo']]
        
        selected_df.columns = ['Ticker', 'Nome', 'Setor', 'Tipo']
        st.dataframe(selected_df, use_container_width=True, height=300)
    else:
        st.write(", ".join(st.session_state.selected_tickers))
    
    # Exportar
    col1, col2 = st.columns(2)
    
    with col1:
        tickers_text = "\n".join(st.session_state.selected_tickers)
        st.download_button(
            "📥 Exportar Lista",
            tickers_text,
            "selected_tickers.txt",
            "text/plain",
            use_container_width=True
        )
    
    with col2:
        if st.button("🗑️ Limpar Tudo", use_container_width=True, key="clear_all"):
            st.session_state.selected_tickers = []
            st.success("✅ Seleção limpa!")
            st.rerun()


def main():
    """Função principal da página."""
    
    # Inicializar
    initialize_session_state()
    
    # Header
    st.markdown('<p class="gradient-title">🎯 Seleção de Ativos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Selecione os ativos da B3 que deseja analisar. Você pode usar filtros automáticos 
    por setor, liquidez e outros critérios, ou fazer seleção manual.
    """)
    
    # Carregar universo se necessário
    if st.session_state.universe_df.empty:
        with st.spinner("Carregando universo de ativos..."):
            universe_df = load_universe()
            
            if universe_df.empty:
                st.error("❌ Não foi possível carregar o universo de ativos")
                st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Filtros Automáticos",
        "✍️ Seleção Manual",
        "✅ Seleção Atual",
        "📊 Estatísticas"
    ])
    
    with tab1:
        # Filtro de liquidez
        apply_liquidity_filter()
        
        st.markdown("---")
        
        # Filtros adicionais
        if st.session_state.liquidity_applied or not st.session_state.universe_df.empty:
            show_simple_filters()
        else:
            st.info("ℹ️ Aplique o filtro de liquidez acima para começar")
    
    with tab2:
        show_manual_selection()
    
    with tab3:
        show_current_selection()
    
    with tab4:
        st.markdown("### 📊 Estatísticas do Universo")
        
        universe_df = st.session_state.universe_df
        
        if not universe_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Ativos", len(universe_df))
            
            with col2:
                st.metric("Setores Únicos", universe_df['setor'].nunique())
            
            with col3:
                st.metric("Subsetores", universe_df['subsetor'].nunique())
            
            with col4:
                st.metric("Segmentos", universe_df['segmento_listagem'].nunique())
            
            # Distribuição por setor
            st.markdown("#### Distribuição por Setor")
            sector_counts = universe_df['setor'].value_counts()
            st.bar_chart(sector_counts)
        else:
            st.info("ℹ️ Carregue os dados para ver as estatísticas")
    
    # Próximos passos
    if st.session_state.selected_tickers:
        st.markdown("---")
        st.markdown("### 🚀 Próximos Passos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("💸 Continue para **Análise de Dividendos** →")
        
        with col2:
            st.info("📊 Ou vá direto para **Portfólios Eficientes** →")
        
        with col3:
            st.info("📋 Ou veja o **Resumo Executivo** →")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem 0;">
            <p>💡 Dica: Selecione entre 10-30 ativos para uma análise balanceada entre diversificação e complexidade.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
