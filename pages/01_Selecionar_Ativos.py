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


def load_universe():
    """Carrega universo de ativos."""
    try:
        universe_df = data.load_ticker_universe()
        
        if universe_df.empty:
            st.error("❌ Erro ao carregar universo de ativos. Verifique o arquivo b3_universe.csv")
            return pd.DataFrame()
        
        st.session_state.universe_df = universe_df
        return universe_df
    
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        st.error(f"❌ Erro ao carregar universo: {e}")
        return pd.DataFrame()


def filter_by_liquidity(universe_df: pd.DataFrame):
    """Aplica filtro de liquidez (últimos 30 dias)."""
    
    ui.create_section_header(
        "💧 Filtro de Liquidez",
        "Verificando ativos negociados nos últimos 30 dias...",
        "💧"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_sessions = st.number_input(
            "Sessões mínimas negociadas (30d):",
            min_value=1,
            max_value=30,
            value=5,
            help="Número mínimo de dias com negociação nos últimos 30 dias"
        )
    
    with col2:
        min_volume = st.number_input(
            "Volume médio mínimo:",
            min_value=0,
            max_value=10000000,
            value=10000,
            step=10000,
            help="Volume médio diário mínimo"
        )
    
    if st.button("🔍 Aplicar Filtro de Liquidez", use_container_width=True, type="primary"):
        with st.spinner("Verificando liquidez dos ativos..."):
            filtered_df = data.filter_traded_last_30d(
                universe_df,
                min_sessions=min_sessions,
                min_avg_volume=min_volume,
                show_progress=True
            )
            
            # Filtrar apenas os negociados
            traded_df = filtered_df[filtered_df['is_traded_30d'] == True].copy()
            
            st.session_state.filtered_universe_df = traded_df
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ui.create_metric_card(
                    "Total no Universo",
                    f"{len(universe_df)}",
                    icon="📊"
                )
            
            with col2:
                ui.create_metric_card(
                    "Ativos Líquidos",
                    f"{len(traded_df)}",
                    icon="✅"
                )
            
            with col3:
                pct = (len(traded_df) / len(universe_df) * 100) if len(universe_df) > 0 else 0
                ui.create_metric_card(
                    "% Aprovado",
                    f"{pct:.1f}%",
                    icon="📈"
                )
            
            if len(traded_df) > 0:
                st.success(f"✅ {len(traded_df)} ativos líquidos identificados!")
            else:
                st.warning("⚠️ Nenhum ativo atende aos critérios de liquidez. Tente reduzir os limites.")
    
    return st.session_state.filtered_universe_df


def show_filter_interface(universe_df: pd.DataFrame):
    """Exibe interface de filtros."""
    
    ui.create_section_header(
        "🎯 Filtros de Seleção",
        "Use os filtros abaixo para refinar sua seleção de ativos",
        "🎯"
    )
    
    # Criar filtro
    asset_filter = filters.create_filter_ui(universe_df, key_prefix="page1")
    
    return asset_filter


def show_selection_summary(asset_filter: filters.AssetFilter):
    """Exibe resumo da seleção."""
    
    filtered_df = asset_filter.get_filtered_dataframe()
    
    if filtered_df.empty:
        ui.create_info_box(
            "Nenhum ativo selecionado. Use os filtros acima para selecionar ativos.",
            "warning"
        )
        return
    
    ui.create_section_header(
        "📋 Ativos Selecionados",
        f"{len(filtered_df)} ativos disponíveis para análise",
        "📋"
    )
    
    # Estatísticas por setor
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribuição por Setor")
        sector_dist = filters.create_sector_distribution(filtered_df)
        
        if not sector_dist.empty:
            fig = ui.plot_portfolio_weights(
                dict(zip(sector_dist['setor'], sector_dist['count'] / sector_dist['count'].sum())),
                title="Distribuição por Setor",
                show_percentage=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Distribuição por Segmento")
        segment_dist = filters.create_segment_distribution(filtered_df)
        
        if not segment_dist.empty:
            fig = ui.plot_portfolio_weights(
                dict(zip(segment_dist['segmento'], segment_dist['count'] / segment_dist['count'].sum())),
                title="Distribuição por Segmento",
                show_percentage=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de ativos
    st.markdown("### 📑 Lista de Ativos")
    
    # Preparar colunas para exibição
    display_cols = ['ticker', 'nome', 'setor', 'subsetor', 'segmento_listagem', 'tipo']
    
    if 'avg_volume_30d' in filtered_df.columns:
        display_cols.append('avg_volume_30d')
        display_cols.append('sessions_traded_30d')
    
    display_df = filtered_df[display_cols].copy()
    
    # Formatar volume
    if 'avg_volume_30d' in display_df.columns:
        display_df['avg_volume_30d'] = display_df['avg_volume_30d'].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
        )
    
    # Renomear colunas
    rename_map = {
        'ticker': 'Ticker',
        'nome': 'Nome',
        'setor': 'Setor',
        'subsetor': 'Subsetor',
        'segmento_listagem': 'Segmento',
        'tipo': 'Tipo',
        'avg_volume_30d': 'Volume Médio (30d)',
        'sessions_traded_30d': 'Sessões Negociadas'
    }
    
    display_df = display_df.rename(columns=rename_map)
    
    # Exibir com busca
    search_term = st.text_input(
        "🔍 Buscar na tabela:",
        placeholder="Digite ticker ou nome...",
        key="table_search"
    )
    
    if search_term:
        mask = (
            display_df['Ticker'].str.contains(search_term, case=False, na=False) |
            display_df['Nome'].str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Botões de ação
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Selecionar Todos", use_container_width=True):
            st.session_state.selected_tickers = filtered_df['ticker'].tolist()
            st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados!")
            st.rerun()
    
    with col2:
        if st.button("🔥 Selecionar Top 20 Liquidez", use_container_width=True):
            top_tickers = filters.get_top_liquid_tickers(filtered_df, 20)
            st.session_state.selected_tickers = top_tickers
            st.success(f"✅ {len(top_tickers)} ativos mais líquidos selecionados!")
            st.rerun()
    
    with col3:
        if st.button("🎲 Seleção Diversificada", use_container_width=True):
            diversified = filters.get_diversified_selection(filtered_df, n_per_sector=3)
            st.session_state.selected_tickers = diversified
            st.success(f"✅ {len(diversified)} ativos diversificados selecionados!")
            st.rerun()


def show_manual_selection(universe_df: pd.DataFrame):
    """Interface para seleção manual de ativos."""
    
    ui.create_section_header(
        "✍️ Seleção Manual",
        "Selecione ativos específicos manualmente",
        "✍️"
    )
    
    # Multiselect
    available_tickers = universe_df['ticker'].tolist()
    
    selected = st.multiselect(
        "Selecione os tickers:",
        options=available_tickers,
        default=st.session_state.selected_tickers,
        help="Digite ou selecione tickers da lista"
    )
    
    if st.button("💾 Salvar Seleção Manual", use_container_width=True, type="primary"):
        st.session_state.selected_tickers = selected
        st.success(f"✅ {len(selected)} ativos selecionados manualmente!")
        st.rerun()


def show_current_selection():
    """Exibe seleção atual."""
    
    if not st.session_state.selected_tickers:
        ui.create_info_box(
            "Nenhum ativo selecionado ainda. Use os filtros ou seleção manual acima.",
            "info"
        )
        return
    
    ui.create_section_header(
        "✅ Seleção Atual",
        f"{len(st.session_state.selected_tickers)} ativos prontos para análise",
        "✅"
    )
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_metric_card(
            "Total Selecionado",
            f"{len(st.session_state.selected_tickers)}",
            icon="📊"
        )
    
    # Análise de concentração setorial
    if not st.session_state.universe_df.empty:
        concentration = filters.get_sector_concentration(
            st.session_state.selected_tickers,
            st.session_state.universe_df
        )
        
        if concentration:
            max_sector = max(concentration, key=concentration.get)
            max_pct = concentration[max_sector]
            
            with col2:
                ui.create_metric_card(
                    "Setores Únicos",
                    f"{len(concentration)}",
                    icon="🏢"
                )
            
            with col3:
                ui.create_metric_card(
                    "Maior Concentração",
                    f"{max_pct:.1f}%",
                    help_text=f"Setor: {max_sector}",
                    icon="⚠️"
                )
            
            # Validar diversificação
            is_valid, _ = filters.validate_sector_diversification(
                st.session_state.selected_tickers,
                st.session_state.universe_df,
                max_sector_pct=40.0
            )
            
            with col4:
                status = "✅ OK" if is_valid else "⚠️ Alerta"
                color = "success" if is_valid else "warning"
                ui.create_metric_card(
                    "Diversificação",
                    status,
                    help_text="Limite: 40% por setor",
                    icon="🎯"
                )
            
            if not is_valid:
                ui.create_info_box(
                    f"⚠️ Concentração setorial acima de 40% ({max_sector}: {max_pct:.1f}%). "
                    "Considere diversificar para reduzir risco idiossincrático.",
                    "warning"
                )
    
    # Lista de tickers selecionados
    st.markdown("### 📝 Tickers Selecionados")
    
    # Criar DataFrame com informações
    if not st.session_state.universe_df.empty:
        selected_info = st.session_state.universe_df[
            st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
        ][['ticker', 'nome', 'setor', 'tipo']].copy()
        
        st.dataframe(selected_info, use_container_width=True, height=300)
    else:
        # Apenas lista simples
        tickers_text = ", ".join(st.session_state.selected_tickers)
        st.text_area("Tickers:", tickers_text, height=100, disabled=True)
    
    # Ações
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Limpar Seleção", use_container_width=True):
            st.session_state.selected_tickers = []
            st.success("✅ Seleção limpa!")
            st.rerun()
    
    with col2:
        # Exportar lista
        if st.session_state.selected_tickers:
            tickers_csv = "\n".join(st.session_state.selected_tickers)
            ui.create_download_button(
                tickers_csv,
                "selected_tickers.txt",
                "📥 Exportar Lista",
                "txt"
            )
    
    with col3:
        # Sugerir ativos adicionais
        if st.button("💡 Sugerir Mais Ativos", use_container_width=True):
            if not st.session_state.universe_df.empty:
                suggestions = filters.suggest_additional_tickers(
                    st.session_state.selected_tickers,
                    st.session_state.universe_df,
                    target_count=len(st.session_state.selected_tickers) + 5
                )
                
                if suggestions:
                    st.info(f"💡 Sugestões para diversificação: {', '.join(suggestions)}")
                else:
                    st.info("Não há sugestões disponíveis no momento.")


def show_next_steps():
    """Exibe próximos passos."""
    
    if not st.session_state.selected_tickers:
        return
    
    st.markdown("---")
    
    ui.create_section_header(
        "🚀 Próximos Passos",
        "Continue para análise detalhada",
        "🚀"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💸 Análise de Dividendos", use_container_width=True, type="primary"):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")
    
    with col2:
        if st.button("📊 Portfólios Eficientes", use_container_width=True):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
    
    with col3:
        if st.button("📋 Resumo Executivo", use_container_width=True):
            st.switch_page("app/pages/05_Resumo_Executivo.py")


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
    
    # Carregar universo se ainda não carregado
    if st.session_state.universe_df.empty:
        with st.spinner("Carregando universo de ativos..."):
            universe_df = load_universe()
            
            if universe_df.empty:
                st.stop()
    else:
        universe_df = st.session_state.universe_df
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Filtros Automáticos",
        "✍️ Seleção Manual",
        "✅ Seleção Atual",
        "📊 Estatísticas"
    ])
    
    with tab1:
        # Filtro de liquidez primeiro
        filtered_df = filter_by_liquidity(universe_df)
        
        st.markdown("---")
        
        # Se temos ativos filtrados por liquidez, usar esses
        if not filtered_df.empty:
            asset_filter = show_filter_interface(filtered_df)
            
            st.markdown("---")
            
            show_selection_summary(asset_filter)
        else:
            ui.create_info_box(
                "Aplique o filtro de liquidez acima para começar a seleção.",
                "info"
            )
    
    with tab2:
        show_manual_selection(universe_df)
    
    with tab3:
        show_current_selection()
    
    with tab4:
        ui.create_section_header(
            "📊 Estatísticas do Universo",
            "Visão geral de todos os ativos disponíveis",
            "📊"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribuição por Setor")
            sector_dist = filters.create_sector_distribution(universe_df)
            
            if not sector_dist.empty:
                st.dataframe(
                    sector_dist.style.format({'percentage': '{:.1f}%'}),
                    use_container_width=True,
                    height=400
                )
        
        with col2:
            st.markdown("#### Distribuição por Tipo")
            type_dist = filters.create_type_distribution(universe_df)
            
            if not type_dist.empty:
                fig = ui.plot_portfolio_weights(
                    dict(zip(type_dist['tipo'], type_dist['count'] / type_dist['count'].sum())),
                    title="Distribuição por Tipo de Ação"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Resumo geral
        st.markdown("#### 📋 Resumo Geral")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            ui.create_metric_card(
                "Total de Ativos",
                f"{len(universe_df)}",
                icon="📊"
            )
        
        with summary_cols[1]:
            ui.create_metric_card(
                "Setores Únicos",
                f"{universe_df['setor'].nunique()}",
                icon="🏢"
            )
        
        with summary_cols[2]:
            ui.create_metric_card(
                "Subsetores",
                f"{universe_df['subsetor'].nunique()}",
                icon="📁"
            )
        
        with summary_cols[3]:
            ui.create_metric_card(
                "Segmentos",
                f"{universe_df['segmento_listagem'].nunique()}",
                icon="🎯"
            )
    
    # Próximos passos
    show_next_steps()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem 0;">
            <p>💡 Dica: Selecione entre 10-30 ativos para uma análise balanceada entre diversificação e complexidade.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
