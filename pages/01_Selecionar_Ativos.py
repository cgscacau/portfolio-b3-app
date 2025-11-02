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
    """Inicializa variáveis de sessão se não existirem."""
    utils.ensure_session_state_initialized()


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
    
    # Obter modo de operação
    use_mock = utils.get_use_mock_flag()
    
    # Mostrar modo atual
    if use_mock:
        st.info("🎲 Modo simulado ativo - Liquidez será gerada aleatoriamente")
    else:
        st.info("📡 Modo real - Verificando liquidez via yfinance")
    
    # Explicação dos valores
    with st.expander("ℹ️ Como interpretar o volume?", expanded=False):
        st.markdown("""
        O volume é medido em **número de ações negociadas por dia**.
        
        **Referência de liquidez:**
        - **Muito Baixa**: < 100.000 ações/dia
        - **Baixa**: 100.000 - 1.000.000 ações/dia
        - **Média**: 1.000.000 - 10.000.000 ações/dia
        - **Alta**: 10.000.000 - 50.000.000 ações/dia
        - **Muito Alta (Blue Chips)**: > 50.000.000 ações/dia
        
        **Exemplos típicos:**
        - PETR4, VALE3, ITUB4: 100-500 milhões de ações/dia
        - Ações médias: 1-10 milhões de ações/dia
        - Small caps: < 1 milhão de ações/dia
        
        **Nota:** Em modo simulado, os valores são gerados aleatoriamente 
        baseados em características conhecidas dos ativos.
        """)
    
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
        # Selector de nível de liquidez
        liquidity_level = st.selectbox(
            "Nível de liquidez desejado:",
            [
                "Muito Baixa (> 10.000)",
                "Baixa (> 100.000)",
                "Média (> 1.000.000)",
                "Alta (> 10.000.000)",
                "Muito Alta - Blue Chips (> 50.000.000)",
                "Personalizado"
            ],
            index=1,  # Padrão: Baixa
            help="Selecione o nível de liquidez mínimo"
        )
        
        # Mapear para valores
        liquidity_map = {
            "Muito Baixa (> 10.000)": 10000,
            "Baixa (> 100.000)": 100000,
            "Média (> 1.000.000)": 1000000,
            "Alta (> 10.000.000)": 10000000,
            "Muito Alta - Blue Chips (> 50.000.000)": 50000000,
        }
        
        if liquidity_level == "Personalizado":
            min_volume = st.number_input(
                "Volume médio mínimo (ações/dia):",
                min_value=1000,
                max_value=1000000000,
                value=100000,
                step=10000,
                format="%d",
                help="Volume médio diário mínimo em número de ações"
            )
        else:
            min_volume = liquidity_map[liquidity_level]
            st.info(f"📊 Volume mínimo: **{min_volume:,.0f}** ações/dia")
    
    # Opção rápida: usar todos sem verificar
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Usar Todos os Ativos (Sem Filtro)", use_container_width=True, key="skip_liquidity"):
            # Marcar todos como negociados
            universe_df['is_traded_30d'] = True
            universe_df['avg_volume_30d'] = 1000000  # Valor placeholder
            universe_df['sessions_traded_30d'] = 20
            
            st.session_state.filtered_universe_df = universe_df
            st.session_state.liquidity_applied = True
            
            st.success(f"✅ {len(universe_df)} ativos disponíveis (sem verificação de liquidez)")
            st.info("ℹ️ Todos os ativos foram incluídos sem verificar liquidez real")
            st.rerun()
    
    with col2:
        if st.button("🔍 Aplicar Filtro de Liquidez", use_container_width=True, type="primary", key="apply_liquidity"):
            
            with st.spinner("Verificando liquidez dos ativos..."):
                
                filtered_df = data.filter_traded_last_30d(
                    universe_df,
                    min_sessions=min_sessions,
                    min_avg_volume=min_volume,
                    show_progress=True,
                    use_mock=use_mock  # Passar flag de mock
                )
                
                # Filtrar apenas os negociados
                traded_df = filtered_df[filtered_df['is_traded_30d'] == True].copy()
                
                st.session_state.filtered_universe_df = traded_df
                st.session_state.liquidity_applied = True
                
                # Estatísticas detalhadas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total no Universo", len(universe_df))
                
                with col2:
                    st.metric("Ativos Líquidos", len(traded_df))
                
                with col3:
                    pct = (len(traded_df) / len(universe_df) * 100) if len(universe_df) > 0 else 0
                    st.metric("% Aprovado", f"{pct:.1f}%")
                
                with col4:
                    if len(traded_df) > 0:
                        avg_vol = traded_df['avg_volume_30d'].mean()
                        st.metric("Volume Médio", f"{avg_vol/1e6:.1f}M")
                    else:
                        st.metric("Volume Médio", "N/A")
                
                if len(traded_df) > 0:
                    st.success(f"✅ {len(traded_df)} ativos líquidos identificados!")
                    
                    # Mostrar top 10 mais líquidos
                    with st.expander("🔥 Top 10 Mais Líquidos", expanded=False):
                        top10 = traded_df.nlargest(10, 'avg_volume_30d')[
                            ['ticker', 'nome', 'avg_volume_30d', 'sessions_traded_30d']
                        ].copy()
                        
                        top10['avg_volume_30d'] = top10['avg_volume_30d'].apply(
                            lambda x: f"{x/1e6:.2f}M ações/dia"
                        )
                        
                        top10.columns = ['Ticker', 'Nome', 'Volume Médio', 'Sessões']
                        
                        st.dataframe(top10, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhum ativo atende aos critérios de liquidez. Tente reduzir os limites.")
                    
                    if min_volume > 100000:
                        st.info(f"💡 Sugestão: Tente com volume mínimo de 100.000 ações/dia")


def show_simple_filters():
    """Interface simplificada de filtros."""
    
    st.markdown("### 🎯 Filtros de Seleção")
    
    # Usar universo filtrado se disponível, senão usar completo
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
        st.info(f"📊 Trabalhando com {len(working_df)} ativos filtrados por liquidez")
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
            key="filter_sectors",
            help="Filtre por setores específicos da economia"
        )
    
    with col2:
        st.markdown("**Filtro por Tipo**")
        all_types = sorted(working_df['tipo'].unique().tolist())
        selected_types = st.multiselect(
            "Selecione tipos:",
            options=all_types,
            default=[],
            key="filter_types",
            help="ON (Ordinária), PN (Preferencial), UNIT, etc."
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
            utils.log_user_action("Selecionados todos os ativos filtrados", {"count": len(st.session_state.selected_tickers)})
            st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados!")
            st.rerun()
    
    with col2:
        if st.button("🔥 Top 20 Liquidez", use_container_width=True, key="select_top20"):
            if 'avg_volume_30d' in filtered.columns:
                top = filtered.nlargest(20, 'avg_volume_30d')
                st.session_state.selected_tickers = top['ticker'].tolist()
                utils.log_user_action("Selecionados top 20 por liquidez", {"count": len(st.session_state.selected_tickers)})
                st.success(f"✅ {len(st.session_state.selected_tickers)} ativos mais líquidos selecionados!")
                st.rerun()
            else:
                st.warning("⚠️ Aplique o filtro de liquidez primeiro")
    
    with col3:
        if st.button("🎲 10 Aleatórios", use_container_width=True, key="select_random"):
            sample_size = min(10, len(filtered))
            random_sample = filtered.sample(n=sample_size)
            st.session_state.selected_tickers = random_sample['ticker'].tolist()
            utils.log_user_action("Selecionados aleatoriamente", {"count": len(st.session_state.selected_tickers)})
            st.success(f"✅ {len(st.session_state.selected_tickers)} ativos selecionados aleatoriamente!")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Limpar", use_container_width=True, key="clear_selection"):
            st.session_state.selected_tickers = []
            utils.log_user_action("Seleção limpa")
            st.success("✅ Seleção limpa!")
            st.rerun()


def show_manual_selection():
    """Seleção manual com multiselect."""
    
    st.markdown("### ✍️ Seleção Manual")
    
    # Usar universo apropriado
    if not st.session_state.filtered_universe_df.empty:
        working_df = st.session_state.filtered_universe_df
        st.info("📊 Selecionando de ativos filtrados por liquidez")
    elif not st.session_state.universe_df.empty:
        working_df = st.session_state.universe_df
        st.warning("⚠️ Selecionando de universo completo (sem filtro de liquidez)")
    else:
        st.error("❌ Nenhum dado disponível")
        return
    
    available_tickers = sorted(working_df['ticker'].tolist())
    
    # Criar opções com nome
    ticker_options = []
    ticker_map = {}
    
    for ticker in available_tickers:
        nome = working_df[working_df['ticker'] == ticker]['nome'].iloc[0]
        option = f"{ticker} - {nome}"
        ticker_options.append(option)
        ticker_map[option] = ticker
    
    # Pré-selecionar os já selecionados
    default_options = []
    for ticker in st.session_state.selected_tickers:
        if ticker in working_df['ticker'].values:
            nome = working_df[working_df['ticker'] == ticker]['nome'].iloc[0]
            default_options.append(f"{ticker} - {nome}")
    
    selected_options = st.multiselect(
        "Digite ou selecione tickers:",
        options=ticker_options,
        default=default_options,
        key="manual_select",
        help="Busque por ticker ou nome da empresa"
    )
    
    # Extrair apenas os tickers
    selected_tickers = [ticker_map[opt] for opt in selected_options]
    
    # Mostrar contagem
    st.info(f"📊 {len(selected_tickers)} ativos selecionados")
    
    if st.button("💾 Salvar Seleção Manual", use_container_width=True, type="primary", key="save_manual"):
        st.session_state.selected_tickers = selected_tickers
        utils.log_user_action("Seleção manual salva", {"count": len(selected_tickers)})
        st.success(f"✅ {len(selected_tickers)} ativos salvos!")
        st.rerun()


def show_current_selection():
    """Mostra seleção atual."""
    
    st.markdown("### ✅ Seleção Atual")
    
    if not st.session_state.selected_tickers:
        st.info("ℹ️ Nenhum ativo selecionado ainda. Use as abas acima para selecionar.")
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
    
    # Validação de diversificação
    if not st.session_state.universe_df.empty:
        is_valid, concentration = filters.validate_sector_diversification(
            st.session_state.selected_tickers,
            st.session_state.universe_df,
            max_sector_pct=40.0
        )
        
        if not is_valid:
            max_sector = max(concentration, key=concentration.get)
            max_pct = concentration[max_sector]
            
            st.warning(f"""
            ⚠️ **Alerta de Concentração Setorial**
            
            O setor **{max_sector}** representa {max_pct:.1f}% da carteira, 
            acima do limite recomendado de 40%.
            
            **Recomendação:** Considere adicionar ativos de outros setores para 
            melhorar a diversificação e reduzir risco idiossincrático.
            """)
    
    # Lista
    st.markdown("**Ativos Selecionados:**")
    
    if not st.session_state.universe_df.empty:
        selected_df = st.session_state.universe_df[
            st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
        ][['ticker', 'nome', 'setor', 'tipo']].copy()
        
        selected_df.columns = ['Ticker', 'Nome', 'Setor', 'Tipo']
        st.dataframe(selected_df, use_container_width=True, height=300)
    else:
        st.write(", ".join(st.session_state.selected_tickers))
    
    # Ações
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_text = "\n".join(st.session_state.selected_tickers)
        st.download_button(
            "📥 Exportar Lista (TXT)",
            tickers_text,
            "selected_tickers.txt",
            "text/plain",
            use_container_width=True
        )
    
    with col2:
        if not st.session_state.universe_df.empty:
            selected_full = st.session_state.universe_df[
                st.session_state.universe_df['ticker'].isin(st.session_state.selected_tickers)
            ]
            csv = selected_full.to_csv(index=False)
            
            st.download_button(
                "📥 Exportar Lista (CSV)",
                csv,
                "selected_tickers.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("🗑️ Limpar Tudo", use_container_width=True, key="clear_all"):
            st.session_state.selected_tickers = []
            utils.log_user_action("Seleção limpa completamente")
            st.success("✅ Seleção limpa!")
            st.rerun()


def show_statistics():
    """Mostra estatísticas do universo."""
    
    st.markdown("### 📊 Estatísticas do Universo")
    
    universe_df = st.session_state.universe_df
    
    if universe_df.empty:
        st.info("ℹ️ Carregue os dados para ver as estatísticas")
        return
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Ativos", len(universe_df))
    
    with col2:
        st.metric("Setores Únicos", universe_df['setor'].nunique())
    
    with col3:
        st.metric("Subsetores", universe_df['subsetor'].nunique())
    
    with col4:
        st.metric("Segmentos", universe_df['segmento_listagem'].nunique())
    
    # Distribuições
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribuição por Setor")
        sector_counts = universe_df['setor'].value_counts()
        st.bar_chart(sector_counts)
    
    with col2:
        st.markdown("#### Distribuição por Tipo")
        type_counts = universe_df['tipo'].value_counts()
        st.bar_chart(type_counts)
    
    # Tabela detalhada
    with st.expander("📋 Tabela Detalhada por Setor"):
        sector_summary = universe_df.groupby('setor').agg({
            'ticker': 'count',
            'subsetor': 'nunique'
        }).reset_index()
        
        sector_summary.columns = ['Setor', 'Nº Ativos', 'Nº Subsetores']
        sector_summary = sector_summary.sort_values('Nº Ativos', ascending=False)
        
        st.dataframe(sector_summary, use_container_width=True)


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
    
    # Mostrar modo de operação
    utils.show_data_mode_indicator()
    
    # Carregar universo se necessário
    if st.session_state.universe_df.empty:
        with st.spinner("Carregando universo de ativos..."):
            universe_df = load_universe()
            
            if universe_df.empty:
                st.error("❌ Não foi possível carregar o universo de ativos")
                st.stop()
    
    st.markdown("---")
    
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
            st.info("ℹ️ Aplique o filtro de liquidez acima ou use 'Todos os Ativos' para começar")
    
    with tab2:
        show_manual_selection()
    
    with tab3:
        show_current_selection()
    
    with tab4:
        show_statistics()
    
    # Próximos passos
    if st.session_state.selected_tickers:
        st.markdown("---")
        st.markdown("### 🚀 Próximos Passos")
        
        st.success(f"✅ {len(st.session_state.selected_tickers)} ativos prontos para análise!")
        
        st.info("""
        **Continue sua análise:**
        
        Use o menu lateral (☰) para navegar até:
        - 💸 **Análise de Dividendos** - Histórico e regularidade de pagamentos
        - 📊 **Portfólios Eficientes** - Otimização via Markowitz
        - 🎯 **Sharpe e MinVol** - Carteiras especializadas
        - 📋 **Resumo Executivo** - Recomendação final
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem 0;">
            <p>💡 Dica: Selecione entre 10-30 ativos para análise balanceada entre diversificação e complexidade.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
