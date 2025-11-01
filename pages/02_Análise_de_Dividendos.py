"""
Página 2: Análise de Dividendos
Análise detalhada de histórico, regularidade e projeções de dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics, ui
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Análise de Dividendos - Portfolio B3",
    page_icon="💸",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'dividend_data' not in st.session_state:
        st.session_state.dividend_data = {}
    
    if 'price_data' not in st.session_state:
        st.session_state.price_data = pd.DataFrame()
    
    if 'dividend_metrics' not in st.session_state:
        st.session_state.dividend_metrics = pd.DataFrame()


def check_prerequisites():
    """Verifica se há ativos selecionados."""
    if not st.session_state.selected_tickers:
        ui.create_info_box(
            "⚠️ Nenhum ativo selecionado. Por favor, vá para a página 'Selecionar Ativos' primeiro.",
            "warning"
        )
        
        if st.button("🎯 Ir para Seleção de Ativos", type="primary"):
            st.switch_page("app/pages/01_Selecionar_Ativos.py")
        
        return False
    
    return True


def load_dividend_data():
    """Carrega dados de dividendos e preços."""
    
    ui.create_section_header(
        "📥 Carregamento de Dados",
        "Baixando histórico de dividendos e preços",
        "📥"
    )
    
    tickers = st.session_state.selected_tickers
    start_date = st.session_state.period_start
    end_date = st.session_state.period_end
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📊 **Ativos:** {len(tickers)}")
        st.info(f"📅 **Período:** {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
    
    with col2:
        days = (end_date - start_date).days
        st.info(f"⏱️ **Duração:** {days} dias (~{days/365:.1f} anos)")
    
    if st.button("🔄 Carregar/Atualizar Dados", type="primary", use_container_width=True):
        
        # Carregar preços
        with st.spinner("Carregando histórico de preços..."):
            prices_df = data.get_price_history(tickers, start_date, end_date, use_cache=True)
            
            if prices_df.empty:
                st.error("❌ Erro ao carregar dados de preços")
                return False
            
            st.session_state.price_data = prices_df
        
        # Carregar dividendos
        with st.spinner("Carregando histórico de dividendos..."):
            dividends_dict = data.get_dividends(tickers, start_date, end_date, use_cache=True)
            
            st.session_state.dividend_data = dividends_dict
        
        # Validar qualidade
        with st.spinner("Validando qualidade dos dados..."):
            clean_prices, removed, reasons = data.validate_data_quality(
                prices_df,
                min_data_points=min(252, len(prices_df) // 2)
            )
            
            if removed:
                st.warning(f"⚠️ {len(removed)} ativos removidos por dados insuficientes")
                
                # Remover dos dividendos também
                for ticker in removed:
                    if ticker in st.session_state.dividend_data:
                        del st.session_state.dividend_data[ticker]
                
                # Atualizar lista de tickers selecionados
                st.session_state.selected_tickers = [
                    t for t in st.session_state.selected_tickers if t not in removed
                ]
            
            st.session_state.price_data = clean_prices
        
        # Calcular métricas de dividendos
        with st.spinner("Calculando métricas de dividendos..."):
            calculate_dividend_metrics()
        
        st.success("✅ Dados carregados com sucesso!")
        st.rerun()
    
    return True


def calculate_dividend_metrics():
    """Calcula métricas de dividendos para todos os ativos."""
    
    if not st.session_state.dividend_data or st.session_state.price_data.empty:
        return
    
    dividend_metrics_obj = metrics.DividendMetrics(
        st.session_state.dividend_data,
        st.session_state.price_data
    )
    
    metrics_list = []
    
    for ticker in st.session_state.selected_tickers:
        ticker_metrics = dividend_metrics_obj.calculate_all_dividend_metrics(ticker)
        metrics_list.append(ticker_metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    st.session_state.dividend_metrics = metrics_df


def show_dividend_overview():
    """Exibe visão geral dos dividendos."""
    
    if st.session_state.dividend_metrics.empty:
        ui.create_info_box(
            "Carregue os dados usando o botão acima para visualizar as métricas.",
            "info"
        )
        return
    
    ui.create_section_header(
        "📊 Visão Geral de Dividendos",
        "Principais métricas de dividendos dos ativos selecionados",
        "📊"
    )
    
    metrics_df = st.session_state.dividend_metrics
    
    # Cards de resumo
    col1, col2, col3, col4 = st.columns(4)
    
    # Ativos com dividendos
    assets_with_divs = (metrics_df['num_payments'] > 0).sum()
    
    with col1:
        ui.create_metric_card(
            "Ativos com Dividendos",
            f"{assets_with_divs}/{len(metrics_df)}",
            icon="💰"
        )
    
    with col2:
        avg_dy = metrics_df['dividend_yield'].mean()
        ui.create_metric_card(
            "DY Médio",
            f"{avg_dy*100:.2f}%" if not np.isnan(avg_dy) else "N/A",
            help_text="Dividend Yield médio (12 meses)",
            icon="📈"
        )
    
    with col3:
        avg_regularity = metrics_df['regularity_index'].mean()
        ui.create_metric_card(
            "Regularidade Média",
            f"{avg_regularity:.1f}" if not np.isnan(avg_regularity) else "N/A",
            help_text="Índice de regularidade (0-100)",
            icon="📅"
        )
    
    with col4:
        total_payments = metrics_df['num_payments'].sum()
        ui.create_metric_card(
            "Total de Pagamentos",
            f"{int(total_payments)}",
            help_text="Soma de todos os pagamentos no período",
            icon="💸"
        )
    
    # Tabela de métricas
    st.markdown("### 📋 Métricas Detalhadas por Ativo")
    
    # Preparar DataFrame para exibição
    display_df = metrics_df.copy()
    
    # Ordenar por dividend yield
    display_df = display_df.sort_values('dividend_yield', ascending=False)
    
    # Formatar colunas
    display_df['dividend_yield'] = display_df['dividend_yield'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    display_df['dividend_yield_avg'] = display_df['dividend_yield_avg'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    display_df['regularity_index'] = display_df['regularity_index'].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
    )
    display_df['consistency_score'] = display_df['consistency_score'].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
    )
    display_df['uniformity_score'] = display_df['uniformity_score'].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
    )
    display_df['dividend_growth_rate'] = display_df['dividend_growth_rate'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    
    # Renomear colunas
    rename_map = {
        'ticker': 'Ticker',
        'dividend_yield': 'DY (12m)',
        'dividend_yield_avg': 'DY Médio',
        'regularity_index': 'Regularidade',
        'consistency_score': 'Consistência',
        'uniformity_score': 'Uniformidade',
        'dividend_growth_rate': 'Crescimento',
        'num_payments': 'Nº Pagamentos'
    }
    
    display_df = display_df.rename(columns=rename_map)
    
    # Selecionar colunas para exibir
    display_cols = ['Ticker', 'DY (12m)', 'DY Médio', 'Regularidade', 
                   'Consistência', 'Uniformidade', 'Crescimento', 'Nº Pagamentos']
    
    display_df = display_df[display_cols]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ui.create_download_button(
            metrics_df,
            "dividend_metrics.csv",
            "📥 Download CSV",
            "csv"
        )
    
    with col2:
        ui.create_download_button(
            metrics_df,
            "dividend_metrics.json",
            "📥 Download JSON",
            "json"
        )


def show_top_performers():
    """Exibe top performers em diferentes categorias."""
    
    if st.session_state.dividend_metrics.empty:
        return
    
    ui.create_section_header(
        "🏆 Destaques",
        "Melhores ativos em cada categoria",
        "🏆"
    )
    
    metrics_df = st.session_state.dividend_metrics
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Maior Dividend Yield")
        top_dy = metrics_df.nlargest(5, 'dividend_yield')[['ticker', 'dividend_yield']]
        
        for idx, row in top_dy.iterrows():
            ticker = row['ticker']
            dy = row['dividend_yield']
            
            if not pd.isna(dy):
                st.markdown(f"**{ticker}**: {dy*100:.2f}%")
    
    with col2:
        st.markdown("#### 📅 Maior Regularidade")
        top_reg = metrics_df.nlargest(5, 'regularity_index')[['ticker', 'regularity_index']]
        
        for idx, row in top_reg.iterrows():
            ticker = row['ticker']
            reg = row['regularity_index']
            
            if not pd.isna(reg):
                st.markdown(f"**{ticker}**: {reg:.1f}/100")
    
    with col3:
        st.markdown("#### 📈 Maior Crescimento")
        top_growth = metrics_df.nlargest(5, 'dividend_growth_rate')[['ticker', 'dividend_growth_rate']]
        
        for idx, row in top_growth.iterrows():
            ticker = row['ticker']
            growth = row['dividend_growth_rate']
            
            if not pd.isna(growth):
                st.markdown(f"**{ticker}**: {growth*100:.2f}%/ano")


def show_dividend_history():
    """Exibe histórico detalhado de dividendos."""
    
    if not st.session_state.dividend_data:
        ui.create_info_box(
            "Carregue os dados para visualizar o histórico de dividendos.",
            "info"
        )
        return
    
    ui.create_section_header(
        "📈 Histórico de Dividendos",
        "Visualização temporal dos pagamentos",
        "📈"
    )
    
    # Seletor de ativo
    ticker = st.selectbox(
        "Selecione um ativo para análise detalhada:",
        options=st.session_state.selected_tickers,
        key="dividend_history_ticker"
    )
    
    if ticker not in st.session_state.dividend_data:
        st.warning(f"⚠️ {ticker} não possui histórico de dividendos no período selecionado.")
        return
    
    divs = st.session_state.dividend_data[ticker]
    
    if divs.empty:
        st.warning(f"⚠️ {ticker} não possui dividendos no período.")
        return
    
    # Métricas do ativo
    col1, col2, col3, col4 = st.columns(4)
    
    ticker_metrics = st.session_state.dividend_metrics[
        st.session_state.dividend_metrics['ticker'] == ticker
    ].iloc[0]
    
    with col1:
        ui.create_metric_card(
            "Dividend Yield",
            f"{ticker_metrics['dividend_yield']*100:.2f}%" if not pd.isna(ticker_metrics['dividend_yield']) else "N/A",
            icon="💰"
        )
    
    with col2:
        ui.create_metric_card(
            "Regularidade",
            f"{ticker_metrics['regularity_index']:.1f}/100" if not pd.isna(ticker_metrics['regularity_index']) else "N/A",
            icon="📅"
        )
    
    with col3:
        ui.create_metric_card(
            "Nº Pagamentos",
            f"{int(ticker_metrics['num_payments'])}",
            icon="💸"
        )
    
    with col4:
        total_divs = divs.sum()
        ui.create_metric_card(
            "Total Pago",
            f"R$ {total_divs:.2f}",
            icon="💵"
        )
    
    # Gráfico de linha temporal
    st.markdown("### 📊 Série Temporal de Dividendos")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=divs.index,
        y=divs.values,
        mode='lines+markers',
        name='Dividendos',
        line=dict(color=ui.COLORS['primary'], width=2),
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='Data: %{x|%d/%m/%Y}<br>' +
                     'Dividendo: R$ %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Média móvel
    if len(divs) >= 3:
        ma = divs.rolling(window=3, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma.values,
            mode='lines',
            name='Média Móvel (3)',
            line=dict(color=ui.COLORS['warning'], width=2, dash='dash'),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>' +
                         'Média: R$ %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Histórico de Dividendos - {ticker}",
        xaxis_title="Data",
        yaxis_title="Dividendo (R$)",
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(38, 39, 48, 0.8)',
            bordercolor=ui.COLORS['primary'],
            borderwidth=1
        ),
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calendário mensal
    st.markdown("### 📅 Calendário Mensal")
    
    monthly_divs = divs.resample('M').sum()
    
    # Criar DataFrame para heatmap
    monthly_df = pd.DataFrame({
        'Mês': monthly_divs.index.strftime('%Y-%m'),
        'Dividendo': monthly_divs.values
    })
    
    # Gráfico de barras mensal
    fig_monthly = go.Figure()
    
    fig_monthly.add_trace(go.Bar(
        x=monthly_df['Mês'],
        y=monthly_df['Dividendo'],
        marker=dict(
            color=monthly_df['Dividendo'],
            colorscale='Greens',
            line=dict(width=1, color='white')
        ),
        text=[f'R$ {v:.2f}' for v in monthly_df['Dividendo']],
        textposition='outside',
        hovertemplate='Mês: %{x}<br>' +
                     'Dividendo: R$ %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig_monthly.update_layout(
        title=f"Dividendos Mensais - {ticker}",
        xaxis_title="Mês",
        yaxis_title="Dividendo (R$)",
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=400
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)


def show_dividend_calendar_heatmap():
    """Exibe heatmap de calendário de dividendos para todos os ativos."""
    
    if not st.session_state.dividend_data:
        return
    
    ui.create_section_header(
        "🗓️ Calendário Consolidado",
        "Heatmap de dividendos mensais de todos os ativos",
        "🗓️"
    )
    
    # Criar matriz de dividendos mensais
    monthly_data = {}
    
    for ticker, divs in st.session_state.dividend_data.items():
        if not divs.empty:
            monthly = divs.resample('M').sum()
            monthly_data[ticker] = monthly
    
    if not monthly_data:
        st.warning("⚠️ Nenhum dividendo disponível para criar o calendário.")
        return
    
    # Consolidar em DataFrame
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df = monthly_df.fillna(0)
    
    # Limitar a últimos 24 meses para visualização
    if len(monthly_df) > 24:
        monthly_df = monthly_df.tail(24)
    
    # Plotar heatmap
    fig = ui.plot_dividend_calendar(monthly_df, "Calendário de Dividendos Mensais")
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas do calendário
    st.markdown("### 📊 Estatísticas do Calendário")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_months = len(monthly_df)
        ui.create_metric_card(
            "Meses Analisados",
            f"{total_months}",
            icon="📅"
        )
    
    with col2:
        avg_monthly = monthly_df.sum(axis=1).mean()
        ui.create_metric_card(
            "Média Mensal Total",
            f"R$ {avg_monthly:.2f}",
            help_text="Soma de todos os ativos",
            icon="💰"
        )
    
    with col3:
        # Mês com maior pagamento
        max_month_idx = monthly_df.sum(axis=1).idxmax()
        max_month_value = monthly_df.sum(axis=1).max()
        
        ui.create_metric_card(
            "Maior Mês",
            f"R$ {max_month_value:.2f}",
            help_text=f"{max_month_idx.strftime('%m/%Y')}",
            icon="🏆"
        )


def show_regularity_analysis():
    """Análise detalhada de regularidade."""
    
    if st.session_state.dividend_metrics.empty:
        return
    
    ui.create_section_header(
        "📊 Análise de Regularidade",
        "Compreenda a consistência dos pagamentos",
        "📊"
    )
    
    metrics_df = st.session_state.dividend_metrics
    
    # Explicação do índice
    with st.expander("ℹ️ Como funciona o Índice de Regularidade?", expanded=False):
        st.markdown("""
        O **Índice de Regularidade** (0-100) é composto por três componentes:
        
        1. **Consistência (40%)**: Percentual de meses com pagamento
           - 100% = pagamento em todos os meses
           - Quanto maior, melhor
        
        2. **Uniformidade (40%)**: Inverso do coeficiente de variação
           - Mede se os valores pagos são similares
           - Valores mais uniformes indicam previsibilidade
        
        3. **Previsibilidade (20%)**: Correlação com média móvel
           - Indica se há padrão nos pagamentos
           - Alta correlação = comportamento previsível
        
        **Interpretação:**
        - **80-100**: Excelente regularidade
        - **60-80**: Boa regularidade
        - **40-60**: Regularidade moderada
        - **< 40**: Baixa regularidade
        """)
    
    # Scatter plot: Regularidade vs Dividend Yield
    st.markdown("### 📈 Regularidade vs Dividend Yield")
    
    # Preparar dados para scatter
    scatter_df = metrics_df[['ticker', 'dividend_yield', 'regularity_index']].copy()
    scatter_df = scatter_df.dropna()
    
    if not scatter_df.empty:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=scatter_df['regularity_index'],
            y=scatter_df['dividend_yield'] * 100,
            mode='markers+text',
            text=scatter_df['ticker'],
            textposition='top center',
            textfont=dict(size=10, color=ui.COLORS['text']),
            marker=dict(
                size=12,
                color=scatter_df['dividend_yield'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="DY"),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Regularidade: %{x:.1f}<br>' +
                         'DY: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Relação entre Regularidade e Dividend Yield",
            xaxis_title="Índice de Regularidade (0-100)",
            yaxis_title="Dividend Yield (%)",
            template='plotly_dark',
            hovermode='closest',
            showlegend=False,
            plot_bgcolor=ui.COLORS['background'],
            paper_bgcolor=ui.COLORS['background'],
            font=dict(color=ui.COLORS['text']),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de quadrantes
        st.markdown("### 🎯 Análise de Quadrantes")
        
        median_reg = scatter_df['regularity_index'].median()
        median_dy = scatter_df['dividend_yield'].median()
        
        # Classificar ativos
        scatter_df['quadrant'] = 'Outros'
        scatter_df.loc[
            (scatter_df['regularity_index'] >= median_reg) & 
            (scatter_df['dividend_yield'] >= median_dy),
            'quadrant'
        ] = '🏆 Alto DY + Alta Regularidade'
        
        scatter_df.loc[
            (scatter_df['regularity_index'] >= median_reg) & 
            (scatter_df['dividend_yield'] < median_dy),
            'quadrant'
        ] = '📅 Alta Regularidade'
        
        scatter_df.loc[
            (scatter_df['regularity_index'] < median_reg) & 
            (scatter_df['dividend_yield'] >= median_dy),
            'quadrant'
        ] = '💰 Alto DY'
        
        # Mostrar por quadrante
        for quadrant in scatter_df['quadrant'].unique():
            if quadrant != 'Outros':
                quad_tickers = scatter_df[scatter_df['quadrant'] == quadrant]['ticker'].tolist()
                
                if quad_tickers:
                    st.markdown(f"**{quadrant}**: {', '.join(quad_tickers)}")


def show_comparison_tool():
    """Ferramenta de comparação entre ativos."""
    
    if not st.session_state.dividend_data or st.session_state.dividend_metrics.empty:
        return
    
    ui.create_section_header(
        "⚖️ Comparação de Ativos",
        "Compare até 5 ativos lado a lado",
        "⚖️"
    )
    
    # Seletor de ativos
    selected_for_comparison = st.multiselect(
        "Selecione ativos para comparar (máx. 5):",
        options=st.session_state.selected_tickers,
        max_selections=5,
        key="comparison_tickers"
    )
    
    if not selected_for_comparison:
        st.info("Selecione pelo menos 2 ativos para comparar.")
        return
    
    if len(selected_for_comparison) < 2:
        st.warning("Selecione pelo menos 2 ativos para comparação.")
        return
    
    # Filtrar métricas
    comparison_df = st.session_state.dividend_metrics[
        st.session_state.dividend_metrics['ticker'].isin(selected_for_comparison)
    ].copy()
    
    comparison_df = comparison_df.set_index('ticker')
    
    # Selecionar métricas para comparar
    metrics_to_compare = [
        'dividend_yield',
        'regularity_index',
        'consistency_score',
        'uniformity_score',
        'num_payments'
    ]
    
    comparison_display = comparison_df[metrics_to_compare].T
    
    # Renomear índice
    rename_map = {
        'dividend_yield': 'Dividend Yield',
        'regularity_index': 'Regularidade',
        'consistency_score': 'Consistência',
        'uniformity_score': 'Uniformidade',
        'num_payments': 'Nº Pagamentos'
    }
    
    comparison_display = comparison_display.rename(index=rename_map)
    
    # Formatar valores
    for col in comparison_display.columns:
        comparison_display[col] = comparison_display[col].apply(
            lambda x: f"{x*100:.2f}%" if 'Yield' in comparison_display.index[comparison_display[col] == x].values[0] else f"{x:.1f}"
        )
    
    st.dataframe(comparison_display, use_container_width=True)
    
    # Gráfico de radar
    st.markdown("### 📊 Comparação Visual")
    
    fig = ui.plot_portfolio_comparison(comparison_df[metrics_to_compare], "Comparação de Métricas")
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Função principal da página."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<p class="gradient-title">💸 Análise de Dividendos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Análise completa do histórico de dividendos, regularidade de pagamentos e projeções 
    para os ativos selecionados.
    """)
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        st.stop()
    
    # Informações do período
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"📊 **{len(st.session_state.selected_tickers)} ativos selecionados** para análise")
    
    with col2:
        if st.button("🔙 Voltar para Seleção", use_container_width=True):
            st.switch_page("app/pages/01_Selecionar_Ativos.py")
    
    st.markdown("---")
    
    # Carregar dados
    load_dividend_data()
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral",
        "📈 Histórico Detalhado",
        "🗓️ Calendário",
        "📊 Regularidade",
        "⚖️ Comparação"
    ])
    
    with tab1:
        show_dividend_overview()
        st.markdown("---")
        show_top_performers()
    
    with tab2:
        show_dividend_history()
    
    with tab3:
        show_dividend_calendar_heatmap()
    
    with tab4:
        show_regularity_analysis()
    
    with tab5:
        show_comparison_tool()
    
    # Próximos passos
    st.markdown("---")
    
    ui.create_section_header(
        "🚀 Próximos Passos",
        "Continue para otimização de portfólios",
        "🚀"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Portfólios Eficientes", use_container_width=True, type="primary"):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
    
    with col2:
        if st.button("🎯 Sharpe e MinVol", use_container_width=True):
            st.switch_page("app/pages/04_Sharpe_e_MinVol.py")
    
    with col3:
        if st.button("📋 Resumo Executivo", use_container_width=True):
            st.switch_page("app/pages/05_Resumo_Executivo.py")


if __name__ == "__main__":
    main()
