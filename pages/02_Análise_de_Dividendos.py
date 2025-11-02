"""
Página 2: Análise de Dividendos
Análise detalhada de histórico, regularidade e projeções
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics, ui, utils
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
    utils.ensure_session_state_initialized()


def check_prerequisites():
    """Verifica se há ativos selecionados."""
    if not st.session_state.selected_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para **Selecionar Ativos** no menu lateral")
        return False
    return True


def load_dividend_data():
    """Carrega dados de dividendos e preços."""
    
    st.markdown("### 📥 Carregamento de Dados")
    
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
    
    if st.button("🔄 Carregar Dados", type="primary", use_container_width=True):
        
        # Preços
        with st.spinner("Carregando preços..."):
            prices_df = data.get_price_history(tickers, start_date, end_date, use_cache=True)
            
            if prices_df.empty:
                st.error("❌ Erro ao carregar preços")
                return False
            
            st.session_state.price_data = prices_df
        
        # Dividendos
        with st.spinner("Carregando dividendos..."):
            dividends_dict = data.get_dividends(tickers, start_date, end_date, use_cache=True)
            st.session_state.dividend_data = dividends_dict
        
        # Validar qualidade
        with st.spinner("Validando qualidade..."):
            clean_prices, removed, reasons = data.validate_data_quality(
                prices_df,
                min_data_points=min(252, len(prices_df) // 2)
            )
            
            if removed:
                st.warning(f"⚠️ {len(removed)} ativos removidos")
                
                for ticker in removed:
                    if ticker in st.session_state.dividend_data:
                        del st.session_state.dividend_data[ticker]
                
                st.session_state.selected_tickers = [
                    t for t in st.session_state.selected_tickers if t not in removed
                ]
            
            st.session_state.price_data = clean_prices
        
        # Calcular métricas
        with st.spinner("Calculando métricas..."):
            calculate_dividend_metrics()
        
        st.success("✅ Dados carregados!")
        st.rerun()
    
    return True


def calculate_dividend_metrics():
    """Calcula métricas de dividendos."""
    
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
    
    st.session_state.dividend_metrics = pd.DataFrame(metrics_list)


def show_dividend_overview():
    """Visão geral de dividendos."""
    
    if st.session_state.dividend_metrics.empty:
        st.info("ℹ️ Carregue os dados acima")
        return
    
    st.markdown("### 📊 Visão Geral")
    
    metrics_df = st.session_state.dividend_metrics
    
    # Cards
    col1, col2, col3, col4 = st.columns(4)
    
    assets_with_divs = (metrics_df['num_payments'] > 0).sum()
    
    with col1:
        ui.create_metric_card(
            "Com Dividendos",
            f"{assets_with_divs}/{len(metrics_df)}",
            icon="💰"
        )
    
    with col2:
        avg_dy = metrics_df['dividend_yield'].mean()
        ui.create_metric_card(
            "DY Médio",
            f"{avg_dy*100:.2f}%" if not np.isnan(avg_dy) else "N/A",
            icon="📈"
        )
    
    with col3:
        avg_reg = metrics_df['regularity_index'].mean()
        ui.create_metric_card(
            "Regularidade Média",
            f"{avg_reg:.1f}" if not np.isnan(avg_reg) else "N/A",
            icon="📅"
        )
    
    with col4:
        total_pay = metrics_df['num_payments'].sum()
        ui.create_metric_card(
            "Total Pagamentos",
            f"{int(total_pay)}",
            icon="💸"
        )
    
    # Tabela
    st.markdown("### 📋 Métricas Detalhadas")
    
    display_df = metrics_df.copy()
    display_df = display_df.sort_values('dividend_yield', ascending=False)
    
    # Formatar
    display_df['dividend_yield'] = display_df['dividend_yield'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    display_df['dividend_yield_avg'] = display_df['dividend_yield_avg'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    display_df['regularity_index'] = display_df['regularity_index'].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
    )
    display_df['dividend_growth_rate'] = display_df['dividend_growth_rate'].apply(
        lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A"
    )
    
    # Renomear
    rename = {
        'ticker': 'Ticker',
        'dividend_yield': 'DY (12m)',
        'dividend_yield_avg': 'DY Médio',
        'regularity_index': 'Regularidade',
        'dividend_growth_rate': 'Crescimento',
        'num_payments': 'Pagamentos'
    }
    
    display_df = display_df.rename(columns=rename)
    
    cols = ['Ticker', 'DY (12m)', 'DY Médio', 'Regularidade', 'Crescimento', 'Pagamentos']
    st.dataframe(display_df[cols], use_container_width=True, height=400)
    
    # Download
    col1, col2 = st.columns(2)
    
    with col1:
        csv = metrics_df.to_csv(index=False)
        st.download_button("📥 CSV", csv, "dividend_metrics.csv", use_container_width=True)
    
    with col2:
        json = metrics_df.to_json(orient='records', indent=2)
        st.download_button("📥 JSON", json, "dividend_metrics.json", use_container_width=True)


def show_top_performers():
    """Top performers."""
    
    if st.session_state.dividend_metrics.empty:
        return
    
    st.markdown("### 🏆 Destaques")
    
    metrics_df = st.session_state.dividend_metrics
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Maior DY")
        top = metrics_df.nlargest(5, 'dividend_yield')[['ticker', 'dividend_yield']]
        
        for _, row in top.iterrows():
            if not pd.isna(row['dividend_yield']):
                st.markdown(f"**{row['ticker']}**: {row['dividend_yield']*100:.2f}%")
    
    with col2:
        st.markdown("#### 📅 Maior Regularidade")
        top = metrics_df.nlargest(5, 'regularity_index')[['ticker', 'regularity_index']]
        
        for _, row in top.iterrows():
            if not pd.isna(row['regularity_index']):
                st.markdown(f"**{row['ticker']}**: {row['regularity_index']:.1f}/100")
    
    with col3:
        st.markdown("#### 📈 Maior Crescimento")
        top = metrics_df.nlargest(5, 'dividend_growth_rate')[['ticker', 'dividend_growth_rate']]
        
        for _, row in top.iterrows():
            if not pd.isna(row['dividend_growth_rate']):
                st.markdown(f"**{row['ticker']}**: {row['dividend_growth_rate']*100:.2f}%/ano")


def show_dividend_history():
    """Histórico detalhado."""
    
    if not st.session_state.dividend_data:
        st.info("ℹ️ Carregue os dados")
        return
    
    st.markdown("### 📈 Histórico Detalhado")
    
    ticker = st.selectbox(
        "Selecione um ativo:",
        options=st.session_state.selected_tickers
    )
    
    if ticker not in st.session_state.dividend_data:
        st.warning(f"⚠️ {ticker} sem dividendos no período")
        return
    
    divs = st.session_state.dividend_data[ticker]
    
    if divs.empty:
        st.warning(f"⚠️ {ticker} sem dividendos")
        return
    
    # Métricas do ativo
    ticker_metrics = st.session_state.dividend_metrics[
        st.session_state.dividend_metrics['ticker'] == ticker
    ].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
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
            "Pagamentos",
            f"{int(ticker_metrics['num_payments'])}",
            icon="💸"
        )
    
    with col4:
        ui.create_metric_card(
            "Total",
            f"R$ {divs.sum():.2f}",
            icon="💵"
        )
    
    # Gráfico temporal
    st.markdown("### 📊 Série Temporal")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=divs.index,
        y=divs.values,
        mode='lines+markers',
        name='Dividendos',
        line=dict(color=ui.COLORS['primary'], width=2),
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='%{x|%d/%m/%Y}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    # Média móvel
    if len(divs) >= 3:
        ma = divs.rolling(window=3, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma.values,
            mode='lines',
            name='MA(3)',
            line=dict(color=ui.COLORS['warning'], width=2, dash='dash'),
            hovertemplate='%{x|%d/%m/%Y}<br>R$ %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Histórico - {ticker}",
        xaxis_title="Data",
        yaxis_title="Dividendo (R$)",
        template='plotly_dark',
        hovermode='x unified',
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calendário mensal
    st.markdown("### 📅 Calendário Mensal")
    
    monthly_divs = divs.resample('M').sum()
    
    fig_monthly = go.Figure()
    
    fig_monthly.add_trace(go.Bar(
        x=monthly_divs.index.strftime('%Y-%m'),
        y=monthly_divs.values,
        marker=dict(
            color=monthly_divs.values,
            colorscale='Greens',
            line=dict(width=1, color='white')
        ),
        text=[f'R$ {v:.2f}' for v in monthly_divs.values],
        textposition='outside',
        hovertemplate='%{x}<br>R$ %{y:.2f}<extra></extra>'
    ))
    
    fig_monthly.update_layout(
        title=f"Mensal - {ticker}",
        xaxis_title="Mês",
        yaxis_title="Dividendo (R$)",
        template='plotly_dark',
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=400
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)


def show_dividend_calendar():
    """Calendário consolidado."""
    
    if not st.session_state.dividend_data:
        return
    
    st.markdown("### 🗓️ Calendário Consolidado")
    
    # Criar matriz mensal
    monthly_data = {}
    
    for ticker, divs in st.session_state.dividend_data.items():
        if not divs.empty:
            monthly = divs.resample('M').sum()
            monthly_data[ticker] = monthly
    
    if not monthly_data:
        st.warning("⚠️ Sem dividendos")
        return
    
    monthly_df = pd.DataFrame(monthly_data).fillna(0)
    
    # Limitar a 24 meses
    if len(monthly_df) > 24:
        monthly_df = monthly_df.tail(24)
    
    # Heatmap
    fig = ui.plot_dividend_calendar(monthly_df, "Calendário Mensal")
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    st.markdown("### 📊 Estatísticas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ui.create_metric_card("Meses", f"{len(monthly_df)}", icon="📅")
    
    with col2:
        avg = monthly_df.sum(axis=1).mean()
        ui.create_metric_card("Média Mensal", f"R$ {avg:.2f}", icon="💰")
    
    with col3:
        max_idx = monthly_df.sum(axis=1).idxmax()
        max_val = monthly_df.sum(axis=1).max()
        ui.create_metric_card(
            "Maior Mês",
            f"R$ {max_val:.2f}",
            help_text=f"{max_idx.strftime('%m/%Y')}",
            icon="🏆"
        )


def show_regularity_analysis():
    """Análise de regularidade."""
    
    if st.session_state.dividend_metrics.empty:
        return
    
    st.markdown("### 📊 Análise de Regularidade")
    
    with st.expander("ℹ️ Como funciona?", expanded=False):
        st.markdown("""
        **Índice de Regularidade (0-100):**
        
        1. **Consistência (40%)**: % de meses com pagamento
        2. **Uniformidade (40%)**: Inverso do CV dos valores
        3. **Previsibilidade (20%)**: Correlação com MA
        
        **Interpretação:**
        - 80-100: Excelente
        - 60-80: Boa
        - 40-60: Moderada
        - < 40: Baixa
        """)
    
    # Scatter
    st.markdown("### 📈 Regularidade vs DY")
    
    metrics_df = st.session_state.dividend_metrics
    scatter_df = metrics_df[['ticker', 'dividend_yield', 'regularity_index']].dropna()
    
    if not scatter_df.empty:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=scatter_df['regularity_index'],
            y=scatter_df['dividend_yield'] * 100,
            mode='markers+text',
            text=scatter_df['ticker'],
            textposition='top center',
            marker=dict(
                size=12,
                color=scatter_df['dividend_yield'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="DY"),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Reg: %{x:.1f}<br>DY: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Regularidade vs Dividend Yield",
            xaxis_title="Regularidade (0-100)",
            yaxis_title="DY (%)",
            template='plotly_dark',
            plot_bgcolor=ui.COLORS['background'],
            paper_bgcolor=ui.COLORS['background'],
            font=dict(color=ui.COLORS['text']),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quadrantes
        st.markdown("### 🎯 Quadrantes")
        
        med_reg = scatter_df['regularity_index'].median()
        med_dy = scatter_df['dividend_yield'].median()
        
        q1 = scatter_df[
            (scatter_df['regularity_index'] >= med_reg) & 
            (scatter_df['dividend_yield'] >= med_dy)
        ]['ticker'].tolist()
        
        q2 = scatter_df[
            (scatter_df['regularity_index'] >= med_reg) & 
            (scatter_df['dividend_yield'] < med_dy)
        ]['ticker'].tolist()
        
        q3 = scatter_df[
            (scatter_df['regularity_index'] < med_reg) & 
            (scatter_df['dividend_yield'] >= med_dy)
        ]['ticker'].tolist()
        
        if q1:
            st.success(f"🏆 **Alto DY + Alta Regularidade:** {', '.join(q1)}")
        if q2:
            st.info(f"📅 **Alta Regularidade:** {', '.join(q2)}")
        if q3:
            st.info(f"💰 **Alto DY:** {', '.join(q3)}")


def show_comparison():
    """Comparação entre ativos."""
    
    if not st.session_state.dividend_data or st.session_state.dividend_metrics.empty:
        return
    
    st.markdown("### ⚖️ Comparação")
    
    selected = st.multiselect(
        "Selecione até 5 ativos:",
        options=st.session_state.selected_tickers,
        max_selections=5
    )
    
    if len(selected) < 2:
        st.info("Selecione pelo menos 2 ativos")
        return
    
    comparison_df = st.session_state.dividend_metrics[
        st.session_state.dividend_metrics['ticker'].isin(selected)
    ].set_index('ticker')
    
    metrics_cols = [
        'dividend_yield',
        'regularity_index',
        'consistency_score',
        'uniformity_score',
        'num_payments'
    ]
    
    comp_display = comparison_df[metrics_cols].T
    
    rename = {
        'dividend_yield': 'DY',
        'regularity_index': 'Regularidade',
        'consistency_score': 'Consistência',
        'uniformity_score': 'Uniformidade',
        'num_payments': 'Pagamentos'
    }
    
    comp_display = comp_display.rename(index=rename)
    
    st.dataframe(comp_display, use_container_width=True)
    
    # Radar
    st.markdown("### 📊 Comparação Visual")
    fig = ui.plot_portfolio_comparison(comparison_df[metrics_cols], "Comparação")
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Função principal."""
    
    initialize_session_state()
    
    st.markdown('<p class="gradient-title">💸 Análise de Dividendos</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Análise completa de dividendos: histórico, regularidade e projeções.
    """)
    
    if not check_prerequisites():
        st.stop()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📊 {len(st.session_state.selected_tickers)} ativos selecionados")
    
    with col2:
        if st.button("🔙 Voltar", use_container_width=True):
            st.info("👈 Use o menu lateral para navegar")
    
    st.markdown("---")
    
    # Carregar
    load_dividend_data()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral",
        "📈 Histórico",
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
        show_dividend_calendar()
    
    with tab4:
        show_regularity_analysis()
    
    with tab5:
        show_comparison()
    
    # Próximos passos
    st.markdown("---")
    st.info("""
    **Próximo passo:** Use o menu lateral para:
    - 📊 Portfólios Eficientes
    - 🎯 Sharpe e MinVol
    - 📋 Resumo Executivo
    """)


if __name__ == "__main__":
    main()
