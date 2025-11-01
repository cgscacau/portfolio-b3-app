"""
core/ui.py
Componentes reutilizáveis de interface para o Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Paleta de cores para tema escuro
COLORS = {
    'primary': '#00D9FF',
    'secondary': '#7B2FFF',
    'success': '#00FF88',
    'warning': '#FFB800',
    'danger': '#FF3366',
    'info': '#00A8FF',
    'background': '#0E1117',
    'surface': '#262730',
    'text': '#FAFAFA',
    'text_secondary': '#B0B0B0',
}

# Paleta para gráficos (múltiplas séries)
CHART_COLORS = [
    '#00D9FF', '#7B2FFF', '#00FF88', '#FFB800', '#FF3366',
    '#00A8FF', '#FF6B9D', '#C69EFF', '#00FFCC', '#FFDD57'
]


def create_metric_card(title: str, value: str, delta: Optional[str] = None,
                      help_text: Optional[str] = None, icon: str = "📊") -> None:
    """
    Cria card de métrica estilizado.
    
    Args:
        title: Título da métrica
        value: Valor principal
        delta: Variação (opcional)
        help_text: Texto de ajuda (opcional)
        icon: Emoji do ícone
    """
    delta_html = f'<p style="color: {COLORS["success"]}; font-size: 0.9rem; margin: 0;">{delta}</p>' if delta else ''
    help_html = f'<p style="color: {COLORS["text_secondary"]}; font-size: 0.8rem; margin-top: 0.5rem;">{help_text}</p>' if help_text else ''
    
    st.markdown(f"""
        <div style="
            background: {COLORS['surface']};
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <h4 style="margin: 0; color: {COLORS['text_secondary']}; font-size: 0.9rem;">{title}</h4>
            </div>
            <p style="font-size: 2rem; font-weight: bold; color: {COLORS['primary']}; margin: 0.5rem 0;">
                {value}
            </p>
            {delta_html}
            {help_html}
        </div>
    """, unsafe_allow_html=True)


def create_info_box(message: str, box_type: str = "info") -> None:
    """
    Cria caixa de informação estilizada.
    
    Args:
        message: Mensagem a exibir
        box_type: Tipo ('info', 'success', 'warning', 'danger')
    """
    color_map = {
        'info': COLORS['info'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'danger': COLORS['danger']
    }
    
    icon_map = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'danger': '❌'
    }
    
    color = color_map.get(box_type, COLORS['info'])
    icon = icon_map.get(box_type, 'ℹ️')
    
    st.markdown(f"""
        <div style="
            background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        ">
            <p style="margin: 0; color: {COLORS['text']};">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                {message}
            </p>
        </div>
    """, unsafe_allow_html=True)


def create_section_header(title: str, subtitle: Optional[str] = None,
                         icon: str = "📊") -> None:
    """
    Cria cabeçalho de seção estilizado.
    
    Args:
        title: Título da seção
        subtitle: Subtítulo (opcional)
        icon: Emoji do ícone
    """
    subtitle_html = f'<p style="color: {COLORS["text_secondary"]}; font-size: 1rem; margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ''
    
    st.markdown(f"""
        <div style="margin: 2rem 0 1rem 0;">
            <h2 style="
                color: {COLORS['text']};
                font-size: 1.8rem;
                font-weight: bold;
                margin: 0;
                display: flex;
                align-items: center;
            ">
                <span style="margin-right: 0.5rem;">{icon}</span>
                {title}
            </h2>
            {subtitle_html}
            <div style="
                height: 3px;
                background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
                border-radius: 2px;
                margin-top: 0.5rem;
                width: 100px;
            "></div>
        </div>
    """, unsafe_allow_html=True)


def create_tooltip(text: str, tooltip: str) -> None:
    """
    Cria texto com tooltip.
    
    Args:
        text: Texto principal
        tooltip: Texto do tooltip
    """
    st.markdown(f"""
        <div style="display: inline-block;">
            <span style="color: {COLORS['text']};">{text}</span>
            <span style="
                color: {COLORS['primary']};
                cursor: help;
                margin-left: 0.3rem;
            " title="{tooltip}">ⓘ</span>
        </div>
    """, unsafe_allow_html=True)


def plot_efficient_frontier(frontier_df: pd.DataFrame,
                           highlighted_portfolios: Optional[Dict[str, Tuple[float, float]]] = None,
                           title: str = "Fronteira Eficiente de Markowitz") -> go.Figure:
    """
    Plota fronteira eficiente.
    
    Args:
        frontier_df: DataFrame com colunas 'return', 'volatility', 'sharpe'
        highlighted_portfolios: Dict {nome: (retorno, volatilidade)}
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Fronteira eficiente
    fig.add_trace(go.Scatter(
        x=frontier_df['volatility'] * 100,
        y=frontier_df['return'] * 100,
        mode='lines',
        name='Fronteira Eficiente',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate='<b>Retorno:</b> %{y:.2f}%<br>' +
                     '<b>Volatilidade:</b> %{x:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Colorir por Sharpe
    if 'sharpe' in frontier_df.columns:
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'] * 100,
            y=frontier_df['return'] * 100,
            mode='markers',
            name='Pontos da Fronteira',
            marker=dict(
                size=8,
                color=frontier_df['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe<br>Ratio", x=1.15),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Retorno:</b> %{y:.2f}%<br>' +
                         '<b>Volatilidade:</b> %{x:.2f}%<br>' +
                         '<b>Sharpe:</b> %{marker.color:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Portfólios destacados
    if highlighted_portfolios:
        for name, (ret, vol) in highlighted_portfolios.items():
            fig.add_trace(go.Scatter(
                x=[vol * 100],
                y=[ret * 100],
                mode='markers+text',
                name=name,
                marker=dict(size=15, symbol='star', line=dict(width=2, color='white')),
                text=[name],
                textposition='top center',
                textfont=dict(size=12, color=COLORS['text']),
                hovertemplate=f'<b>{name}</b><br>' +
                             '<b>Retorno:</b> %{y:.2f}%<br>' +
                             '<b>Volatilidade:</b> %{x:.2f}%<br>' +
                             '<extra></extra>'
            ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno Anualizado (%)",
        template='plotly_dark',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(38, 39, 48, 0.8)',
            bordercolor=COLORS['primary'],
            borderwidth=1
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600
    )
    
    return fig


def plot_portfolio_weights(weights: Dict[str, float],
                          title: str = "Alocação do Portfólio",
                          show_percentage: bool = True) -> go.Figure:
    """
    Plota alocação do portfólio (pie chart ou bar chart).
    
    Args:
        weights: Dicionário {ticker: peso}
        title: Título do gráfico
        show_percentage: Se deve mostrar percentuais
    
    Returns:
        Figura Plotly
    """
    if not weights:
        return go.Figure()
    
    # Ordenar por peso
    sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    tickers = list(sorted_weights.keys())
    values = list(sorted_weights.values())
    
    # Se muitos ativos, usar bar chart; caso contrário, pie chart
    if len(tickers) > 10:
        # Bar chart
        fig = go.Figure(go.Bar(
            x=tickers,
            y=[v * 100 for v in values],
            marker=dict(
                color=values,
                colorscale='Viridis',
                line=dict(width=1, color='white')
            ),
            text=[f'{v*100:.1f}%' for v in values] if show_percentage else None,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Peso: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
            xaxis_title="Ativo",
            yaxis_title="Peso (%)",
            template='plotly_dark',
            showlegend=False,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text']),
            height=500
        )
    else:
        # Pie chart
        fig = go.Figure(go.Pie(
            labels=tickers,
            values=values,
            marker=dict(
                colors=CHART_COLORS[:len(tickers)],
                line=dict(width=2, color=COLORS['background'])
            ),
            textinfo='label+percent',
            textfont=dict(size=12, color=COLORS['text']),
            hovertemplate='<b>%{label}</b><br>' +
                         'Peso: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(38, 39, 48, 0.8)',
                bordercolor=COLORS['primary'],
                borderwidth=1
            ),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text']),
            height=500
        )
    
    return fig


def plot_cumulative_returns(returns_df: pd.DataFrame,
                           tickers: Optional[List[str]] = None,
                           title: str = "Retornos Cumulativos") -> go.Figure:
    """
    Plota retornos cumulativos de múltiplos ativos.
    
    Args:
        returns_df: DataFrame de retornos diários
        tickers: Lista de tickers a plotar (None = todos)
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if returns_df.empty:
        return go.Figure()
    
    if tickers is None:
        tickers = returns_df.columns.tolist()
    
    # Calcular retornos cumulativos
    cum_returns = (1 + returns_df[tickers]).cumprod() - 1
    
    fig = go.Figure()
    
    for idx, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[ticker] * 100,
            mode='lines',
            name=ticker,
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            hovertemplate='<b>' + ticker + '</b><br>' +
                         'Data: %{x|%d/%m/%Y}<br>' +
                         'Retorno: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        xaxis_title="Data",
        yaxis_title="Retorno Cumulativo (%)",
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(38, 39, 48, 0.8)',
            bordercolor=COLORS['primary'],
            borderwidth=1
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=500
    )
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                            title: str = "Matriz de Correlação") -> go.Figure:
    """
    Plota heatmap de correlação.
    
    Args:
        corr_matrix: Matriz de correlação
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if corr_matrix.empty:
        return go.Figure()
    
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=10),
        colorbar=dict(title="Correlação"),
        hovertemplate='<b>%{x} vs %{y}</b><br>' +
                     'Correlação: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        template='plotly_dark',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=max(400, len(corr_matrix) * 30),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_risk_return_scatter(metrics_df: pd.DataFrame,
                            title: str = "Risco vs Retorno",
                            color_by: str = 'sharpe_ratio') -> go.Figure:
    """
    Plota scatter de risco vs retorno.
    
    Args:
        metrics_df: DataFrame com colunas 'annualized_return', 'annualized_volatility', etc.
        title: Título do gráfico
        color_by: Coluna para colorir pontos
    
    Returns:
        Figura Plotly
    """
    if metrics_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics_df['annualized_volatility'] * 100,
        y=metrics_df['annualized_return'] * 100,
        mode='markers+text',
        text=metrics_df['ticker'],
        textposition='top center',
        textfont=dict(size=10, color=COLORS['text']),
        marker=dict(
            size=12,
            color=metrics_df[color_by] if color_by in metrics_df.columns else COLORS['primary'],
            colorscale='Viridis',
            showscale=True if color_by in metrics_df.columns else False,
            colorbar=dict(title=color_by.replace('_', ' ').title()),
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>%{text}</b><br>' +
                     'Retorno: %{y:.2f}%<br>' +
                     'Volatilidade: %{x:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno Anualizado (%)",
        template='plotly_dark',
        hovermode='closest',
        showlegend=False,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600
    )
    
    return fig


def plot_dividend_calendar(monthly_divs: pd.DataFrame,
                          title: str = "Calendário de Dividendos") -> go.Figure:
    """
    Plota heatmap de dividendos mensais.
    
    Args:
        monthly_divs: DataFrame com meses nas linhas e tickers nas colunas
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if monthly_divs.empty:
        return go.Figure()
    
    fig = go.Figure(go.Heatmap(
        z=monthly_divs.values,
        x=monthly_divs.columns,
        y=monthly_divs.index.strftime('%Y-%m'),
        colorscale='Greens',
        text=monthly_divs.values,
        texttemplate='R$ %{text:.2f}',
        textfont=dict(size=9),
        colorbar=dict(title="Dividendos<br>(R$)"),
        hovertemplate='<b>%{x}</b><br>' +
                     'Mês: %{y}<br>' +
                     'Dividendo: R$ %{z:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        template='plotly_dark',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=max(400, len(monthly_divs) * 25),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_monthly_dividend_flow(monthly_series: pd.Series,
                               title: str = "Fluxo Mensal de Dividendos") -> go.Figure:
    """
    Plota fluxo mensal de dividendos do portfólio.
    
    Args:
        monthly_series: Series com dividendos mensais
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if monthly_series.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_series.index.strftime('%Y-%m'),
        y=monthly_series.values,
        marker=dict(
            color=monthly_series.values,
            colorscale='Greens',
            line=dict(width=1, color='white')
        ),
        text=[f'R$ {v:.2f}' for v in monthly_series.values],
        textposition='outside',
        hovertemplate='Mês: %{x}<br>' +
                     'Dividendos: R$ %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Adicionar linha de média
    mean_div = monthly_series.mean()
    fig.add_hline(
        y=mean_div,
        line_dash="dash",
        line_color=COLORS['warning'],
        annotation_text=f"Média: R$ {mean_div:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        xaxis_title="Mês",
        yaxis_title="Dividendos (R$)",
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=500
    )
    
    return fig


def plot_drawdown(prices_series: pd.Series,
                 title: str = "Drawdown") -> go.Figure:
    """
    Plota drawdown de um ativo ou portfólio.
    
    Args:
        prices_series: Series de preços
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if prices_series.empty:
        return go.Figure()
    
    # Calcular drawdown
    running_max = prices_series.expanding().max()
    drawdown = (prices_series - running_max) / running_max * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Preço', 'Drawdown (%)'),
        row_heights=[0.6, 0.4]
    )
    
    # Preço
    fig.add_trace(
        go.Scatter(
            x=prices_series.index,
            y=prices_series.values,
            mode='lines',
            name='Preço',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='Data: %{x|%d/%m/%Y}<br>' +
                         'Preço: %{y:.2f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['danger'], width=2),
            fillcolor=f'rgba(255, 51, 102, 0.3)',
            hovertemplate='Data: %{x|%d/%m/%Y}<br>' +
                         'Drawdown: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        template='plotly_dark',
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600
    )
    
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig


def plot_portfolio_comparison(comparison_df: pd.DataFrame,
                             title: str = "Comparação de Portfólios") -> go.Figure:
    """
    Plota comparação de múltiplos portfólios (radar chart).
    
    Args:
        comparison_df: DataFrame com métricas por portfólio
        title: Título do gráfico
    
    Returns:
        Figura Plotly
    """
    if comparison_df.empty:
        return go.Figure()
    
    # Normalizar métricas para 0-1 (para radar chart)
    metrics = ['expected_return', 'sharpe_ratio', 'num_assets']
    
    # Volatilidade invertida (menor é melhor)
    if 'volatility' in comparison_df.columns:
        comparison_df['inv_volatility'] = 1 / (1 + comparison_df['volatility'])
        metrics.append('inv_volatility')
    
    normalized = comparison_df[metrics].copy()
    
    for col in metrics:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
    
    fig = go.Figure()
    
    for idx, portfolio in enumerate(comparison_df.index):
        fig.add_trace(go.Scatterpolar(
            r=normalized.loc[portfolio].values,
            theta=[m.replace('_', ' ').title() for m in metrics],
            fill='toself',
            name=portfolio,
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False
            ),
            bgcolor=COLORS['surface']
        ),
        title=dict(text=title, font=dict(size=20, color=COLORS['text'])),
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(38, 39, 48, 0.8)',
            bordercolor=COLORS['primary'],
            borderwidth=1
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600
    )
    
    return fig


def create_metrics_table(metrics_df: pd.DataFrame,
                        format_percentages: bool = True) -> None:
    """
    Exibe tabela de métricas estilizada.
    
    Args:
        metrics_df: DataFrame com métricas
        format_percentages: Se deve formatar como percentual
    """
    if metrics_df.empty:
        st.warning("Nenhuma métrica disponível")
        return
    
    # Formatar colunas
    styled_df = metrics_df.copy()
    
    percentage_cols = ['annualized_return', 'annualized_volatility', 'dividend_yield', 
                      'max_drawdown', 'dividend_yield_avg']
    
    for col in percentage_cols:
        if col in styled_df.columns and format_percentages:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else "N/A")
    
    ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'regularity_index']
    
    for col in ratio_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
    
    # Renomear colunas para português
    rename_map = {
        'ticker': 'Ticker',
        'annualized_return': 'Retorno Anual',
        'annualized_volatility': 'Volatilidade',
        'sharpe_ratio': 'Sharpe',
        'sortino_ratio': 'Sortino',
        'calmar_ratio': 'Calmar',
        'max_drawdown': 'Max Drawdown',
        'dividend_yield': 'Dividend Yield',
        'regularity_index': 'Regularidade',
        'num_payments': 'Nº Pagamentos'
    }
    
    styled_df = styled_df.rename(columns=rename_map)
    
    # Exibir com estilo
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(400, len(styled_df) * 35 + 38)
    )


def create_download_button(data: Union[pd.DataFrame, str],
                          filename: str,
                          label: str = "📥 Download",
                          file_type: str = "csv") -> None:
    """
    Cria botão de download estilizado.
    
    Args:
        data: DataFrame ou string para download
        filename: Nome do arquivo
        label: Texto do botão
        file_type: Tipo do arquivo ('csv', 'txt', 'json')
    """
    if isinstance(data, pd.DataFrame):
        if file_type == "csv":
            data_str = data.to_csv(index=False)
            mime = "text/csv"
        elif file_type == "json":
            data_str = data.to_json(orient='records', indent=2)
            mime = "application/json"
        else:
            data_str = str(data)
            mime = "text/plain"
    else:
        data_str = data
        mime = "text/plain"
    
    st.download_button(
        label=label,
        data=data_str,
        file_name=filename,
        mime=mime,
        use_container_width=True
    )


def show_loading_spinner(message: str = "Processando...") -> None:
    """
    Exibe spinner de carregamento.
    
    Args:
        message: Mensagem a exibir
    """
    with st.spinner(message):
        pass


def create_progress_indicator(current: int, total: int, message: str = "") -> None:
    """
    Cria indicador de progresso.
    
    Args:
        current: Valor atual
        total: Valor total
        message: Mensagem adicional
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{message} ({current}/{total})")
