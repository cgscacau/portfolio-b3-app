"""
Página 4: Sharpe e MinVol
Otimizações específicas: Máximo Sharpe, Mínima Volatilidade e Dividendos Regulares
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics, opt, ui
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Sharpe e MinVol - Portfolio B3",
    page_icon="🎯",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'price_data' not in st.session_state:
        st.session_state.price_data = pd.DataFrame()
    
    if 'expected_returns' not in st.session_state:
        st.session_state.expected_returns = pd.Series()
    
    if 'cov_matrix' not in st.session_state:
        st.session_state.cov_matrix = pd.DataFrame()
    
    if 'dividend_data' not in st.session_state:
        st.session_state.dividend_data = {}
    
    if 'specialized_portfolios' not in st.session_state:
        st.session_state.specialized_portfolios = {}


def check_prerequisites():
    """Verifica se há dados necessários."""
    if not st.session_state.selected_tickers:
        ui.create_info_box(
            "⚠️ Nenhum ativo selecionado. Por favor, vá para a página 'Selecionar Ativos' primeiro.",
            "warning"
        )
        
        if st.button("🎯 Ir para Seleção de Ativos", type="primary"):
            st.switch_page("app/pages/01_Selecionar_Ativos.py")
        
        return False
    
    if st.session_state.expected_returns.empty or st.session_state.cov_matrix.empty:
        ui.create_info_box(
            "⚠️ Parâmetros de otimização não calculados. Por favor, calcule na página 'Portfólios Eficientes'.",
            "warning"
        )
        
        if st.button("📊 Ir para Portfólios Eficientes", type="primary"):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
        
        return False
    
    return True


def optimize_max_sharpe():
    """Otimiza para máximo Sharpe ratio."""
    
    ui.create_section_header(
        "⭐ Portfólio de Máximo Sharpe",
        "Melhor relação risco-retorno ajustada",
        "⭐"
    )
    
    st.markdown("""
    O **Portfólio de Máximo Sharpe** busca a melhor relação entre retorno excedente 
    (acima da taxa livre de risco) e volatilidade. É ideal para investidores que 
    buscam **eficiência** na alocação.
    """)
    
    # Parâmetros
    col1, col2 = st.columns(2)
    
    with col1:
        max_weight = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="sharpe_max_weight"
        ) / 100
    
    with col2:
        min_weight = st.slider(
            "Peso mínimo por ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="sharpe_min_weight"
        ) / 100
    
    # Restrições setoriais
    apply_sector = st.checkbox(
        "Aplicar restrições setoriais",
        value=True,
        key="sharpe_sector_constraints"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="sharpe_max_sector"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            st.session_state.expected_returns.index.tolist(),
            max_sector
        )
    
    if st.button("⭐ Otimizar Máximo Sharpe", type="primary", use_container_width=True):
        
        with st.spinner("Otimizando para máximo Sharpe..."):
            
            try:
                optimizer = opt.MaxSharpeOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                weights = optimizer.optimize(
                    max_weight=max_weight,
                    min_weight=min_weight,
                    sector_constraints=sector_constraints
                )
                
                if not weights:
                    st.error("❌ Não foi possível otimizar com os parâmetros fornecidos")
                    return
                
                # Calcular estatísticas
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                # Salvar
                st.session_state.specialized_portfolios['Máximo Sharpe'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'max_sharpe'
                }
                
                st.success("✅ Portfólio de Máximo Sharpe otimizado!")
                
                # Exibir métricas
                show_portfolio_metrics(stats, weights, "Máximo Sharpe")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro na otimização Sharpe: {e}")
                st.error(f"❌ Erro na otimização: {e}")


def optimize_min_volatility():
    """Otimiza para mínima volatilidade."""
    
    ui.create_section_header(
        "🛡️ Portfólio de Mínima Volatilidade",
        "Menor risco possível",
        "🛡️"
    )
    
    st.markdown("""
    O **Portfólio de Mínima Volatilidade** busca o menor risco possível, 
    independente do retorno. É ideal para investidores **conservadores** 
    que priorizam preservação de capital.
    """)
    
    # Parâmetros
    col1, col2 = st.columns(2)
    
    with col1:
        max_weight = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="minvol_max_weight"
        ) / 100
    
    with col2:
        min_weight = st.slider(
            "Peso mínimo por ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="minvol_min_weight"
        ) / 100
    
    # Restrições setoriais
    apply_sector = st.checkbox(
        "Aplicar restrições setoriais",
        value=True,
        key="minvol_sector_constraints"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="minvol_max_sector"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            st.session_state.expected_returns.index.tolist(),
            max_sector
        )
    
    if st.button("🛡️ Otimizar Mínima Volatilidade", type="primary", use_container_width=True):
        
        with st.spinner("Otimizando para mínima volatilidade..."):
            
            try:
                optimizer = opt.MinVolatilityOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                weights = optimizer.optimize(
                    max_weight=max_weight,
                    min_weight=min_weight,
                    sector_constraints=sector_constraints
                )
                
                if not weights:
                    st.error("❌ Não foi possível otimizar com os parâmetros fornecidos")
                    return
                
                # Calcular estatísticas
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                # Salvar
                st.session_state.specialized_portfolios['Mínima Volatilidade'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'min_vol'
                }
                
                st.success("✅ Portfólio de Mínima Volatilidade otimizado!")
                
                # Exibir métricas
                show_portfolio_metrics(stats, weights, "Mínima Volatilidade")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro na otimização MinVol: {e}")
                st.error(f"❌ Erro na otimização: {e}")


def optimize_dividend_regularity():
    """Otimiza para dividendos regulares."""
    
    ui.create_section_header(
        "💸 Portfólio de Dividendos Regulares",
        "Fluxo mensal consistente de dividendos",
        "💸"
    )
    
    st.markdown("""
    O **Portfólio de Dividendos Regulares** busca maximizar o dividend yield 
    enquanto minimiza a variabilidade dos pagamentos mensais. Ideal para 
    investidores que buscam **renda passiva consistente**.
    """)
    
    # Verificar se há dados de dividendos
    if not st.session_state.dividend_data:
        ui.create_info_box(
            "⚠️ Dados de dividendos não disponíveis. Carregue os dados na página 'Análise de Dividendos'.",
            "warning"
        )
        
        if st.button("💸 Ir para Análise de Dividendos", type="primary"):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")
        
        return
    
    # Preparar dados de dividendos
    with st.spinner("Preparando dados de dividendos..."):
        
        # Calcular dividend yield mensal médio
        expected_monthly_divs = {}
        div_monthly_series = {}
        
        for ticker, divs in st.session_state.dividend_data.items():
            if not divs.empty and ticker in st.session_state.price_data.columns:
                # Dividendos mensais
                monthly = divs.resample('M').sum()
                
                if len(monthly) > 0:
                    # Yield mensal médio
                    avg_price = st.session_state.price_data[ticker].mean()
                    avg_monthly_div = monthly.mean()
                    
                    if avg_price > 0:
                        expected_monthly_divs[ticker] = avg_monthly_div / avg_price
                        div_monthly_series[ticker] = monthly
        
        if not expected_monthly_divs:
            st.warning("⚠️ Nenhum ativo com dados de dividendos suficientes")
            return
        
        # Converter para Series
        expected_monthly_divs_series = pd.Series(expected_monthly_divs)
        
        # Criar matriz de covariância dos fluxos mensais
        # Alinhar todas as séries temporais
        all_dates = pd.DatetimeIndex([])
        for series in div_monthly_series.values():
            all_dates = all_dates.union(series.index)
        
        div_df = pd.DataFrame(index=all_dates.sort_values())
        for ticker, series in div_monthly_series.items():
            div_df[ticker] = series
        
        div_df = div_df.fillna(0)
        
        # Covariância dos fluxos mensais
        div_cov = div_df.cov()
    
    st.success(f"✅ Dados preparados: {len(expected_monthly_divs)} ativos com dividendos")
    
    # Parâmetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lambda_penalty = st.slider(
            "Penalização da variância (λ):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Quanto maior, mais prioriza regularidade vs yield total"
        )
    
    with col2:
        max_weight = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
            key="div_max_weight"
        ) / 100
    
    with col3:
        min_yield = st.slider(
            "Yield mínimo mensal (%):",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Yield mensal mínimo do portfólio"
        ) / 100
    
    # Restrições setoriais
    apply_sector = st.checkbox(
        "Aplicar restrições setoriais",
        value=True,
        key="div_sector_constraints"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="div_max_sector"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            expected_monthly_divs_series.index.tolist(),
            max_sector
        )
    
    if st.button("💸 Otimizar Dividendos Regulares", type="primary", use_container_width=True):
        
        with st.spinner("Otimizando para dividendos regulares..."):
            
            try:
                # Alinhar retornos e covariância de preços com ativos que têm dividendos
                aligned_tickers = expected_monthly_divs_series.index.tolist()
                aligned_returns = st.session_state.expected_returns[aligned_tickers]
                aligned_cov = st.session_state.cov_matrix.loc[aligned_tickers, aligned_tickers]
                
                optimizer = opt.DividendRegularityOptimizer(
                    expected_monthly_divs_series,
                    div_cov,
                    aligned_returns,
                    aligned_cov
                )
                
                weights = optimizer.optimize(
                    lambda_penalty=lambda_penalty,
                    max_weight=max_weight,
                    min_weight=0.0,
                    min_yield=min_yield if min_yield > 0 else None,
                    sector_constraints=sector_constraints
                )
                
                if not weights:
                    st.error("❌ Não foi possível otimizar com os parâmetros fornecidos")
                    return
                
                # Calcular estatísticas de preços
                stats = opt.calculate_portfolio_stats(
                    weights,
                    aligned_returns,
                    aligned_cov,
                    st.session_state.risk_free_rate
                )
                
                # Calcular estatísticas de dividendos
                portfolio_monthly_yield = sum(weights[t] * expected_monthly_divs_series[t] for t in weights.keys())
                portfolio_annual_yield = portfolio_monthly_yield * 12
                
                # Variância dos fluxos mensais
                w_array = np.array([weights.get(t, 0) for t in div_cov.index])
                portfolio_div_variance = np.dot(w_array, np.dot(div_cov.values, w_array))
                portfolio_div_std = np.sqrt(portfolio_div_variance)
                
                # Adicionar métricas de dividendos
                stats['monthly_yield'] = portfolio_monthly_yield
                stats['annual_yield'] = portfolio_annual_yield
                stats['dividend_volatility'] = portfolio_div_std
                
                # Salvar
                st.session_state.specialized_portfolios['Dividendos Regulares'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'dividend_regularity'
                }
                
                st.success("✅ Portfólio de Dividendos Regulares otimizado!")
                
                # Exibir métricas
                show_portfolio_metrics(stats, weights, "Dividendos Regulares", include_dividends=True)
                
                # Projeção de fluxo mensal
                st.markdown("### 📅 Projeção de Fluxo Mensal")
                
                dividend_metrics_obj = metrics.DividendMetrics(
                    st.session_state.dividend_data,
                    st.session_state.price_data
                )
                
                portfolio_monthly = dividend_metrics_obj.get_portfolio_monthly_dividends(weights)
                
                if not portfolio_monthly.empty:
                    fig = ui.plot_monthly_dividend_flow(
                        portfolio_monthly,
                        "Fluxo Mensal Projetado - Portfólio de Dividendos Regulares"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estatísticas do fluxo
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ui.create_metric_card(
                            "Média Mensal",
                            f"R$ {portfolio_monthly.mean():.2f}",
                            icon="💰"
                        )
                    
                    with col2:
                        cv = portfolio_monthly.std() / portfolio_monthly.mean() if portfolio_monthly.mean() > 0 else 0
                        ui.create_metric_card(
                            "Coef. Variação",
                            f"{cv:.3f}",
                            help_text="Quanto menor, mais regular",
                            icon="📊"
                        )
                    
                    with col3:
                        ui.create_metric_card(
                            "Total Anual",
                            f"R$ {portfolio_monthly.sum():.2f}",
                            icon="💵"
                        )
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro na otimização de dividendos: {e}")
                st.error(f"❌ Erro na otimização: {e}")


def show_portfolio_metrics(stats: dict, weights: dict, portfolio_name: str, include_dividends: bool = False):
    """Exibe métricas de um portfólio."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_metric_card(
            "Retorno Esperado",
            f"{stats['expected_return']*100:.2f}%",
            icon="📈"
        )
    
    with col2:
        ui.create_metric_card(
            "Volatilidade",
            f"{stats['volatility']*100:.2f}%",
            icon="📊"
        )
    
    with col3:
        ui.create_metric_card(
            "Sharpe Ratio",
            f"{stats['sharpe_ratio']:.3f}",
            icon="⭐"
        )
    
    with col4:
        ui.create_metric_card(
            "Nº de Ativos",
            f"{stats['num_assets']}",
            icon="🎯"
        )
    
    if include_dividends:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui.create_metric_card(
                "Yield Mensal",
                f"{stats.get('monthly_yield', 0)*100:.2f}%",
                icon="💰"
            )
        
        with col2:
            ui.create_metric_card(
                "Yield Anual",
                f"{stats.get('annual_yield', 0)*100:.2f}%",
                icon="💵"
            )
        
        with col3:
            ui.create_metric_card(
                "Volatilidade Divs",
                f"{stats.get('dividend_volatility', 0):.4f}",
                help_text="Desvio padrão dos fluxos mensais",
                icon="📊"
            )
    
    # Alocação
    st.markdown("### 📊 Alocação do Portfólio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = ui.plot_portfolio_weights(weights, f"Alocação - {portfolio_name}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weights_df = pd.DataFrame({
            'Ticker': list(weights.keys()),
            'Peso (%)': [w * 100 for w in weights.values()]
        })
        weights_df = weights_df.sort_values('Peso (%)', ascending=False)
        
        st.dataframe(weights_df, use_container_width=True, height=400)


def compare_specialized_portfolios():
    """Compara os portfólios especializados."""
    
    if not st.session_state.specialized_portfolios:
        ui.create_info_box(
            "Nenhum portfólio especializado criado ainda. Use as ferramentas acima para otimizar.",
            "info"
        )
        return
    
    ui.create_section_header(
        "⚖️ Comparação de Portfólios",
        "Análise lado a lado dos portfólios especializados",
        "⚖️"
    )
    
    # Criar DataFrame de comparação
    comparison_data = []
    
    for name, portfolio in st.session_state.specialized_portfolios.items():
        stats = portfolio['stats']
        
        row = {
            'Portfólio': name,
            'Retorno (%)': stats['expected_return'] * 100,
            'Volatilidade (%)': stats['volatility'] * 100,
            'Sharpe': stats['sharpe_ratio'],
            'Nº Ativos': stats['num_assets'],
            'Peso Máx (%)': stats['max_weight'] * 100,
        }
        
        # Adicionar métricas de dividendos se disponível
        if 'annual_yield' in stats:
            row['DY Anual (%)'] = stats['annual_yield'] * 100
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Tabela formatada
    st.markdown("### 📋 Tabela Comparativa")
    
    display_df = comparison_df.copy()
    
    for col in display_df.columns:
        if col != 'Portfólio' and col != 'Nº Ativos':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Gráfico scatter
    st.markdown("### 📊 Risco vs Retorno")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    colors = {
        'Máximo Sharpe': ui.COLORS['primary'],
        'Mínima Volatilidade': ui.COLORS['success'],
        'Dividendos Regulares': ui.COLORS['warning']
    }
    
    for idx, row in comparison_df.iterrows():
        name = row['Portfólio']
        
        fig.add_trace(go.Scatter(
            x=[row['Volatilidade (%)']],
            y=[row['Retorno (%)']],
            mode='markers+text',
            name=name,
            text=[name],
            textposition='top center',
            textfont=dict(size=12, color=ui.COLORS['text']),
            marker=dict(
                size=20,
                color=colors.get(name, ui.COLORS['info']),
                symbol='star',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f"<b>{name}</b><br>" +
                         'Retorno: %{y:.2f}%<br>' +
                         'Volatilidade: %{x:.2f}%<br>' +
                         f"Sharpe: {row['Sharpe']:.3f}<br>" +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Comparação de Portfólios Especializados",
        xaxis_title="Volatilidade (%)",
        yaxis_title="Retorno (%)",
        template='plotly_dark',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(38, 39, 48, 0.8)',
            bordercolor=ui.COLORS['primary'],
            borderwidth=1
        ),
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise detalhada
    st.markdown("### 🔍 Análise Detalhada")
    
    selected_portfolio = st.selectbox(
        "Selecione um portfólio para ver detalhes:",
        options=list(st.session_state.specialized_portfolios.keys())
    )
    
    if selected_portfolio:
        portfolio = st.session_state.specialized_portfolios[selected_portfolio]
        weights = portfolio['weights']
        stats = portfolio['stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Composição")
            
            weights_df = pd.DataFrame({
                'Ticker': list(weights.keys()),
                'Peso (%)': [w * 100 for w in weights.values()]
            })
            weights_df = weights_df.sort_values('Peso (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True, height=400)
        
        with col2:
            st.markdown("#### 📈 Métricas")
            
            metrics_display = {
                'Retorno Esperado': f"{stats['expected_return']*100:.2f}%",
                'Volatilidade': f"{stats['volatility']*100:.2f}%",
                'Sharpe Ratio': f"{stats['sharpe_ratio']:.3f}",
                'Número de Ativos': f"{stats['num_assets']}",
                'Peso Máximo': f"{stats['max_weight']*100:.2f}%",
                'Peso Mínimo': f"{stats['min_weight']*100:.2f}%",
                'Nº Efetivo de Ativos': f"{stats.get('effective_n', 0):.2f}"
            }
            
            if 'annual_yield' in stats:
                metrics_display['Dividend Yield Anual'] = f"{stats['annual_yield']*100:.2f}%"
            
            for metric, value in metrics_display.items():
                st.markdown(f"**{metric}:** {value}")
        
        # Download
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ui.create_download_button(
                weights_df,
                f"{selected_portfolio.replace(' ', '_')}_weights.csv",
                "📥 Download Alocação",
                "csv"
            )
        
        with col2:
            # Criar relatório completo
            report_data = {
                'Portfólio': selected_portfolio,
                **stats,
                'Pesos': weights
            }
            
            report_df = pd.DataFrame([report_data])
            
            ui.create_download_button(
                report_df,
                f"{selected_portfolio.replace(' ', '_')}_report.json",
                "📥 Download Relatório Completo",
                "json"
            )


def show_risk_parity_option():
    """Opção de criar portfólio Risk Parity."""
    
    ui.create_section_header(
        "⚖️ Portfólio Risk Parity (Opcional)",
        "Contribuição igual de risco por ativo",
        "⚖️"
    )
    
    st.markdown("""
    O **Portfólio Risk Parity** aloca pesos de forma que cada ativo contribua 
    igualmente para o risco total do portfólio. É uma alternativa ao equally weighted 
    que considera as diferenças de volatilidade entre ativos.
    """)
    
    with st.expander("ℹ️ Como funciona o Risk Parity?", expanded=False):
        st.markdown("""
        ### Conceito
        
        Em vez de pesos iguais (1/N), o Risk Parity ajusta os pesos para que:

        
        $$\\text{Contribuição de Risco}_i = \\text{Peso}_i \\times \\text{Risco Marginal}_i$$
        
        Todos os ativos contribuem igualmente para a volatilidade total do portfólio.
        
        ### Vantagens
        - Diversificação mais efetiva que equally weighted
        - Reduz impacto de ativos muito voláteis
        - Aumenta exposição a ativos menos voláteis
        
        ### Desvantagens
        - Pode concentrar em ativos de baixa volatilidade
        - Ignora retornos esperados
        - Pode ter turnover alto em rebalanceamentos
        """)
    
    if st.button("⚖️ Criar Portfólio Risk Parity", use_container_width=True):
        
        with st.spinner("Otimizando Risk Parity..."):
            
            try:
                optimizer = opt.RiskParityOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                weights = optimizer.optimize(max_weight=0.50, min_weight=0.0)
                
                if not weights:
                    st.error("❌ Não foi possível otimizar Risk Parity")
                    return
                
                # Calcular estatísticas
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                # Salvar
                st.session_state.specialized_portfolios['Risk Parity'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'risk_parity'
                }
                
                st.success("✅ Portfólio Risk Parity criado!")
                
                show_portfolio_metrics(stats, weights, "Risk Parity")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro na otimização Risk Parity: {e}")
                st.error(f"❌ Erro na otimização: {e}")


def main():
    """Função principal da página."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<p class="gradient-title">🎯 Sharpe e MinVol</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Portfólios especializados com objetivos específicos: **Máximo Sharpe** (eficiência), 
    **Mínima Volatilidade** (conservadorismo) e **Dividendos Regulares** (renda mensal).
    """)
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        st.stop()
    
    # Informações
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"📊 **{len(st.session_state.selected_tickers)} ativos** disponíveis para otimização")
    
    with col2:
        if st.button("🔙 Voltar", use_container_width=True):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⭐ Máximo Sharpe",
        "🛡️ Mínima Volatilidade",
        "💸 Dividendos Regulares",
        "⚖️ Risk Parity",
        "📊 Comparação"
    ])
    
    with tab1:
        optimize_max_sharpe()
    
    with tab2:
        optimize_min_volatility()
    
    with tab3:
        optimize_dividend_regularity()
    
    with tab4:
        show_risk_parity_option()
    
    with tab5:
        compare_specialized_portfolios()
    
    # Próximos passos
    st.markdown("---")
    
    ui.create_section_header(
        "🚀 Próximos Passos",
        "Finalize com o resumo executivo",
        "🚀"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Resumo Executivo", use_container_width=True, type="primary"):
            st.switch_page("app/pages/05_Resumo_Executivo.py")
    
    with col2:
        if st.button("📊 Voltar para Fronteira", use_container_width=True):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
    
    with col3:
        if st.button("💸 Voltar para Dividendos", use_container_width=True):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")


if __name__ == "__main__":
    main()
