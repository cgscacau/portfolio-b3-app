"""
Página 4: Sharpe e MinVol
Otimizações específicas: Máximo Sharpe, Mínima Volatilidade e Dividendos Regulares
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics
from core.init import init_all

st.set_page_config(page_title="Sharpe e MinVol", page_icon="⚖️", layout="wide")

# INICIALIZAR
init_all()

def initialize_session_state():
    """Inicializa variáveis de sessão."""
    utils.ensure_session_state_initialized()


def check_prerequisites():
    """Verifica pré-requisitos."""
    if not st.session_state.selected_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para **Selecionar Ativos**")
        return False
    
    if st.session_state.expected_returns is None or st.session_state.expected_returns.empty:
        st.warning("⚠️ Parâmetros não calculados")
        st.info("👉 Vá para **Portfólios Eficientes** e calcule os parâmetros")
        return False
    
    return True


def optimize_max_sharpe():
    """Otimiza para máximo Sharpe."""
    
    st.markdown("### ⭐ Máximo Sharpe")
    
    st.markdown("""
    Busca a melhor relação retorno/risco. Ideal para **eficiência**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_weight = st.slider(
            "Peso máx/ativo (%):",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="sharpe_max_weight_slider"
        ) / 100
    
    with col2:
        min_weight = st.slider(
            "Peso mín/ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="sharpe_min_weight_slider"
        ) / 100
    
    apply_sector = st.checkbox(
        "Restrições setoriais",
        value=True,
        key="sharpe_sector_check"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máx/setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="sharpe_max_sector_slider"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            st.session_state.expected_returns.index.tolist(),
            max_sector
        )
    
    if st.button("⭐ Otimizar Sharpe", type="primary", use_container_width=True, key="btn_opt_sharpe"):
        
        with st.spinner("Otimizando..."):
            
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
                    st.error("❌ Não foi possível otimizar")
                    return
                
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                st.session_state.specialized_portfolios['Máximo Sharpe'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'max_sharpe'
                }
                
                st.success("✅ Máximo Sharpe otimizado!")
                
                show_portfolio_metrics(stats, weights, "Máximo Sharpe")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def optimize_min_volatility():
    """Otimiza para mínima volatilidade."""
    
    st.markdown("### 🛡️ Mínima Volatilidade")
    
    st.markdown("""
    Menor risco possível. Ideal para **conservadores**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_weight = st.slider(
            "Peso máx/ativo (%):",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="minvol_max_weight_slider"
        ) / 100
    
    with col2:
        min_weight = st.slider(
            "Peso mín/ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="minvol_min_weight_slider"
        ) / 100
    
    apply_sector = st.checkbox(
        "Restrições setoriais",
        value=True,
        key="minvol_sector_check"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máx/setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="minvol_max_sector_slider"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            st.session_state.expected_returns.index.tolist(),
            max_sector
        )
    
    if st.button("🛡️ Otimizar MinVol", type="primary", use_container_width=True, key="btn_opt_minvol"):
        
        with st.spinner("Otimizando..."):
            
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
                    st.error("❌ Não foi possível otimizar")
                    return
                
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                st.session_state.specialized_portfolios['Mínima Volatilidade'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'min_vol'
                }
                
                st.success("✅ Mínima Volatilidade otimizada!")
                
                show_portfolio_metrics(stats, weights, "Mínima Volatilidade")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def optimize_dividend_regularity():
    """Otimiza para dividendos regulares."""
    
    st.markdown("### 💸 Dividendos Regulares")
    
    st.markdown("""
    Maximiza yield com fluxo mensal consistente. Ideal para **renda passiva**.
    """)
    
    # Verificar dividendos
    if not st.session_state.dividend_data:
        st.warning("⚠️ Dados de dividendos não disponíveis")
        st.info("👉 Vá para **Análise de Dividendos** e carregue os dados")
        return
    
    # Preparar dados
    with st.spinner("Preparando dados de dividendos..."):
        
        expected_monthly_divs = {}
        div_monthly_series = {}
        
        for ticker, divs in st.session_state.dividend_data.items():
            if not divs.empty and ticker in st.session_state.price_data.columns:
                monthly = divs.resample('M').sum()
                
                if len(monthly) > 0:
                    avg_price = st.session_state.price_data[ticker].mean()
                    avg_monthly_div = monthly.mean()
                    
                    if avg_price > 0:
                        expected_monthly_divs[ticker] = avg_monthly_div / avg_price
                        div_monthly_series[ticker] = monthly
        
        if not expected_monthly_divs:
            st.warning("⚠️ Nenhum ativo com dividendos suficientes")
            return
        
        expected_monthly_divs_series = pd.Series(expected_monthly_divs)
        
        # Matriz de covariância dos fluxos
        all_dates = pd.DatetimeIndex([])
        for series in div_monthly_series.values():
            all_dates = all_dates.union(series.index)
        
        div_df = pd.DataFrame(index=all_dates.sort_values())
        for ticker, series in div_monthly_series.items():
            div_df[ticker] = series
        
        div_df = div_df.fillna(0)
        div_cov = div_df.cov()
    
    st.success(f"✅ {len(expected_monthly_divs)} ativos com dividendos")
    
    # Parâmetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lambda_penalty = st.slider(
            "Penalização (λ):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="div_lambda_slider",
            help="Maior = prioriza regularidade"
        )
    
    with col2:
        max_weight = st.slider(
            "Peso máx (%):",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
            key="div_max_weight_slider"
        ) / 100
    
    with col3:
        min_yield = st.slider(
            "Yield mín mensal (%):",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key="div_min_yield_slider"
        ) / 100
    
    apply_sector = st.checkbox(
        "Restrições setoriais",
        value=True,
        key="div_sector_check"
    )
    
    sector_constraints = None
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máx/setor (%):",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="div_max_sector_slider"
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            expected_monthly_divs_series.index.tolist(),
            max_sector
        )
    
    if st.button("💸 Otimizar Dividendos", type="primary", use_container_width=True, key="btn_opt_div"):
        
        with st.spinner("Otimizando..."):
            
            try:
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
                    st.error("❌ Não foi possível otimizar")
                    return
                
                stats = opt.calculate_portfolio_stats(
                    weights,
                    aligned_returns,
                    aligned_cov,
                    st.session_state.risk_free_rate
                )
                
                # Métricas de dividendos
                portfolio_monthly_yield = sum(
                    weights[t] * expected_monthly_divs_series[t] for t in weights.keys()
                )
                portfolio_annual_yield = portfolio_monthly_yield * 12
                
                w_array = np.array([weights.get(t, 0) for t in div_cov.index])
                portfolio_div_variance = np.dot(w_array, np.dot(div_cov.values, w_array))
                portfolio_div_std = np.sqrt(portfolio_div_variance)
                
                stats['monthly_yield'] = portfolio_monthly_yield
                stats['annual_yield'] = portfolio_annual_yield
                stats['dividend_volatility'] = portfolio_div_std
                
                st.session_state.specialized_portfolios['Dividendos Regulares'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'dividend_regularity'
                }
                
                st.success("✅ Dividendos Regulares otimizado!")
                
                show_portfolio_metrics(stats, weights, "Dividendos Regulares", include_dividends=True)
                
                # Projeção mensal
                st.markdown("### 📅 Projeção Mensal")
                
                dividend_metrics_obj = metrics.DividendMetrics(
                    st.session_state.dividend_data,
                    st.session_state.price_data
                )
                
                portfolio_monthly = dividend_metrics_obj.get_portfolio_monthly_dividends(weights)
                
                if not portfolio_monthly.empty:
                    fig = ui.plot_monthly_dividend_flow(
                        portfolio_monthly,
                        "Fluxo Mensal Projetado"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
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
                            help_text="Menor = mais regular",
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
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def show_portfolio_metrics(stats: dict, weights: dict, name: str, include_dividends: bool = False):
    """Exibe métricas de um portfólio."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.create_metric_card(
            "Retorno",
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
            "Sharpe",
            f"{stats['sharpe_ratio']:.3f}",
            icon="⭐"
        )
    
    with col4:
        ui.create_metric_card(
            "Nº Ativos",
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
                "Vol. Divs",
                f"{stats.get('dividend_volatility', 0):.4f}",
                help_text="Desvio padrão dos fluxos",
                icon="📊"
            )
    
    # Alocação
    st.markdown("### 📊 Alocação")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = ui.plot_portfolio_weights(weights, name)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weights_df = pd.DataFrame({
            'Ticker': list(weights.keys()),
            'Peso (%)': [w * 100 for w in weights.values()]
        }).sort_values('Peso (%)', ascending=False)
        
        st.dataframe(weights_df, use_container_width=True, height=400)


def compare_specialized_portfolios():
    """Compara portfólios especializados."""
    
    if not st.session_state.specialized_portfolios:
        st.info("ℹ️ Nenhum portfólio especializado criado")
        return
    
    st.markdown("### ⚖️ Comparação")
    
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
        
        if 'annual_yield' in stats:
            row['DY Anual (%)'] = stats['annual_yield'] * 100
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Tabela
    st.markdown("#### 📋 Tabela")
    
    display_df = comparison_df.copy()
    
    for col in display_df.columns:
        if col not in ['Portfólio', 'Nº Ativos']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Gráfico
    st.markdown("#### 📊 Risco vs Retorno")
    
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
            marker=dict(
                size=20,
                color=colors.get(name, ui.COLORS['info']),
                symbol='star',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f"<b>{name}</b><br>" +
                         'Ret: %{y:.2f}%<br>' +
                         'Vol: %{x:.2f}%<br>' +
                         f"Sharpe: {row['Sharpe']:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Comparação de Portfólios",
        xaxis_title="Volatilidade (%)",
        yaxis_title="Retorno (%)",
        template='plotly_dark',
        plot_bgcolor=ui.COLORS['background'],
        paper_bgcolor=ui.COLORS['background'],
        font=dict(color=ui.COLORS['text']),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes
    st.markdown("#### 🔍 Detalhes")
    
    selected = st.selectbox(
        "Selecione:",
        options=list(st.session_state.specialized_portfolios.keys()),
        key="comp_portfolio_select"
    )
    
    if selected:
        portfolio = st.session_state.specialized_portfolios[selected]
        weights = portfolio['weights']
        stats = portfolio['stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Composição")
            
            weights_df = pd.DataFrame({
                'Ticker': list(weights.keys()),
                'Peso (%)': [w * 100 for w in weights.values()]
            }).sort_values('Peso (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True, height=400)
        
        with col2:
            st.markdown("##### 📈 Métricas")
            
            metrics_display = {
                'Retorno': f"{stats['expected_return']*100:.2f}%",
                'Volatilidade': f"{stats['volatility']*100:.2f}%",
                'Sharpe': f"{stats['sharpe_ratio']:.3f}",
                'Nº Ativos': f"{stats['num_assets']}",
                'Peso Máx': f"{stats['max_weight']*100:.2f}%",
                'Peso Mín': f"{stats['min_weight']*100:.2f}%",
            }
            
            if 'annual_yield' in stats:
                metrics_display['DY Anual'] = f"{stats['annual_yield']*100:.2f}%"
            
            for metric, value in metrics_display.items():
                st.markdown(f"**{metric}:** {value}")
        
        # Download
        st.markdown("---")
        
        csv = weights_df.to_csv(index=False)
        st.download_button(
            "📥 Download",
            csv,
            f"{selected.replace(' ', '_')}.csv",
            use_container_width=True,
            key=f"btn_download_spec_{selected.replace(' ', '_')}"
        )


def show_risk_parity():
    """Risk Parity opcional."""
    
    st.markdown("### ⚖️ Risk Parity (Opcional)")
    
    st.markdown("""
    Contribuição igual de risco por ativo.
    """)
    
    with st.expander("ℹ️ Como funciona?", expanded=False):
        st.markdown("""
        **Risk Parity:** Ajusta pesos para que cada ativo contribua 
        igualmente para o risco total.
        
        **Vantagens:**
        - Diversificação mais efetiva
        - Reduz impacto de ativos voláteis
        
        **Desvantagens:**
        - Ignora retornos esperados
        - Pode concentrar em baixa volatilidade
        """)
    
    if st.button("⚖️ Criar Risk Parity", use_container_width=True, key="btn_create_rp"):
        
        with st.spinner("Otimizando..."):
            
            try:
                optimizer = opt.RiskParityOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                weights = optimizer.optimize(max_weight=0.50, min_weight=0.0)
                
                if not weights:
                    st.error("❌ Não foi possível otimizar")
                    return
                
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                st.session_state.specialized_portfolios['Risk Parity'] = {
                    'weights': weights,
                    'stats': stats,
                    'type': 'risk_parity'
                }
                
                st.success("✅ Risk Parity criado!")
                
                show_portfolio_metrics(stats, weights, "Risk Parity")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def main():
    """Função principal."""
    
    initialize_session_state()
    
    st.markdown('<p class="gradient-title">🎯 Sharpe e MinVol</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Portfólios especializados: **Máximo Sharpe**, **Mínima Volatilidade** e **Dividendos Regulares**.
    """)
    
    if not check_prerequisites():
        st.stop()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📊 {len(st.session_state.selected_tickers)} ativos")
    
    with col2:
        if st.button("🔙 Voltar", use_container_width=True, key="btn_back_page4"):
            st.info("👈 Use o menu lateral")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⭐ Sharpe",
        "🛡️ MinVol",
        "💸 Dividendos",
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
        show_risk_parity()
    
    with tab5:
        compare_specialized_portfolios()
    
    # Próximos
    st.markdown("---")
    st.info("""
    **Finalize:** Menu lateral (☰) →
    - 📋 Resumo Executivo
    """)


if __name__ == "__main__":
    main()
