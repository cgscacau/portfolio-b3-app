"""
Página 3: Portfólios Eficientes
Otimização de Markowitz e fronteira eficiente
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics, opt, filters, ui, utils
import logging

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Portfólios Eficientes - Portfolio B3",
    page_icon="📊",
    layout="wide"
)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    utils.ensure_session_state_initialized()


def check_prerequisites():
    """Verifica pré-requisitos."""
    if not st.session_state.selected_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para **Selecionar Ativos** no menu lateral")
        return False
    
    if st.session_state.price_data is None or st.session_state.price_data.empty:
        st.warning("⚠️ Dados não carregados")
        st.info("👉 Vá para **Análise de Dividendos** e carregue os dados")
        return False
    
    return True


def calculate_portfolio_inputs():
    """Calcula retornos esperados e covariância."""
    
    st.markdown("### 🧮 Cálculo de Parâmetros")
    
    if st.session_state.price_data.empty:
        st.error("❌ Sem dados de preços")
        return False
    
    prices_df = st.session_state.price_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Ativos:** {len(prices_df.columns)}")
    
    with col2:
        st.info(f"📅 **Período:** {len(prices_df)} dias")
    
    with col3:
        years = len(prices_df) / 252
        st.info(f"⏱️ **Anos:** {years:.1f}")
    
    if st.button("🔄 Calcular", type="primary", use_container_width=True, key="btn_calc_params"):
        
        with st.spinner("Calculando..."):
            
            perf_metrics = metrics.PerformanceMetrics(
                prices_df,
                risk_free_rate=st.session_state.risk_free_rate
            )
            
            # Retornos anualizados
            expected_returns = pd.Series({
                ticker: perf_metrics.calculate_annualized_return(ticker) 
                for ticker in prices_df.columns
            }).dropna()
            
            # Covariância anualizada
            cov_matrix = perf_metrics.get_covariance_matrix(annualized=True)
            
            # Alinhar
            common = expected_returns.index.intersection(cov_matrix.index)
            expected_returns = expected_returns[common]
            cov_matrix = cov_matrix.loc[common, common]
            
            st.session_state.expected_returns = expected_returns
            st.session_state.cov_matrix = cov_matrix
            
            st.success("✅ Parâmetros calculados!")
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ui.create_metric_card(
                    "Retorno Médio",
                    f"{expected_returns.mean()*100:.2f}%",
                    icon="📈"
                )
            
            with col2:
                ui.create_metric_card(
                    "Retorno Máx",
                    f"{expected_returns.max()*100:.2f}%",
                    icon="🔝"
                )
            
            with col3:
                ui.create_metric_card(
                    "Retorno Mín",
                    f"{expected_returns.min()*100:.2f}%",
                    icon="📉"
                )
            
            with col4:
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                
                ui.create_metric_card(
                    "Corr. Média",
                    f"{avg_corr:.3f}",
                    icon="🔗"
                )
            
            st.rerun()
    
    return True


def show_input_statistics():
    """Estatísticas dos dados."""
    
    if st.session_state.expected_returns is None or st.session_state.expected_returns.empty:
        st.info("ℹ️ Calcule os parâmetros acima")
        return
    
    st.markdown("### 📊 Estatísticas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Retornos")
        
        returns_df = pd.DataFrame({
            'Ticker': st.session_state.expected_returns.index,
            'Retorno (%)': st.session_state.expected_returns.values * 100
        }).sort_values('Retorno (%)', ascending=False)
        
        returns_df['Retorno (%)'] = returns_df['Retorno (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(returns_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("#### 🔗 Correlação")
        
        std_devs = np.sqrt(np.diag(st.session_state.cov_matrix))
        corr_matrix = st.session_state.cov_matrix / np.outer(std_devs, std_devs)
        
        fig = ui.plot_correlation_heatmap(corr_matrix, "Correlação")
        st.plotly_chart(fig, use_container_width=True)


def compute_efficient_frontier():
    """Computa fronteira eficiente."""
    
    if st.session_state.expected_returns is None or st.session_state.expected_returns.empty:
        return
    
    st.markdown("### 🎯 Fronteira Eficiente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_points = st.slider(
            "Pontos na fronteira:",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            key="frontier_n_points"
        )
    
    with col2:
        max_weight = st.slider(
            "Peso máx/ativo (%):",
            min_value=5,
            max_value=100,
            value=int(st.session_state.max_weight_per_asset * 100),
            step=5,
            key="frontier_max_weight"
        ) / 100
    
    with col3:
        min_weight = st.slider(
            "Peso mín/ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="frontier_min_weight"
        ) / 100
    
    # Restrições setoriais
    apply_sector = st.checkbox(
        "Aplicar restrições setoriais",
        value=True,
        key="frontier_sector_check"
    )
    
    if apply_sector and not st.session_state.universe_df.empty:
        max_sector = st.slider(
            "Peso máx/setor (%):",
            min_value=10,
            max_value=100,
            value=int(st.session_state.max_weight_per_sector * 100),
            step=5,
            key="frontier_max_sector"
        ) / 100
    
    if st.button("🚀 Calcular Fronteira", type="primary", use_container_width=True, key="btn_calc_frontier"):
        
        with st.spinner("Calculando fronteira... Pode levar alguns minutos."):
            
            try:
                optimizer = opt.MarkowitzOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                frontier_df = optimizer.compute_efficient_frontier(
                    n_points=n_points,
                    max_weight=max_weight,
                    min_weight=min_weight
                )
                
                if frontier_df.empty:
                    st.error("❌ Erro ao calcular fronteira")
                    return
                
                st.session_state.efficient_frontier = frontier_df
                
                st.success(f"✅ Fronteira com {len(frontier_df)} pontos!")
                
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    min_vol = frontier_df['volatility'].min()
                    ui.create_metric_card("Mín Vol", f"{min_vol*100:.2f}%", icon="🛡️")
                
                with col2:
                    max_ret = frontier_df['return'].max()
                    ui.create_metric_card("Máx Ret", f"{max_ret*100:.2f}%", icon="📈")
                
                with col3:
                    max_sharpe = frontier_df['sharpe'].max()
                    ui.create_metric_card("Máx Sharpe", f"{max_sharpe:.3f}", icon="⭐")
                
                with col4:
                    max_sharpe_idx = frontier_df['sharpe'].idxmax()
                    sharpe_ret = frontier_df.loc[max_sharpe_idx, 'return']
                    ui.create_metric_card("Ret (Sharpe)", f"{sharpe_ret*100:.2f}%", icon="🎯")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def show_efficient_frontier_plot():
    """Exibe gráfico da fronteira."""
    
    if st.session_state.efficient_frontier is None or st.session_state.efficient_frontier.empty:
        st.info("ℹ️ Calcule a fronteira acima")
        return
    
    st.markdown("### 📊 Visualização")
    
    frontier_df = st.session_state.efficient_frontier
    
    # Identificar pontos especiais
    max_sharpe_idx = frontier_df['sharpe'].idxmax()
    min_vol_idx = frontier_df['volatility'].idxmin()
    
    highlighted = {
        'Máximo Sharpe': (
            frontier_df.loc[max_sharpe_idx, 'return'],
            frontier_df.loc[max_sharpe_idx, 'volatility']
        ),
        'Mínima Vol': (
            frontier_df.loc[min_vol_idx, 'return'],
            frontier_df.loc[min_vol_idx, 'volatility']
        )
    }
    
    fig = ui.plot_efficient_frontier(frontier_df, highlighted, "Fronteira Eficiente")
    st.plotly_chart(fig, use_container_width=True)
    
    # Explicação
    with st.expander("ℹ️ Como interpretar?", expanded=False):
        st.markdown("""
        **Fronteira Eficiente:** Portfólios com máximo retorno para cada nível de risco.
        
        **Pontos-chave:**
        - **Máximo Sharpe** ⭐: Melhor risco-retorno
        - **Mínima Vol** 🛡️: Menor risco possível
        
        **Escolha:**
        - **Conservador**: Próximo à Mínima Vol
        - **Balanceado**: Próximo ao Máximo Sharpe
        - **Agressivo**: Maior retorno (mais risco)
        """)


def optimize_target_portfolio():
    """Otimiza para alvo específico."""
    
    if st.session_state.expected_returns is None or st.session_state.expected_returns.empty:
        return
    
    st.markdown("### 🎯 Portfólio Alvo")
    
    opt_type = st.radio(
        "Tipo:",
        ["Retorno Alvo", "Risco Alvo"],
        horizontal=True,
        key="target_opt_type"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if opt_type == "Retorno Alvo":
            target_return = st.slider(
                "Retorno alvo (%):",
                min_value=float(st.session_state.expected_returns.min() * 100),
                max_value=float(st.session_state.expected_returns.max() * 100),
                value=float(st.session_state.expected_returns.mean() * 100),
                step=0.5,
                key="target_return_slider"
            ) / 100
        else:
            min_vol = st.session_state.cov_matrix.values.diagonal().min() ** 0.5
            max_vol = st.session_state.cov_matrix.values.diagonal().max() ** 0.5
            
            target_vol = st.slider(
                "Volatilidade alvo (%):",
                min_value=float(min_vol * 100),
                max_value=float(max_vol * 100),
                value=float((min_vol + max_vol) / 2 * 100),
                step=0.5,
                key="target_vol_slider"
            ) / 100
    
    with col2:
        max_weight_target = st.slider(
            "Peso máx (%):",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
            key="target_max_weight_slider"
        ) / 100
    
    if st.button("🎯 Otimizar", type="primary", use_container_width=True, key="btn_optimize_target"):
        
        with st.spinner("Otimizando..."):
            
            try:
                optimizer = opt.MarkowitzOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                if opt_type == "Retorno Alvo":
                    weights = optimizer.optimize_for_return(
                        target_return=target_return,
                        max_weight=max_weight_target
                    )
                else:
                    weights = optimizer.optimize_for_risk(
                        target_volatility=target_vol,
                        max_weight=max_weight_target
                    )
                
                if not weights:
                    st.error("❌ Não foi possível otimizar")
                    return
                
                # Stats
                stats = opt.calculate_portfolio_stats(
                    weights,
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                # Salvar
                st.session_state.optimized_portfolios['Portfólio Alvo'] = {
                    'weights': weights,
                    'stats': stats
                }
                
                st.success("✅ Otimizado!")
                
                # Métricas
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
                
                # Alocação
                st.markdown("### 📊 Alocação")
                
                fig = ui.plot_portfolio_weights(weights, "Portfólio Alvo")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                weights_df = pd.DataFrame({
                    'Ticker': list(weights.keys()),
                    'Peso (%)': [w * 100 for w in weights.values()]
                }).sort_values('Peso (%)', ascending=False)
                
                st.dataframe(weights_df, use_container_width=True)
            
            except Exception as e:
                logger.error(f"Erro: {e}")
                st.error(f"❌ Erro: {e}")


def show_saved_portfolios():
    """Portfólios salvos."""
    
    if not st.session_state.optimized_portfolios:
        st.info("ℹ️ Nenhum portfólio otimizado ainda")
        return
    
    st.markdown("### 💼 Portfólios Salvos")
    
    # Comparação
    comparison_data = []
    
    for name, portfolio in st.session_state.optimized_portfolios.items():
        stats = portfolio['stats']
        comparison_data.append({
            'Portfólio': name,
            'Retorno': stats['expected_return'],
            'Volatilidade': stats['volatility'],
            'Sharpe': stats['sharpe_ratio'],
            'Nº Ativos': stats['num_assets']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Formatar
    display_df = comparison_df.copy()
    display_df['Retorno'] = display_df['Retorno'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Volatilidade'] = display_df['Volatilidade'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Gráfico
    if len(comparison_df) > 1:
        st.markdown("### 📊 Comparação")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for idx, row in comparison_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Volatilidade'] * 100],
                y=[row['Retorno'] * 100],
                mode='markers+text',
                name=row['Portfólio'],
                text=[row['Portfólio']],
                textposition='top center',
                marker=dict(size=15, line=dict(width=2, color='white')),
                hovertemplate=f"<b>{row['Portfólio']}</b><br>" +
                             'Ret: %{y:.2f}%<br>' +
                             'Vol: %{x:.2f}%<br>' +
                             f"Sharpe: {row['Sharpe']:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Comparação",
            xaxis_title="Volatilidade (%)",
            yaxis_title="Retorno (%)",
            template='plotly_dark',
            plot_bgcolor=ui.COLORS['background'],
            paper_bgcolor=ui.COLORS['background'],
            font=dict(color=ui.COLORS['text']),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes
    st.markdown("### 📋 Detalhes")
    
    selected = st.selectbox(
        "Selecione um portfólio:",
        options=list(st.session_state.optimized_portfolios.keys()),
        key="portfolio_detail_select"
    )
    
    if selected:
        portfolio = st.session_state.optimized_portfolios[selected]
        weights = portfolio['weights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Alocação")
            fig = ui.plot_portfolio_weights(weights, selected)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Pesos")
            
            weights_df = pd.DataFrame({
                'Ticker': list(weights.keys()),
                'Peso (%)': [w * 100 for w in weights.values()]
            }).sort_values('Peso (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True, height=400)
        
        # Download
        csv = weights_df.to_csv(index=False)
        st.download_button(
            "📥 Download",
            csv,
            f"{selected.replace(' ', '_')}.csv",
            use_container_width=True,
            key=f"btn_download_{selected.replace(' ', '_')}"
        )


def show_equal_weight():
    """Portfólio equally weighted."""
    
    if st.session_state.expected_returns is None or st.session_state.expected_returns.empty:
        return
    
    st.markdown("### ⚖️ Equally Weighted (Baseline)")
    
    st.markdown("""
    Alocação uniforme como **baseline** para comparação.
    """)
    
    if st.button("⚖️ Criar", use_container_width=True, key="btn_create_ew"):
        
        tickers = st.session_state.expected_returns.index.tolist()
        
        ew_optimizer = opt.EqualWeightOptimizer(tickers)
        weights = ew_optimizer.optimize()
        
        stats = opt.calculate_portfolio_stats(
            weights,
            st.session_state.expected_returns,
            st.session_state.cov_matrix,
            st.session_state.risk_free_rate
        )
        
        st.session_state.optimized_portfolios['Equally Weighted'] = {
            'weights': weights,
            'stats': stats
        }
        
        st.success("✅ Equally Weighted criado!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ui.create_metric_card("Retorno", f"{stats['expected_return']*100:.2f}%", icon="📈")
        
        with col2:
            ui.create_metric_card("Volatilidade", f"{stats['volatility']*100:.2f}%", icon="📊")
        
        with col3:
            ui.create_metric_card("Sharpe", f"{stats['sharpe_ratio']:.3f}", icon="⭐")
        
        with col4:
            ui.create_metric_card("Peso/Ativo", f"{100/len(tickers):.2f}%", icon="⚖️")
        
        st.rerun()


def main():
    """Função principal."""
    
    initialize_session_state()
    
    st.markdown('<p class="gradient-title">📊 Portfólios Eficientes</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Otimização via **Teoria Moderna de Portfólio** (Markowitz).
    """)
    
    if not check_prerequisites():
        st.stop()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📊 {len(st.session_state.selected_tickers)} ativos prontos")
    
    with col2:
        if st.button("🔙 Voltar", use_container_width=True, key="btn_back_page3"):
            st.info("👈 Use o menu lateral")
    
    st.markdown("---")
    
    calculate_portfolio_inputs()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Stats",
        "🎯 Fronteira",
        "🎯 Alvo",
        "💼 Salvos"
    ])
    
    with tab1:
        show_input_statistics()
        st.markdown("---")
        show_equal_weight()
    
    with tab2:
        compute_efficient_frontier()
        st.markdown("---")
        show_efficient_frontier_plot()
    
    with tab3:
        optimize_target_portfolio()
    
    with tab4:
        show_saved_portfolios()
    
    # Próximos
    st.markdown("---")
    st.info("""
    **Continue:** Menu lateral (☰) →
    - 🎯 Sharpe e MinVol
    - 📋 Resumo Executivo
    """)


if __name__ == "__main__":
    main()
