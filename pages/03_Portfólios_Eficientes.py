"""
Página 3: Portfólios Eficientes
Otimização de Markowitz e fronteira eficiente
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

from core import data, metrics, opt, filters, ui
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
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'price_data' not in st.session_state:
        st.session_state.price_data = pd.DataFrame()
    
    if 'expected_returns' not in st.session_state:
        st.session_state.expected_returns = pd.Series()
    
    if 'cov_matrix' not in st.session_state:
        st.session_state.cov_matrix = pd.DataFrame()
    
    if 'efficient_frontier' not in st.session_state:
        st.session_state.efficient_frontier = pd.DataFrame()
    
    if 'optimized_portfolios' not in st.session_state:
        st.session_state.optimized_portfolios = {}


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
    
    if st.session_state.price_data.empty:
        ui.create_info_box(
            "⚠️ Dados de preços não carregados. Por favor, carregue os dados na página 'Análise de Dividendos'.",
            "warning"
        )
        
        if st.button("💸 Ir para Análise de Dividendos", type="primary"):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")
        
        return False
    
    return True


def calculate_portfolio_inputs():
    """Calcula retornos esperados e matriz de covariância."""
    
    ui.create_section_header(
        "🧮 Cálculo de Parâmetros",
        "Preparando dados para otimização",
        "🧮"
    )
    
    if st.session_state.price_data.empty:
        st.error("❌ Dados de preços não disponíveis")
        return False
    
    # Informações do período
    prices_df = st.session_state.price_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Ativos:** {len(prices_df.columns)}")
    
    with col2:
        st.info(f"📅 **Período:** {len(prices_df)} dias")
    
    with col3:
        years = len(prices_df) / 252
        st.info(f"⏱️ **Duração:** {years:.1f} anos")
    
    if st.button("🔄 Calcular/Atualizar Parâmetros", type="primary", use_container_width=True):
        
        with st.spinner("Calculando retornos esperados e covariância..."):
            
            # Criar objeto de métricas
            perf_metrics = metrics.PerformanceMetrics(
                prices_df,
                risk_free_rate=st.session_state.risk_free_rate
            )
            
            # Retornos esperados (anualizados)
            expected_returns = pd.Series(
                {ticker: perf_metrics.calculate_annualized_return(ticker) 
                 for ticker in prices_df.columns}
            )
            
            # Remover NaN
            expected_returns = expected_returns.dropna()
            
            # Matriz de covariância (anualizada)
            cov_matrix = perf_metrics.get_covariance_matrix(annualized=True)
            
            # Alinhar
            common_tickers = expected_returns.index.intersection(cov_matrix.index)
            expected_returns = expected_returns[common_tickers]
            cov_matrix = cov_matrix.loc[common_tickers, common_tickers]
            
            # Salvar
            st.session_state.expected_returns = expected_returns
            st.session_state.cov_matrix = cov_matrix
            
            st.success("✅ Parâmetros calculados com sucesso!")
            
            # Estatísticas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ui.create_metric_card(
                    "Retorno Médio",
                    f"{expected_returns.mean()*100:.2f}%",
                    icon="📈"
                )
            
            with col2:
                ui.create_metric_card(
                    "Retorno Máximo",
                    f"{expected_returns.max()*100:.2f}%",
                    icon="🔝"
                )
            
            with col3:
                ui.create_metric_card(
                    "Retorno Mínimo",
                    f"{expected_returns.min()*100:.2f}%",
                    icon="📉"
                )
            
            with col4:
                avg_corr = cov_matrix.values[np.triu_indices_from(cov_matrix.values, k=1)].mean()
                # Converter covariância para correlação
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                
                ui.create_metric_card(
                    "Correlação Média",
                    f"{avg_corr:.3f}",
                    icon="🔗"
                )
            
            st.rerun()
    
    return True


def show_input_statistics():
    """Exibe estatísticas dos dados de entrada."""
    
    if st.session_state.expected_returns.empty or st.session_state.cov_matrix.empty:
        ui.create_info_box(
            "Calcule os parâmetros usando o botão acima para visualizar as estatísticas.",
            "info"
        )
        return
    
    ui.create_section_header(
        "📊 Estatísticas dos Dados",
        "Análise dos retornos esperados e correlações",
        "📊"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Retornos Esperados")
        
        # Criar DataFrame
        returns_df = pd.DataFrame({
            'Ticker': st.session_state.expected_returns.index,
            'Retorno Anualizado': st.session_state.expected_returns.values * 100
        })
        
        returns_df = returns_df.sort_values('Retorno Anualizado', ascending=False)
        returns_df['Retorno Anualizado'] = returns_df['Retorno Anualizado'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(returns_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 🔗 Matriz de Correlação")
        
        # Converter covariância para correlação
        std_devs = np.sqrt(np.diag(st.session_state.cov_matrix))
        corr_matrix = st.session_state.cov_matrix / np.outer(std_devs, std_devs)
        
        # Heatmap
        fig = ui.plot_correlation_heatmap(corr_matrix, "Correlação entre Ativos")
        st.plotly_chart(fig, use_container_width=True)


def compute_efficient_frontier():
    """Computa a fronteira eficiente."""
    
    if st.session_state.expected_returns.empty or st.session_state.cov_matrix.empty:
        return
    
    ui.create_section_header(
        "🎯 Fronteira Eficiente",
        "Calculando portfólios ótimos",
        "🎯"
    )
    
    # Parâmetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_points = st.slider(
            "Número de pontos na fronteira:",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="Mais pontos = maior precisão, mas mais lento"
        )
    
    with col2:
        max_weight = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=100,
            value=int(st.session_state.max_weight_per_asset * 100),
            step=5
        ) / 100
    
    with col3:
        min_weight = st.slider(
            "Peso mínimo por ativo (%):",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        ) / 100
    
    # Restrições setoriais
    apply_sector_constraints = st.checkbox(
        "Aplicar restrições setoriais",
        value=True,
        help="Limita concentração por setor"
    )
    
    sector_constraints = None
    if apply_sector_constraints and not st.session_state.universe_df.empty:
        max_sector_weight = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=int(st.session_state.max_weight_per_sector * 100),
            step=5
        ) / 100
        
        sector_constraints = opt.create_sector_constraints(
            st.session_state.universe_df,
            st.session_state.expected_returns.index.tolist(),
            max_sector_weight
        )
    
    if st.button("🚀 Calcular Fronteira Eficiente", type="primary", use_container_width=True):
        
        with st.spinner("Calculando fronteira eficiente... Isso pode levar alguns minutos."):
            
            try:
                # Criar otimizador
                optimizer = opt.MarkowitzOptimizer(
                    st.session_state.expected_returns,
                    st.session_state.cov_matrix,
                    st.session_state.risk_free_rate
                )
                
                # Computar fronteira
                frontier_df = optimizer.compute_efficient_frontier(
                    n_points=n_points,
                    max_weight=max_weight,
                    min_weight=min_weight
                )
                
                if frontier_df.empty:
                    st.error("❌ Erro ao calcular fronteira eficiente")
                    return
                
                st.session_state.efficient_frontier = frontier_df
                
                st.success(f"✅ Fronteira calculada com {len(frontier_df)} pontos!")
                
                # Estatísticas da fronteira
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    min_vol_idx = frontier_df['volatility'].idxmin()
                    min_vol = frontier_df.loc[min_vol_idx, 'volatility']
                    
                    ui.create_metric_card(
                        "Mínima Volatilidade",
                        f"{min_vol*100:.2f}%",
                        icon="🛡️"
                    )
                
                with col2:
                    max_ret_idx = frontier_df['return'].idxmax()
                    max_ret = frontier_df.loc[max_ret_idx, 'return']
                    
                    ui.create_metric_card(
                        "Máximo Retorno",
                        f"{max_ret*100:.2f}%",
                        icon="📈"
                    )
                
                with col3:
                    max_sharpe_idx = frontier_df['sharpe'].idxmax()
                    max_sharpe = frontier_df.loc[max_sharpe_idx, 'sharpe']
                    
                    ui.create_metric_card(
                        "Máximo Sharpe",
                        f"{max_sharpe:.3f}",
                        icon="⭐"
                    )
                
                with col4:
                    # Retorno do ponto de máximo Sharpe
                    sharpe_ret = frontier_df.loc[max_sharpe_idx, 'return']
                    
                    ui.create_metric_card(
                        "Retorno (Max Sharpe)",
                        f"{sharpe_ret*100:.2f}%",
                        icon="🎯"
                    )
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Erro ao calcular fronteira: {e}")
                st.error(f"❌ Erro ao calcular fronteira: {e}")


def show_efficient_frontier_plot():
    """Exibe gráfico da fronteira eficiente."""
    
    if st.session_state.efficient_frontier.empty:
        ui.create_info_box(
            "Calcule a fronteira eficiente usando o botão acima para visualizar o gráfico.",
            "info"
        )
        return
    
    ui.create_section_header(
        "📊 Visualização da Fronteira",
        "Gráfico interativo risco vs retorno",
        "📊"
    )
    
    frontier_df = st.session_state.efficient_frontier
    
    # Identificar portfólios especiais
    max_sharpe_idx = frontier_df['sharpe'].idxmax()
    min_vol_idx = frontier_df['volatility'].idxmin()
    
    highlighted = {
        'Máximo Sharpe': (
            frontier_df.loc[max_sharpe_idx, 'return'],
            frontier_df.loc[max_sharpe_idx, 'volatility']
        ),
        'Mínima Volatilidade': (
            frontier_df.loc[min_vol_idx, 'return'],
            frontier_df.loc[min_vol_idx, 'volatility']
        )
    }
    
    # Plotar
    fig = ui.plot_efficient_frontier(
        frontier_df,
        highlighted_portfolios=highlighted,
        title="Fronteira Eficiente de Markowitz"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explicação
    with st.expander("ℹ️ Como interpretar a fronteira eficiente?", expanded=False):
        st.markdown("""
        A **Fronteira Eficiente** representa todos os portfólios que oferecem o **máximo retorno esperado** 
        para cada nível de risco (volatilidade).
        
        **Pontos-chave:**
        
        - **Máximo Sharpe** ⭐: Melhor relação risco-retorno ajustada pela taxa livre de risco
        - **Mínima Volatilidade** 🛡️: Portfólio com menor risco possível
        - **Cores**: Indicam o Índice de Sharpe (quanto mais claro, melhor)
        
        **Interpretação:**
        - Portfólios **acima** da fronteira são impossíveis
        - Portfólios **abaixo** são ineficientes (existe alternativa melhor)
        - Portfólios **na fronteira** são ótimos para seu nível de risco
        
        **Escolha seu portfólio:**
        - **Conservador**: Próximo à Mínima Volatilidade
        - **Balanceado**: Próximo ao Máximo Sharpe
        - **Agressivo**: Maior retorno (aceita mais risco)
        """)


def optimize_target_portfolio():
    """Otimiza portfólio para alvo específico."""
    
    if st.session_state.expected_returns.empty or st.session_state.cov_matrix.empty:
        return
    
    ui.create_section_header(
        "🎯 Portfólio Alvo",
        "Otimize para retorno ou risco específico",
        "🎯"
    )
    
    # Escolher tipo de otimização
    opt_type = st.radio(
        "Tipo de otimização:",
        ["Retorno Alvo", "Risco Alvo"],
        horizontal=True,
        help="Escolha se quer fixar o retorno ou o risco"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if opt_type == "Retorno Alvo":
            target_return = st.slider(
                "Retorno anualizado alvo (%):",
                min_value=float(st.session_state.expected_returns.min() * 100),
                max_value=float(st.session_state.expected_returns.max() * 100),
                value=float(st.session_state.expected_returns.mean() * 100),
                step=0.5
            ) / 100
        else:
            # Estimar range de volatilidade
            min_vol = st.session_state.cov_matrix.values.diagonal().min() ** 0.5
            max_vol = st.session_state.cov_matrix.values.diagonal().max() ** 0.5
            
            target_vol = st.slider(
                "Volatilidade anualizada alvo (%):",
                min_value=float(min_vol * 100),
                max_value=float(max_vol * 100),
                value=float((min_vol + max_vol) / 2 * 100),
                step=0.5
            ) / 100
    
    with col2:
        max_weight_target = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=100,
            value=15,
            step=5,
            key="target_max_weight"
        ) / 100
    
    if st.button("🎯 Otimizar Portfólio Alvo", type="primary", use_container_width=True):
        
        with st.spinner("Otimizando portfólio..."):
            
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
                st.session_state.optimized_portfolios['Portfólio Alvo'] = {
                    'weights': weights,
                    'stats': stats
                }
                
                st.success("✅ Portfólio otimizado com sucesso!")
                
                # Exibir resultados
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
                
                # Alocação
                st.markdown("### 📊 Alocação do Portfólio")
                
                fig = ui.plot_portfolio_weights(weights, "Alocação - Portfólio Alvo")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela de pesos
                weights_df = pd.DataFrame({
                    'Ticker': list(weights.keys()),
                    'Peso (%)': [w * 100 for w in weights.values()]
                })
                weights_df = weights_df.sort_values('Peso (%)', ascending=False)
                
                st.dataframe(weights_df, use_container_width=True)
            
            except Exception as e:
                logger.error(f"Erro na otimização: {e}")
                st.error(f"❌ Erro na otimização: {e}")


def show_saved_portfolios():
    """Exibe portfólios salvos."""
    
    if not st.session_state.optimized_portfolios:
        ui.create_info_box(
            "Nenhum portfólio otimizado ainda. Use as ferramentas acima para criar portfólios.",
            "info"
        )
        return
    
    ui.create_section_header(
        "💼 Portfólios Salvos",
        "Comparação dos portfólios otimizados",
        "💼"
    )
    
    # Criar DataFrame de comparação
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
    
    # Formatar para exibição
    display_df = comparison_df.copy()
    display_df['Retorno'] = display_df['Retorno'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Volatilidade'] = display_df['Volatilidade'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Gráfico de comparação
    if len(comparison_df) > 1:
        st.markdown("### 📊 Comparação Visual")
        
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
                             'Retorno: %{y:.2f}%<br>' +
                             'Volatilidade: %{x:.2f}%<br>' +
                             f"Sharpe: {row['Sharpe']:.3f}<br>" +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Comparação de Portfólios",
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
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes de cada portfólio
    st.markdown("### 📋 Detalhes dos Portfólios")
    
    selected_portfolio = st.selectbox(
        "Selecione um portfólio para ver detalhes:",
        options=list(st.session_state.optimized_portfolios.keys())
    )
    
    if selected_portfolio:
        portfolio = st.session_state.optimized_portfolios[selected_portfolio]
        weights = portfolio['weights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Alocação")
            fig = ui.plot_portfolio_weights(weights, f"Alocação - {selected_portfolio}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Pesos Detalhados")
            
            weights_df = pd.DataFrame({
                'Ticker': list(weights.keys()),
                'Peso (%)': [w * 100 for w in weights.values()]
            })
            weights_df = weights_df.sort_values('Peso (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True, height=400)
        
        # Download
        ui.create_download_button(
            weights_df,
            f"{selected_portfolio.replace(' ', '_')}_weights.csv",
            "📥 Download Alocação",
            "csv"
        )


def show_equal_weight_baseline():
    """Cria portfólio equally weighted como baseline."""
    
    if st.session_state.expected_returns.empty:
        return
    
    ui.create_section_header(
        "⚖️ Portfólio Equally Weighted (Baseline)",
        "Comparação com alocação uniforme",
        "⚖️"
    )
    
    st.markdown("""
    O portfólio **Equally Weighted** aloca peso igual para todos os ativos, 
    servindo como **baseline** para comparação com portfólios otimizados.
    """)
    
    if st.button("⚖️ Criar Portfólio Equally Weighted", use_container_width=True):
        
        tickers = st.session_state.expected_returns.index.tolist()
        
        ew_optimizer = opt.EqualWeightOptimizer(tickers)
        weights = ew_optimizer.optimize()
        
        # Calcular estatísticas
        stats = opt.calculate_portfolio_stats(
            weights,
            st.session_state.expected_returns,
            st.session_state.cov_matrix,
            st.session_state.risk_free_rate
        )
        
        # Salvar
        st.session_state.optimized_portfolios['Equally Weighted'] = {
            'weights': weights,
            'stats': stats
        }
        
        st.success("✅ Portfólio Equally Weighted criado!")
        
        # Exibir métricas
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
                "Peso por Ativo",
                f"{100/len(tickers):.2f}%",
                icon="⚖️"
            )
        
        st.rerun()


def main():
    """Função principal da página."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<p class="gradient-title">📊 Portfólios Eficientes</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Otimização de portfólios usando a **Teoria Moderna de Portfólio** (Markowitz). 
    Encontre a melhor combinação de ativos para seu perfil de risco-retorno.
    """)
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        st.stop()
    
    # Informações
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"📊 **{len(st.session_state.selected_tickers)} ativos** prontos para otimização")
    
    with col2:
        if st.button("🔙 Voltar", use_container_width=True):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")
    
    st.markdown("---")
    
    # Calcular parâmetros
    calculate_portfolio_inputs()
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estatísticas",
        "🎯 Fronteira Eficiente",
        "🎯 Portfólio Alvo",
        "💼 Portfólios Salvos"
    ])
    
    with tab1:
        show_input_statistics()
        st.markdown("---")
        show_equal_weight_baseline()
    
    with tab2:
        compute_efficient_frontier()
        st.markdown("---")
        show_efficient_frontier_plot()
    
    with tab3:
        optimize_target_portfolio()
    
    with tab4:
        show_saved_portfolios()
    
    # Próximos passos
    st.markdown("---")
    
    ui.create_section_header(
        "🚀 Próximos Passos",
        "Continue para otimizações específicas",
        "🚀"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Sharpe e MinVol", use_container_width=True, type="primary"):
            st.switch_page("app/pages/04_Sharpe_e_MinVol.py")
    
    with col2:
        if st.button("📋 Resumo Executivo", use_container_width=True):
            st.switch_page("app/pages/05_Resumo_Executivo.py")
    
    with col3:
        if st.button("🔙 Voltar para Dividendos", use_container_width=True):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")


if __name__ == "__main__":
    main()
