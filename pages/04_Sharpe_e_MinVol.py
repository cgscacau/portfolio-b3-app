"""
Análise Comparativa: Portfólio de Máximo Sharpe vs Mínima Volatilidade
Comparação detalhada entre estratégias de otimização
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data
from core.init import init_all

# Configuração da página
st.set_page_config(
    page_title="Sharpe vs MinVol",
    page_icon="⚖️",
    layout="wide"
)

# Inicializar session state
init_all()


# ==========================================
# FUNÇÕES AUXILIARES DE CÁLCULO
# ==========================================

def calcular_retornos_diarios(prices):
    """
    Calcula retornos diários percentuais
    
    Args:
        prices: DataFrame com preços
        
    Returns:
        DataFrame com retornos
    """
    return prices.pct_change().dropna()


def calcular_retorno_anual(weights, returns):
    """
    Calcula retorno anual esperado do portfólio
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos
        
    Returns:
        Float com retorno anualizado
    """
    return np.sum(returns.mean() * weights) * 252


def calcular_volatilidade_anual(weights, returns):
    """
    Calcula volatilidade anual do portfólio
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos
        
    Returns:
        Float com volatilidade anualizada
    """
    cov_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)


def calcular_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calcula Sharpe Ratio do portfólio
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos
        risk_free_rate: Taxa livre de risco anual
        
    Returns:
        Float com Sharpe Ratio
    """
    ret = calcular_retorno_anual(weights, returns)
    vol = calcular_volatilidade_anual(weights, returns)
    return (ret - risk_free_rate) / vol


# ==========================================
# FUNÇÕES DE OTIMIZAÇÃO
# ==========================================

def otimizar_sharpe_maximo(returns, risk_free_rate):
    """
    Encontra o portfólio com máximo Sharpe Ratio
    
    Args:
        returns: DataFrame com retornos
        risk_free_rate: Taxa livre de risco
        
    Returns:
        Tuple (pesos, métricas)
    """
    num_assets = len(returns.columns)
    
    # Função objetivo: maximizar Sharpe = minimizar -Sharpe
    def objective(weights):
        return -calcular_sharpe_ratio(weights, returns, risk_free_rate)
    
    # Restrições: soma dos pesos = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Limites: cada peso entre 0 e 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Chute inicial: pesos iguais
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Otimizar
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return None, None
    
    weights = result.x
    
    # Calcular métricas
    metrics = {
        'return': calcular_retorno_anual(weights, returns),
        'volatility': calcular_volatilidade_anual(weights, returns),
        'sharpe': calcular_sharpe_ratio(weights, returns, risk_free_rate)
    }
    
    # Criar dicionário de pesos
    weights_dict = dict(zip(returns.columns, weights))
    
    return weights_dict, metrics


def otimizar_minima_volatilidade(returns):
    """
    Encontra o portfólio com mínima volatilidade
    
    Args:
        returns: DataFrame com retornos
        
    Returns:
        Tuple (pesos, métricas)
    """
    num_assets = len(returns.columns)
    
    # Função objetivo: minimizar volatilidade
    def objective(weights):
        return calcular_volatilidade_anual(weights, returns)
    
    # Restrições
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Otimizar
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return None, None
    
    weights = result.x
    
    # Calcular métricas
    metrics = {
        'return': calcular_retorno_anual(weights, returns),
        'volatility': calcular_volatilidade_anual(weights, returns),
        'sharpe': calcular_sharpe_ratio(weights, returns, 0.0)
    }
    
    weights_dict = dict(zip(returns.columns, weights))
    
    return weights_dict, metrics


# ==========================================
# FUNÇÕES DE SIMULAÇÃO
# ==========================================

def simular_performance(weights_dict, prices, capital_inicial=10000):
    """
    Simula a performance histórica de um portfólio
    
    Args:
        weights_dict: Dicionário {ticker: peso}
        prices: DataFrame com preços
        capital_inicial: Capital inicial em R$
        
    Returns:
        Series com valores do portfólio ao longo do tempo
    """
    # Calcular retornos
    returns = calcular_retornos_diarios(prices)
    
    # Criar array de pesos na ordem correta
    weights = np.array([weights_dict.get(col, 0) for col in returns.columns])
    
    # Calcular retornos do portfólio
    portfolio_returns = returns.dot(weights)
    
    # Calcular valor acumulado
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = capital_inicial * cumulative_returns
    
    return portfolio_value


def calcular_drawdown(portfolio_value):
    """
    Calcula o drawdown (queda do pico) do portfólio
    
    Args:
        portfolio_value: Series com valores do portfólio
        
    Returns:
        Series com drawdown percentual
    """
    cumulative_max = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    return drawdown


# ==========================================
# FUNÇÕES DE VISUALIZAÇÃO
# ==========================================

def criar_grafico_pizza(weights_dict, title):
    """
    Cria gráfico de pizza com composição do portfólio
    
    Args:
        weights_dict: Dicionário {ticker: peso}
        title: Título do gráfico
        
    Returns:
        Figura Plotly
    """
    # Filtrar pesos > 1%
    weights_filtrado = {k: v for k, v in weights_dict.items() if v > 0.01}
    
    # Ordenar por peso
    weights_ordenado = dict(sorted(weights_filtrado.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights_ordenado.keys()),
        values=list(weights_ordenado.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='%{label}<br>%{value:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )
    
    return fig


def criar_grafico_barras_comparacao(weights_sharpe, weights_minvol):
    """
    Cria gráfico de barras comparando alocações
    
    Args:
        weights_sharpe: Pesos do portfólio Sharpe
        weights_minvol: Pesos do portfólio MinVol
        
    Returns:
        Figura Plotly
    """
    # Obter todos os tickers
    all_tickers = sorted(set(list(weights_sharpe.keys()) + list(weights_minvol.keys())))
    
    # Preparar dados
    sharpe_values = [weights_sharpe.get(t, 0) * 100 for t in all_tickers]
    minvol_values = [weights_minvol.get(t, 0) * 100 for t in all_tickers]
    
    # Filtrar apenas ativos com peso > 1% em algum portfólio
    filtered_data = [(t, s, m) for t, s, m in zip(all_tickers, sharpe_values, minvol_values) if s > 1 or m > 1]
    
    if filtered_data:
        tickers_filtered, sharpe_filtered, minvol_filtered = zip(*filtered_data)
    else:
        tickers_filtered, sharpe_filtered, minvol_filtered = [], [], []
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Máximo Sharpe',
        x=list(tickers_filtered),
        y=list(sharpe_filtered),
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        name='Mínima Volatilidade',
        x=list(tickers_filtered),
        y=list(minvol_filtered),
        marker_color='#2ecc71'
    ))
    
    fig.update_layout(
        title='Comparação de Alocação por Ativo',
        xaxis_title='Ativo',
        yaxis_title='Alocação (%)',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def criar_grafico_performance(value_sharpe, value_minvol):
    """
    Cria gráfico de performance histórica
    
    Args:
        value_sharpe: Series com valores do portfólio Sharpe
        value_minvol: Series com valores do portfólio MinVol
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=value_sharpe.index,
        y=value_sharpe.values,
        mode='lines',
        name='Máximo Sharpe',
        line=dict(color='#3498db', width=2),
        hovertemplate='%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=value_minvol.index,
        y=value_minvol.values,
        mode='lines',
        name='Mínima Volatilidade',
        line=dict(color='#2ecc71', width=2),
        hovertemplate='%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Performance Histórica dos Portfólios',
        xaxis_title='Data',
        yaxis_title='Valor do Portfólio (R$)',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def criar_grafico_drawdown(dd_sharpe, dd_minvol):
    """
    Cria gráfico de drawdown
    
    Args:
        dd_sharpe: Series com drawdown do Sharpe
        dd_minvol: Series com drawdown do MinVol
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dd_sharpe.index,
        y=dd_sharpe.values * 100,
        mode='lines',
        name='Máximo Sharpe',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=dd_minvol.index,
        y=dd_minvol.values * 100,
        mode='lines',
        name='Mínima Volatilidade',
        line=dict(color='#2ecc71', width=2),
        fill='tozeroy',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Drawdown dos Portfólios',
        xaxis_title='Data',
        yaxis_title='Drawdown (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal da página"""
    
    st.title("⚖️ Sharpe vs Mínima Volatilidade")
    st.markdown("Comparação detalhada entre duas estratégias de otimização de portfólio")
    st.markdown("---")
    
    # Verificar se há ativos selecionados
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo selecionado")
        st.info("👉 Vá para a página **Selecionar Ativos** para escolher os ativos do seu portfólio")
        st.stop()
    
    if len(st.session_state.portfolio_tickers) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos para otimização")
        st.stop()
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("📅 Período de Análise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_inicio = st.date_input(
                "Data Inicial",
                value=st.session_state.period_start,
                key="data_inicio_sharpe"
            )
        
        with col2:
            data_fim = st.date_input(
                "Data Final",
                value=st.session_state.period_end,
                key="data_fim_sharpe"
            )
        
        # Atualizar session state
        st.session_state.period_start = datetime.combine(data_inicio, datetime.min.time())
        st.session_state.period_end = datetime.combine(data_fim, datetime.min.time())
        
        st.markdown("---")
        
        st.subheader("💰 Parâmetros")
        
        taxa_livre_risco = st.number_input(
            "Taxa Livre de Risco (anual)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_free_rate,
            step=0.0001,
            format="%.4f",
            help="Taxa CDI ou Selic anualizada"
        )
        st.session_state.risk_free_rate = taxa_livre_risco
        
        capital_inicial = st.number_input(
            "Capital Inicial (R$)",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            format="%.2f"
        )
        
        st.markdown("---")
        
        # Botão de análise
        btn_analisar = st.button(
            "🔄 Analisar Portfólios",
            type="primary",
            use_container_width=True
        )
    
    # Mostrar informações dos ativos selecionados
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** selecionados para análise")
    
    with st.expander("📋 Ver lista de ativos"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Executar análise quando botão for pressionado
    if btn_analisar:
        
        # Carregar dados
        with st.spinner("📥 Carregando dados históricos..."):
            try:
                prices = data.get_price_history(
                    st.session_state.portfolio_tickers,
                    st.session_state.period_start,
                    st.session_state.period_end,
                    use_cache=False
                )
                
                if prices.empty:
                    st.error("❌ Não foi possível carregar os dados. Tente novamente.")
                    st.stop()
                
                # Limpar dados
                prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)
                prices = prices.fillna(method='ffill').fillna(method='bfill')
                
                if prices.empty or len(prices.columns) < 2:
                    st.error("❌ Dados insuficientes após limpeza")
                    st.stop()
                
                st.success(f"✅ Dados carregados: **{len(prices)} dias** de histórico para **{len(prices.columns)} ativos**")
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar dados: {str(e)}")
                st.stop()
        
        # Calcular retornos
        with st.spinner("📊 Calculando retornos..."):
            returns = calcular_retornos_diarios(prices)
        
        # Otimizar portfólios
        st.subheader("🎯 Otimização de Portfólios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Otimizando Máximo Sharpe..."):
                weights_sharpe, metrics_sharpe = otimizar_sharpe_maximo(returns, taxa_livre_risco)
                
                if weights_sharpe is None:
                    st.error("❌ Falha na otimização do Sharpe")
                else:
                    st.success("✅ Sharpe otimizado")
        
        with col2:
            with st.spinner("Otimizando Mínima Volatilidade..."):
                weights_minvol, metrics_minvol = otimizar_minima_volatilidade(returns)
                
                if weights_minvol is None:
                    st.error("❌ Falha na otimização MinVol")
                else:
                    st.success("✅ MinVol otimizado")
        
        if weights_sharpe is None or weights_minvol is None:
            st.stop()
        
        st.markdown("---")
        
        # Exibir métricas
        st.header("📈 Métricas dos Portfólios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Máximo Sharpe Ratio")
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(
                    "Retorno Anual",
                    f"{metrics_sharpe['return']:.2%}",
                    help="Retorno esperado anualizado"
                )
            with metric_cols[1]:
                st.metric(
                    "Volatilidade",
                    f"{metrics_sharpe['volatility']:.2%}",
                    help="Risco anualizado"
                )
            with metric_cols[2]:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics_sharpe['sharpe']:.3f}",
                    help="Retorno ajustado ao risco"
                )
        
        with col2:
            st.subheader("🛡️ Mínima Volatilidade")
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(
                    "Retorno Anual",
                    f"{metrics_minvol['return']:.2%}",
                    help="Retorno esperado anualizado"
                )
            with metric_cols[1]:
                st.metric(
                    "Volatilidade",
                    f"{metrics_minvol['volatility']:.2%}",
                    help="Risco anualizado"
                )
            with metric_cols[2]:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics_minvol['sharpe']:.3f}",
                    help="Retorno ajustado ao risco"
                )
        
        st.markdown("---")
        
        # Gráficos de composição
        st.header("🥧 Composição dos Portfólios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pizza_sharpe = criar_grafico_pizza(weights_sharpe, "Máximo Sharpe")
            st.plotly_chart(fig_pizza_sharpe, use_container_width=True)
        
        with col2:
            fig_pizza_minvol = criar_grafico_pizza(weights_minvol, "Mínima Volatilidade")
            st.plotly_chart(fig_pizza_minvol, use_container_width=True)
        
        # Gráfico de barras comparativo
        fig_barras = criar_grafico_barras_comparacao(weights_sharpe, weights_minvol)
        st.plotly_chart(fig_barras, use_container_width=True)
        
        st.markdown("---")
        
        # Simulação de performance
        st.header("📊 Performance Histórica")
        
        with st.spinner("Simulando performance..."):
            value_sharpe = simular_performance(weights_sharpe, prices, capital_inicial)
            value_minvol = simular_performance(weights_minvol, prices, capital_inicial)
        
        # Gráfico de performance
        fig_performance = criar_grafico_performance(value_sharpe, value_minvol)
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Métricas de performance
        col1, col2 = st.columns(2)
        
        with col1:
            valor_final_sharpe = value_sharpe.iloc[-1]
            retorno_total_sharpe = (valor_final_sharpe / capital_inicial - 1) * 100
            
            st.metric(
                "Valor Final - Máximo Sharpe",
                f"R$ {valor_final_sharpe:,.2f}",
                f"{retorno_total_sharpe:+.2f}%",
                delta_color="normal"
            )
        
        with col2:
            valor_final_minvol = value_minvol.iloc[-1]
            retorno_total_minvol = (valor_final_minvol / capital_inicial - 1) * 100
            
            st.metric(
                "Valor Final - Mínima Volatilidade",
                f"R$ {valor_final_minvol:,.2f}",
                f"{retorno_total_minvol:+.2f}%",
                delta_color="normal"
            )
        
        st.markdown("---")
        
        # Análise de drawdown
        st.header("📉 Análise de Drawdown")
        
        dd_sharpe = calcular_drawdown(value_sharpe)
        dd_minvol = calcular_drawdown(value_minvol)
        
        fig_drawdown = criar_grafico_drawdown(dd_sharpe, dd_minvol)
        st.plotly_chart(fig_drawdown, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_dd_sharpe = dd_sharpe.min() * 100
            st.metric(
                "Máximo Drawdown - Sharpe",
                f"{max_dd_sharpe:.2f}%",
                help="Maior queda do pico histórico"
            )
        
        with col2:
            max_dd_minvol = dd_minvol.min() * 100
            st.metric(
                "Máximo Drawdown - MinVol",
                f"{max_dd_minvol:.2f}%",
                help="Maior queda do pico histórico"
            )
        
        st.markdown("---")
        
        # Tabela comparativa final
        st.header("📋 Resumo Comparativo")
        
        df_comparacao = pd.DataFrame({
            'Métrica': [
                'Retorno Anual',
                'Volatilidade Anual',
                'Sharpe Ratio',
                'Valor Final',
                'Retorno Total',
                'Máximo Drawdown'
            ],
            'Máximo Sharpe': [
                f"{metrics_sharpe['return']:.2%}",
                f"{metrics_sharpe['volatility']:.2%}",
                f"{metrics_sharpe['sharpe']:.3f}",
                f"R$ {valor_final_sharpe:,.2f}",
                f"{retorno_total_sharpe:+.2f}%",
                f"{max_dd_sharpe:.2f}%"
            ],
            'Mínima Volatilidade': [
                f"{metrics_minvol['return']:.2%}",
                f"{metrics_minvol['volatility']:.2%}",
                f"{metrics_minvol['sharpe']:.3f}",
                f"R$ {valor_final_minvol:,.2f}",
                f"{retorno_total_minvol:+.2f}%",
                f"{max_dd_minvol:.2f}%"
            ]
        })
        
        st.dataframe(
            df_comparacao,
            use_container_width=True,
            hide_index=True
        )
    
    else:
        # Mensagem quando não há análise
        st.info("👈 Configure os parâmetros na barra lateral e clique em **Analisar Portfólios** para começar")
        
        # Informações sobre a página
        with st.expander("ℹ️ Sobre esta análise"):
            st.markdown("""
            ### Portfólio de Máximo Sharpe Ratio
            
            O **Sharpe Ratio** mede o retorno excedente por unidade de risco. Um portfólio com máximo Sharpe 
            oferece a melhor relação risco-retorno possível.
            
            **Ideal para:**
            - Investidores que buscam eficiência
            - Maximizar retorno ajustado ao risco
            - Perfil moderado a agressivo
            
            ### Portfólio de Mínima Volatilidade
            
            Este portfólio busca **minimizar o risco** (volatilidade), independente do retorno. 
            Resulta na carteira mais estável possível.
            
            **Ideal para:**
            - Investidores conservadores
            - Preservação de capital
            - Menor exposição a quedas
            
            ### Como interpretar
            
            - **Retorno Anual**: Ganho esperado em um ano
            - **Volatilidade**: Medida de risco (quanto maior, mais instável)
            - **Sharpe Ratio**: Quanto maior, melhor a relação risco-retorno
            - **Drawdown**: Maior queda do valor do pico até o vale
            """)


if __name__ == "__main__":
    main()
