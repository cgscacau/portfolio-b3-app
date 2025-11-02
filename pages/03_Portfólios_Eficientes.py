"""
Página de Portfólios Eficientes
Otimização de carteiras usando Teoria Moderna de Portfólio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data
from core.init import init_all
from core.cache import salvar_dados_cache, carregar_dados_cache

# Configuração da página
st.set_page_config(
    page_title="Portfólios Eficientes",
    page_icon="📈",
    layout="wide"
)

# INICIALIZAR SESSION STATE
init_all()


# ==========================================
# FUNÇÕES DE CÁLCULO
# ==========================================

def calcular_retornos(price_data):
    """
    Calcula retornos diários dos ativos
    
    Args:
        price_data: DataFrame com preços históricos
        
    Returns:
        DataFrame com retornos diários
    """
    returns = price_data.pct_change().dropna()
    return returns


def calcular_metricas_portfolio(weights, returns, risk_free_rate=0.0):
    """
    Calcula métricas de um portfólio
    
    Args:
        weights: Array com pesos dos ativos
        returns: DataFrame com retornos
        risk_free_rate: Taxa livre de risco anual
        
    Returns:
        Dict com métricas (retorno, volatilidade, sharpe)
    """
    # Retorno esperado anualizado
    portfolio_return = np.sum(returns.mean() * weights) * 252
    
    # Volatilidade anualizada
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_std,
        'sharpe': sharpe_ratio
    }


def gerar_portfolios_aleatorios(returns, num_portfolios=5000, risk_free_rate=0.0):
    """
    Gera portfólios aleatórios para a fronteira eficiente
    
    Args:
        returns: DataFrame com retornos
        num_portfolios: Número de portfólios a gerar
        risk_free_rate: Taxa livre de risco
        
    Returns:
        DataFrame com portfólios gerados
    """
    num_assets = len(returns.columns)
    results = []
    
    for _ in range(num_portfolios):
        # Gerar pesos aleatórios
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalizar para somar 1
        
        # Calcular métricas
        metrics = calcular_metricas_portfolio(weights, returns, risk_free_rate)
        
        # Armazenar
        result = {
            'return': metrics['return'],
            'volatility': metrics['volatility'],
            'sharpe': metrics['sharpe']
        }
        
        # Adicionar pesos individuais
        for i, ticker in enumerate(returns.columns):
            result[ticker] = weights[i]
        
        results.append(result)
    
    return pd.DataFrame(results)


def encontrar_portfolio_sharpe_maximo(returns, risk_free_rate=0.0):
    """
    Encontra o portfólio com maior Sharpe Ratio
    
    Args:
        returns: DataFrame com retornos
        risk_free_rate: Taxa livre de risco
        
    Returns:
        Dict com pesos e métricas do portfólio ótimo
    """
    from scipy.optimize import minimize
    
    num_assets = len(returns.columns)
    
    def neg_sharpe(weights):
        metrics = calcular_metricas_portfolio(weights, returns, risk_free_rate)
        return -metrics['sharpe']
    
    # Restrições: soma dos pesos = 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: cada peso entre 0 e 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Chute inicial: igual peso
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Otimizar
    result = minimize(
        neg_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimal_weights = result.x
        metrics = calcular_metricas_portfolio(optimal_weights, returns, risk_free_rate)
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'return': metrics['return'],
            'volatility': metrics['volatility'],
            'sharpe': metrics['sharpe']
        }
    
    return None


def encontrar_portfolio_minima_volatilidade(returns):
    """
    Encontra o portfólio de mínima volatilidade
    
    Args:
        returns: DataFrame com retornos
        
    Returns:
        Dict com pesos e métricas do portfólio
    """
    from scipy.optimize import minimize
    
    num_assets = len(returns.columns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Restrições
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Otimizar
    result = minimize(
        portfolio_volatility,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        optimal_weights = result.x
        metrics = calcular_metricas_portfolio(optimal_weights, returns, 0.0)
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'return': metrics['return'],
            'volatility': metrics['volatility'],
            'sharpe': metrics['sharpe']
        }
    
    return None


# ==========================================
# FUNÇÕES DE VISUALIZAÇÃO
# ==========================================

def plotar_fronteira_eficiente(portfolios_df, max_sharpe=None, min_vol=None):
    """
    Plota a fronteira eficiente
    
    Args:
        portfolios_df: DataFrame com portfólios simulados
        max_sharpe: Dict com portfólio de máximo Sharpe
        min_vol: Dict com portfólio de mínima volatilidade
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Portfólios simulados
    fig.add_trace(go.Scatter(
        x=portfolios_df['volatility'],
        y=portfolios_df['return'],
        mode='markers',
        marker=dict(
            size=5,
            color=portfolios_df['sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        text=[f"Sharpe: {s:.2f}" for s in portfolios_df['sharpe']],
        name='Portfólios Simulados'
    ))
    
    # Portfólio de máximo Sharpe
    if max_sharpe:
        fig.add_trace(go.Scatter(
            x=[max_sharpe['volatility']],
            y=[max_sharpe['return']],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Máximo Sharpe'
        ))
    
    # Portfólio de mínima volatilidade
    if min_vol:
        fig.add_trace(go.Scatter(
            x=[min_vol['volatility']],
            y=[min_vol['return']],
            mode='markers',
            marker=dict(color='green', size=15, symbol='diamond'),
            name='Mínima Volatilidade'
        ))
    
    fig.update_layout(
        title='Fronteira Eficiente',
        xaxis_title='Volatilidade (Risco)',
        yaxis_title='Retorno Esperado',
        hovermode='closest',
        height=600
    )
    
    return fig


def exibir_composicao_portfolio(weights_dict, title):
    """
    Exibe a composição de um portfólio
    
    Args:
        weights_dict: Dicionário com pesos {ticker: peso}
        title: Título do gráfico
    """
    # Filtrar pesos significativos (> 1%)
    weights_filtrados = {k: v for k, v in weights_dict.items() if v > 0.01}
    
    # Ordenar por peso
    weights_ordenados = dict(sorted(weights_filtrados.items(), key=lambda x: x[1], reverse=True))
    
    # Criar gráfico de pizza
    fig = go.Figure(data=[go.Pie(
        labels=list(weights_ordenados.keys()),
        values=list(weights_ordenados.values()),
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{value:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela com pesos
    df_weights = pd.DataFrame({
        'Ativo': list(weights_ordenados.keys()),
        'Peso': list(weights_ordenados.values())
    })
    
    st.dataframe(
        df_weights.style.format({'Peso': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# FUNÇÕES PRINCIPAIS
# ==========================================

def carregar_dados():
    """Carrega dados de preços usando cache global"""
    
    tickers = st.session_state.portfolio_tickers
    
    if not tickers:
        st.warning("⚠ Nenhum ativo no portfólio")
        return False
    
    if len(tickers) < 2:
        st.warning("⚠ Selecione pelo menos 2 ativos")
        return False
    
    start_date = st.session_state.period_start
    end_date = st.session_state.period_end
    
    # USAR CACHE
    price_data, _ = carregar_dados_cache(tickers, start_date, end_date)
    
    if price_data is not None and not price_data.empty:
        st.info("📦 Dados carregados do cache")
        st.session_state.price_data = price_data
        st.success(f"✓ {len(price_data)} dias, {len(price_data.columns)} ativos")
        return True
    
    # Se não tem cache, baixar
    st.info(f"📥 Baixando dados de {len(tickers)} ativos...")
    
    with st.spinner("Baixando preços históricos..."):
        try:
            price_data = data.get_price_history(tickers, start_date, end_date)
            
            if price_data.empty:
                st.error("❌ Nenhum dado obtido")
                return False
            
            # Limpar dados
            price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.8)
            
            if price_data.empty:
                st.error("❌ Dados insuficientes após limpeza")
                return False
            
            # SALVAR NO CACHE
            salvar_dados_cache(tickers, start_date, end_date, price_data, None)
            
            st.session_state.price_data = price_data
            st.success(f"✓ Dados carregados: {len(price_data)} dias, {len(price_data.columns)} ativos")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
            return False


def otimizar_portfolios():
    """Executa otimização de portfólios"""
    
    if st.session_state.price_data is None:
        st.warning("⚠ Carregue os dados primeiro")
        return
    
    price_data = st.session_state.price_data
    risk_free_rate = st.session_state.risk_free_rate
    
    with st.spinner("Calculando retornos..."):
        returns = calcular_retornos(price_data)
    
    with st.spinner("Gerando fronteira eficiente..."):
        portfolios_df = gerar_portfolios_aleatorios(returns, 5000, risk_free_rate)
        st.session_state.efficient_frontier = portfolios_df
    
    with st.spinner("Encontrando portfólio de máximo Sharpe..."):
        max_sharpe = encontrar_portfolio_sharpe_maximo(returns, risk_free_rate)
    
    with st.spinner("Encontrando portfólio de mínima volatilidade..."):
        min_vol = encontrar_portfolio_minima_volatilidade(returns)
    
    st.session_state.optimal_portfolios = {
        'max_sharpe': max_sharpe,
        'min_vol': min_vol
    }
    
    st.success("✓ Otimização concluída!")


def exibir_resultados():
    """Exibe resultados da otimização"""
    
    if st.session_state.efficient_frontier is None:
        st.info("Execute a otimização primeiro")
        return
    
    portfolios_df = st.session_state.efficient_frontier
    optimal = st.session_state.optimal_portfolios
    
    st.header("📊 Resultados da Otimização")
    
    # Gráfico da fronteira eficiente
    st.subheader("Fronteira Eficiente")
    fig = plotar_fronteira_eficiente(
        portfolios_df,
        optimal['max_sharpe'],
        optimal['min_vol']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Portfólios ótimos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Portfólio de Máximo Sharpe")
        
        if optimal['max_sharpe']:
            max_sharpe = optimal['max_sharpe']
            
            # Métricas
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Retorno", f"{max_sharpe['return']:.2%}")
            with col_b:
                st.metric("Volatilidade", f"{max_sharpe['volatility']:.2%}")
            with col_c:
                st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.2f}")
            
            # Composição
            exibir_composicao_portfolio(
                max_sharpe['weights'],
                "Composição - Máximo Sharpe"
            )
    
    with col2:
        st.subheader("🛡️ Portfólio de Mínima Volatilidade")
        
        if optimal['min_vol']:
            min_vol = optimal['min_vol']
            
            # Métricas
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Retorno", f"{min_vol['return']:.2%}")
            with col_b:
                st.metric("Volatilidade", f"{min_vol['volatility']:.2%}")
            with col_c:
                st.metric("Sharpe Ratio", f"{min_vol['sharpe']:.2f}")
            
            # Composição
            exibir_composicao_portfolio(
                min_vol['weights'],
                "Composição - Mínima Volatilidade"
            )


# ==========================================
# FUNÇÃO PRINCIPAL
# ==========================================

def main():
    """Função principal"""
    
    st.title("📈 Portfólios Eficientes")
    st.markdown("Otimização de carteiras usando Teoria Moderna de Portfólio (Markowitz)")
    st.markdown("---")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Período
        st.subheader("Período de Análise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start = st.date_input(
                "Início",
                value=st.session_state.period_start
            )
        
        with col2:
            end = st.date_input(
                "Fim",
                value=st.session_state.period_end
            )
        
        st.session_state.period_start = datetime.combine(start, datetime.min.time())
        st.session_state.period_end = datetime.combine(end, datetime.min.time())
        
        st.markdown("---")
        
        # Taxa livre de risco
        st.subheader("Parâmetros")
        
        risk_free = st.number_input(
            "Taxa Livre de Risco (anual)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_free_rate,
            step=0.01,
            format="%.4f",
            help="Taxa CDI ou Selic anualizada"
        )
        st.session_state.risk_free_rate = risk_free
        
        st.markdown("---")
        
        # Botões de ação
        if st.button("📥 Carregar Dados", type="primary", use_container_width=True):
            carregar_dados()
        
        if st.button("🔄 Otimizar", use_container_width=True):
            if st.session_state.price_data is not None:
                otimizar_portfolios()
            else:
                st.warning("Carregue os dados primeiro")
    
    # Conteúdo principal
    if not st.session_state.portfolio_tickers:
        st.warning("⚠ Nenhum ativo no portfólio. Vá para 'Selecionar Ativos' primeiro.")
        st.stop()
    
    # Info sobre ativos
    st.info(f"📊 {len(st.session_state.portfolio_tickers)} ativos no portfólio")
    
    with st.expander("Ver ativos selecionados"):
        st.write(st.session_state.portfolio_tickers)
    
    st.markdown("---")
    
    # Mostrar dados carregados
    if st.session_state.price_data is not None:
        price_data = st.session_state.price_data
        
        st.subheader("📈 Dados Carregados")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ativos", len(price_data.columns))
        with col2:
            st.metric("Dias", len(price_data))
        with col3:
            st.metric("Período", f"{price_data.index[0].date()} a {price_data.index[-1].date()}")
        
        with st.expander("Ver dados"):
            st.dataframe(price_data.tail(10), use_container_width=True)
        
        st.markdown("---")
    
    # Mostrar resultados
    exibir_resultados()
    
    # Informações
    with st.expander("ℹ️ Sobre a Otimização"):
        st.markdown("""
        **Teoria Moderna de Portfólio (Markowitz)**
        
        A otimização de portfólios busca encontrar a melhor combinação de ativos que:
        - Maximiza o retorno para um dado nível de risco, ou
        - Minimiza o risco para um dado nível de retorno
        
        **Portfólio de Máximo Sharpe:**
        - Melhor relação risco-retorno
        - Ideal para investidores que buscam eficiência
        
        **Portfólio de Mínima Volatilidade:**
        - Menor risco possível
        - Ideal para investidores conservadores
        
        **Fronteira Eficiente:**
        - Conjunto de portfólios ótimos
        - Cada ponto representa uma alocação diferente
        - Cor indica o Sharpe Ratio (quanto maior, melhor)
        """)


if __name__ == "__main__":
    main()
