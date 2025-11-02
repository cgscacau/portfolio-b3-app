"""
⚖️ Análise Comparativa: Sharpe Máximo vs Mínima Volatilidade
Comparação detalhada entre duas estratégias de otimização
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import sys
from pathlib import Path

# Configurar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core.init import init_all
from core.cache import salvar_dados_cache, carregar_dados_cache

# Configuração da página
st.set_page_config(
    page_title="Sharpe vs MinVol",
    page_icon="⚖️",
    layout="wide"
)

# Inicializar
init_all()


# ==========================================
# CARREGAMENTO DE DADOS COM CACHE
# ==========================================

def carregar_dados_com_cache(tickers, start_date, end_date):
    """
    Carrega dados usando cache global
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        
    Returns:
        DataFrame com preços ou None
    """
    # Tentar cache
    price_data, _ = carregar_dados_cache(tickers, start_date, end_date)
    
    if price_data is not None and not price_data.empty:
        st.info("📦 Dados carregados do cache")
        return price_data
    
    # Baixar
    st.info("📥 Baixando dados do mercado...")
    
    from core import data
    
    with st.spinner("Carregando preços..."):
        try:
            price_data = data.get_price_history(tickers, start_date, end_date)
            
            if not price_data.empty:
                # Salvar cache
                salvar_dados_cache(tickers, start_date, end_date, price_data, None)
                st.success(f"✓ {len(price_data)} dias carregados")
                return price_data
            
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    return None


# ==========================================
# FUNÇÕES DE CÁLCULO
# ==========================================

def calcular_retornos(prices):
    """Calcula retornos diários"""
    return prices.pct_change().dropna()


def calcular_retorno_anual(weights, returns):
    """Calcula retorno anualizado"""
    return np.sum(returns.mean() * weights) * 252


def calcular_volatilidade_anual(weights, returns):
    """Calcula volatilidade anualizada"""
    cov_matrix = returns.cov() * 252
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)


def calcular_sharpe(weights, returns, rf_rate):
    """Calcula Sharpe Ratio"""
    ret = calcular_retorno_anual(weights, returns)
    vol = calcular_volatilidade_anual(weights, returns)
    return (ret - rf_rate) / vol


# ==========================================
# OTIMIZAÇÃO
# ==========================================

def otimizar_sharpe_maximo(returns, rf_rate):
    """
    Encontra portfólio de máximo Sharpe
    
    Args:
        returns: DataFrame de retornos
        rf_rate: Taxa livre de risco
        
    Returns:
        Tuple (weights_dict, metrics_dict)
    """
    n_assets = len(returns.columns)
    
    def objetivo(weights):
        return -calcular_sharpe(weights, returns, rf_rate)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objetivo,
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return None, None
    
    weights = result.x
    
    metrics = {
        'return': calcular_retorno_anual(weights, returns),
        'volatility': calcular_volatilidade_anual(weights, returns),
        'sharpe': calcular_sharpe(weights, returns, rf_rate)
    }
    
    weights_dict = dict(zip(returns.columns, weights))
    
    return weights_dict, metrics


def otimizar_minima_volatilidade(returns):
    """
    Encontra portfólio de mínima volatilidade
    
    Args:
        returns: DataFrame de retornos
        
    Returns:
        Tuple (weights_dict, metrics_dict)
    """
    n_assets = len(returns.columns)
    
    def objetivo(weights):
        return calcular_volatilidade_anual(weights, returns)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        objetivo,
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        return None, None
    
    weights = result.x
    
    metrics = {
        'return': calcular_retorno_anual(weights, returns),
        'volatility': calcular_volatilidade_anual(weights, returns),
        'sharpe': calcular_sharpe(weights, returns, 0.0)
    }
    
    weights_dict = dict(zip(returns.columns, weights))
    
    return weights_dict, metrics


# ==========================================
# SIMULAÇÃO
# ==========================================

def simular_performance(weights_dict, prices, capital=10000):
    """
    Simula performance histórica
    
    Args:
        weights_dict: Dict com pesos
        prices: DataFrame com preços
        capital: Capital inicial
        
    Returns:
        Series com valores ao longo do tempo
    """
    returns = calcular_retornos(prices)
    
    weights = np.array([weights_dict.get(col, 0) for col in returns.columns])
    
    portfolio_returns = returns.dot(weights)
    
    portfolio_value = capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value


def calcular_drawdown(portfolio_value):
    """Calcula drawdown"""
    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    return drawdown


# ==========================================
# VISUALIZAÇÕES
# ==========================================

def criar_grafico_composicao(weights_dict, title):
    """Gráfico de pizza da composição"""
    
    # Filtrar > 1%
    weights_filtrado = {k: v for k, v in weights_dict.items() if v > 0.01}
    weights_ordenado = dict(sorted(weights_filtrado.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(data=[go.Pie(
        labels=list(weights_ordenado.keys()),
        values=list(weights_ordenado.values()),
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{value:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )
    
    return fig


def criar_grafico_barras(weights_sharpe, weights_minvol):
    """Gráfico de barras comparativo"""
    
    all_tickers = sorted(set(list(weights_sharpe.keys()) + list(weights_minvol.keys())))
    
    sharpe_values = [weights_sharpe.get(t, 0) * 100 for t in all_tickers]
    minvol_values = [weights_minvol.get(t, 0) * 100 for t in all_tickers]
    
    # Filtrar > 1%
    filtered = [(t, s, m) for t, s, m in zip(all_tickers, sharpe_values, minvol_values) if s > 1 or m > 1]
    
    if not filtered:
        return None
    
    tickers, sharpe_vals, minvol_vals = zip(*filtered)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Máximo Sharpe',
        x=list(tickers),
        y=list(sharpe_vals),
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        name='Mínima Volatilidade',
        x=list(tickers),
        y=list(minvol_vals),
        marker_color='#2ecc71'
    ))
    
    fig.update_layout(
        title='Comparação de Alocação',
        xaxis_title='Ativo',
        yaxis_title='Alocação (%)',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def criar_grafico_performance(value_sharpe, value_minvol):
    """Gráfico de performance histórica"""
    
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
        title='Performance Histórica',
        xaxis_title='Data',
        yaxis_title='Valor do Portfólio (R$)',
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def criar_grafico_drawdown(dd_sharpe, dd_minvol):
    """Gráfico de drawdown"""
    
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
    """Função principal"""
    
    st.title("⚖️ Sharpe vs Mínima Volatilidade")
    st.markdown("Comparação detalhada entre duas estratégias de otimização")
    st.markdown("---")
    
    # Verificar portfólio
    if not st.session_state.portfolio_tickers:
        st.warning("⚠️ Nenhum ativo no portfólio")
        st.info("👉 Vá para **Selecionar Ativos** primeiro")
        st.stop()
    
    if len(st.session_state.portfolio_tickers) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Cache info
        from core.cache import info_cache
        cache_info = info_cache()
        if cache_info['entries'] > 0:
            st.success(f"📦 {cache_info['entries']} cache(s)")
            if st.button("🗑️ Limpar Cache"):
                from core.cache import limpar_cache
                limpar_cache()
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📅 Período")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start = st.date_input("Início", value=st.session_state.period_start)
        
        with col2:
            end = st.date_input("Fim", value=st.session_state.period_end)
        
        st.session_state.period_start = datetime.combine(start, datetime.min.time())
        st.session_state.period_end = datetime.combine(end, datetime.min.time())
        
        st.markdown("---")
        
        st.subheader("💰 Parâmetros")
        
        rf_rate = st.number_input(
            "Taxa Livre de Risco (anual)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_free_rate,
            step=0.0001,
            format="%.4f"
        )
        st.session_state.risk_free_rate = rf_rate
        
        capital = st.number_input(
            "Capital Inicial (R$)",
            min_value=1000.0,
            value=10000.0,
            step=1000.0
        )
        
        st.markdown("---")
        
        btn_calcular = st.button(
            "🔄 Calcular",
            type="primary",
            use_container_width=True
        )
    
    # Info
    st.info(f"📊 **{len(st.session_state.portfolio_tickers)} ativos** selecionados")
    
    with st.expander("📋 Ver lista"):
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.portfolio_tickers):
            with cols[idx % 5]:
                st.write(f"• {ticker}")
    
    st.markdown("---")
    
    # Calcular
    if btn_calcular:
        
        # Carregar dados COM CACHE
        price_data = carregar_dados_com_cache(
            st.session_state.portfolio_tickers,
            st.session_state.period_start,
            st.session_state.period_end
        )
        
        if price_data is None or price_data.empty:
            st.error("❌ Não foi possível carregar dados")
            st.stop()
        
        # Limpar
        price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.8)
        
        if price_data.empty or len(price_data.columns) < 2:
            st.error("❌ Dados insuficientes")
            st.stop()
        
        st.success(f"✓ {len(price_data)} dias, {len(price_data.columns)} ativos")
        
        # Calcular retornos
        with st.spinner("📊 Calculando retornos..."):
            returns = calcular_retornos(price_data)
        
        # Otimizar
        st.subheader("🎯 Otimização")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Otimizando Sharpe..."):
                weights_sharpe, metrics_sharpe = otimizar_sharpe_maximo(returns, rf_rate)
                
                if weights_sharpe:
                    st.success("✅ Sharpe otimizado")
                else:
                    st.error("❌ Falha na otimização")
        
        with col2:
            with st.spinner("Otimizando MinVol..."):
                weights_minvol, metrics_minvol = otimizar_minima_volatilidade(returns)
                
                if weights_minvol:
                    st.success("✅ MinVol otimizado")
                else:
                    st.error("❌ Falha na otimização")
        
        if not weights_sharpe or not weights_minvol:
            st.stop()
            # Salvar no session_state para uso em outras páginas
            st.session_state.portfolios_otimizados = {
                'sharpe_maximo': {
                    'tickers': list(weights_sharpe.keys()),
                    'pesos': list(weights_sharpe.values()),
                    'metricas': metrics_sharpe,
                    'data_calculo': datetime.now(),
                    'periodo': {
                        'inicio': st.session_state.period_start,
                        'fim': st.session_state.period_end
                    }
                },
                'minima_volatilidade': {
                    'tickers': list(weights_minvol.keys()),
                    'pesos': list(weights_minvol.values()),
                    'metricas': metrics_minvol,
                    'data_calculo': datetime.now(),
                    'periodo': {
                        'inicio': st.session_state.period_start,
                        'fim': st.session_state.period_end
                    }
                }
            }

     
        
        st.markdown("---")
        
        # Métricas
        st.header("📈 Métricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Máximo Sharpe")
            
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Retorno", f"{metrics_sharpe['return']:.2%}")
            with subcol2:
                st.metric("Volatilidade", f"{metrics_sharpe['volatility']:.2%}")
            with subcol3:
                st.metric("Sharpe", f"{metrics_sharpe['sharpe']:.3f}")
        
        with col2:
            st.subheader("🛡️ Mínima Volatilidade")
            
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Retorno", f"{metrics_minvol['return']:.2%}")
            with subcol2:
                st.metric("Volatilidade", f"{metrics_minvol['volatility']:.2%}")
            with subcol3:
                st.metric("Sharpe", f"{metrics_minvol['sharpe']:.3f}")
        
        st.markdown("---")
        
        # Composição
        st.header("🥧 Composição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = criar_grafico_composicao(weights_sharpe, "Máximo Sharpe")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = criar_grafico_composicao(weights_minvol, "Mínima Volatilidade")
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparação
        fig = criar_grafico_barras(weights_sharpe, weights_minvol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Performance
        st.header("📊 Performance Histórica")
        
        with st.spinner("Simulando..."):
            value_sharpe = simular_performance(weights_sharpe, price_data, capital)
            value_minvol = simular_performance(weights_minvol, price_data, capital)
        
        fig = criar_grafico_performance(value_sharpe, value_minvol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Valores finais
        col1, col2 = st.columns(2)
        
        with col1:
            final_sharpe = value_sharpe.iloc[-1]
            ret_sharpe = (final_sharpe / capital - 1) * 100
            st.metric(
                "Valor Final - Sharpe",
                f"R$ {final_sharpe:,.2f}",
                f"{ret_sharpe:+.2f}%"
            )
        
        with col2:
            final_minvol = value_minvol.iloc[-1]
            ret_minvol = (final_minvol / capital - 1) * 100
            st.metric(
                "Valor Final - MinVol",
                f"R$ {final_minvol:,.2f}",
                f"{ret_minvol:+.2f}%"
            )
        
        st.markdown("---")
        
        # Drawdown
        st.header("📉 Análise de Drawdown")
        
        dd_sharpe = calcular_drawdown(value_sharpe)
        dd_minvol = calcular_drawdown(value_minvol)
        
        fig = criar_grafico_drawdown(dd_sharpe, dd_minvol)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_dd_sharpe = dd_sharpe.min() * 100
            st.metric("Máximo DD - Sharpe", f"{max_dd_sharpe:.2f}%")
        
        with col2:
            max_dd_minvol = dd_minvol.min() * 100
            st.metric("Máximo DD - MinVol", f"{max_dd_minvol:.2f}%")
        
        st.markdown("---")
        
        # Tabela resumo
        st.header("📋 Resumo")
        
        df_resumo = pd.DataFrame({
            'Métrica': [
                'Retorno Anual',
                'Volatilidade',
                'Sharpe Ratio',
                'Valor Final',
                'Retorno Total',
                'Máximo Drawdown'
            ],
            'Máximo Sharpe': [
                f"{metrics_sharpe['return']:.2%}",
                f"{metrics_sharpe['volatility']:.2%}",
                f"{metrics_sharpe['sharpe']:.3f}",
                f"R$ {final_sharpe:,.2f}",
                f"{ret_sharpe:+.2f}%",
                f"{max_dd_sharpe:.2f}%"
            ],
            'Mínima Volatilidade': [
                f"{metrics_minvol['return']:.2%}",
                f"{metrics_minvol['volatility']:.2%}",
                f"{metrics_minvol['sharpe']:.3f}",
                f"R$ {final_minvol:,.2f}",
                f"{ret_minvol:+.2f}%",
                f"{max_dd_minvol:.2f}%"
            ]
        })
        
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)
    
    else:
        st.info("👈 Configure os parâmetros e clique em **Calcular**")


if __name__ == "__main__":
    main()
