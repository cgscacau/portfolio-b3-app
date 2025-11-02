"""
Inicialização global do session state
Importar este módulo em todas as páginas para garantir que todas as variáveis existam
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd


def init_all():
    """
    Inicializa todas as variáveis do session state usadas no app
    Chamado no início de cada página
    """
    
    # ==========================================
    # UNIVERSO DE ATIVOS
    # ==========================================
    if 'universe_df' not in st.session_state:
        st.session_state.universe_df = pd.DataFrame()
    
    # ==========================================
    # SELEÇÃO DE ATIVOS
    # ==========================================
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []
    
    # ==========================================
    # PERÍODO DE ANÁLISE
    # ==========================================
    if 'period_start' not in st.session_state:
        st.session_state.period_start = datetime.now() - timedelta(days=365)
    
    if 'period_end' not in st.session_state:
        st.session_state.period_end = datetime.now()
    
    # ==========================================
    # DADOS DE MERCADO
    # ==========================================
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    
    if 'dividend_data' not in st.session_state:
        st.session_state.dividend_data = None
    
    if 'returns_data' not in st.session_state:
        st.session_state.returns_data = None
    
    # ==========================================
    # PARÂMETROS DE OTIMIZAÇÃO
    # ==========================================
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 0.1175  # CDI aproximado 11.75% a.a.
    
    if 'num_portfolios' not in st.session_state:
        st.session_state.num_portfolios = 5000
    
    # ==========================================
    # RESULTADOS DE OTIMIZAÇÃO
    # ==========================================
    if 'efficient_frontier' not in st.session_state:
        st.session_state.efficient_frontier = None
    
    if 'optimal_portfolios' not in st.session_state:
        st.session_state.optimal_portfolios = None
    
    if 'portfolio_metrics' not in st.session_state:
        st.session_state.portfolio_metrics = None
    
    # ==========================================
    # CONFIGURAÇÕES DE VISUALIZAÇÃO
    # ==========================================
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False
    
    if 'chart_height' not in st.session_state:
        st.session_state.chart_height = 600
    
    # ==========================================
    # CACHE DE CÁLCULOS
    # ==========================================
    if 'last_calculation_time' not in st.session_state:
        st.session_state.last_calculation_time = None
    
    if 'calculation_status' not in st.session_state:
        st.session_state.calculation_status = 'idle'  # idle, loading, success, error


def reset_data():
    """
    Reseta todos os dados carregados
    Útil quando o usuário muda a seleção de ativos ou período
    """
    st.session_state.price_data = None
    st.session_state.dividend_data = None
    st.session_state.returns_data = None
    st.session_state.efficient_frontier = None
    st.session_state.optimal_portfolios = None
    st.session_state.portfolio_metrics = None
    st.session_state.calculation_status = 'idle'


def reset_optimization():
    """
    Reseta apenas os resultados de otimização
    Mantém os dados de preços carregados
    """
    st.session_state.efficient_frontier = None
    st.session_state.optimal_portfolios = None
    st.session_state.portfolio_metrics = None
    st.session_state.calculation_status = 'idle'


def get_session_info():
    """
    Retorna informações sobre o estado atual da sessão
    Útil para debug
    
    Returns:
        Dict com informações da sessão
    """
    return {
        'num_selected_tickers': len(st.session_state.selected_tickers),
        'num_portfolio_tickers': len(st.session_state.portfolio_tickers),
        'has_price_data': st.session_state.price_data is not None,
        'has_dividend_data': st.session_state.dividend_data is not None,
        'has_optimization': st.session_state.optimal_portfolios is not None,
        'period': f"{st.session_state.period_start.date()} to {st.session_state.period_end.date()}",
        'risk_free_rate': st.session_state.risk_free_rate,
        'calculation_status': st.session_state.calculation_status
    }


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def validate_session_state():
    """
    Valida o estado da sessão e retorna warnings se houver problemas
    
    Returns:
        List de mensagens de aviso
    """
    warnings = []
    
    # Verificar se há ativos selecionados
    if not st.session_state.portfolio_tickers:
        warnings.append("Nenhum ativo no portfólio")
    
    # Verificar se há poucos ativos para otimização
    if len(st.session_state.portfolio_tickers) < 2:
        warnings.append("Selecione pelo menos 2 ativos para otimização")
    
    # Verificar período
    if st.session_state.period_start >= st.session_state.period_end:
        warnings.append("Data inicial deve ser anterior à data final")
    
    # Verificar se período é muito curto
    dias = (st.session_state.period_end - st.session_state.period_start).days
    if dias < 30:
        warnings.append("Período muito curto (mínimo recomendado: 30 dias)")
    
    return warnings


def is_ready_for_optimization():
    """
    Verifica se o sistema está pronto para otimização
    
    Returns:
        Tuple (bool, str): (está pronto, mensagem)
    """
    # Verificar ativos
    if not st.session_state.portfolio_tickers:
        return False, "Nenhum ativo selecionado"
    
    if len(st.session_state.portfolio_tickers) < 2:
        return False, "Selecione pelo menos 2 ativos"
    
    # Verificar dados
    if st.session_state.price_data is None:
        return False, "Carregue os dados de preços primeiro"
    
    # Verificar período
    if st.session_state.period_start >= st.session_state.period_end:
        return False, "Período inválido"
    
    return True, "Pronto para otimização"
