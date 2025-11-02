"""
Portfólios Eficientes
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data, metrics

st.set_page_config(
    page_title="Portfólios Eficientes",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# INICIALIZAÇÃO DO SESSION STATE
# ==========================================
def init_session_state():
    """Inicializa todas as variáveis necessárias"""
    
    # Taxa livre de risco (CDI anual aproximado)
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 0.1175  # 11.75% ao ano
    
    # Período
    if 'period_start' not in st.session_state:
        st.session_state.period_start = datetime.now() - timedelta(days=365)
    
    if 'period_end' not in st.session_state:
        st.session_state.period_end = datetime.now()
    
    # Tickers
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    # Dados de preços
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    
    # Resultados de otimização
    if 'efficient_frontier' not in st.session_state:
        st.session_state.efficient_frontier = None
    
    if 'optimal_portfolios' not in st.session_state:
        st.session_state.optimal_portfolios = None

# CHAMAR IMEDIATAMENTE
init_session_state()

# ... resto do código
