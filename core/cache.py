"""
Sistema de cache global para dados de mercado
Evita downloads repetidos entre páginas
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib


def criar_chave_cache(tickers, start_date, end_date):
    """
    Cria chave única para cache baseada nos parâmetros
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        
    Returns:
        String com hash único
    """
    # Ordenar tickers para garantir mesma chave
    tickers_sorted = sorted(tickers)
    
    # Criar string única
    cache_str = f"{','.join(tickers_sorted)}_{start_date}_{end_date}"
    
    # Gerar hash
    return hashlib.md5(cache_str.encode()).hexdigest()


def salvar_dados_cache(tickers, start_date, end_date, price_data, dividend_data=None):
    """
    Salva dados no cache global do session_state
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        price_data: DataFrame com preços
        dividend_data: Dict com dividendos (opcional)
    """
    chave = criar_chave_cache(tickers, start_date, end_date)
    
    # Inicializar cache se não existir
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    
    # Salvar dados
    st.session_state.data_cache[chave] = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'price_data': price_data,
        'dividend_data': dividend_data,
        'timestamp': datetime.now()
    }


def carregar_dados_cache(tickers, start_date, end_date):
    """
    Carrega dados do cache se existirem
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        
    Returns:
        Tuple (price_data, dividend_data) ou (None, None) se não existir
    """
    if 'data_cache' not in st.session_state:
        return None, None
    
    chave = criar_chave_cache(tickers, start_date, end_date)
    
    if chave in st.session_state.data_cache:
        cache_entry = st.session_state.data_cache[chave]
        
        # Verificar se não está muito antigo (máx 1 hora)
        idade = (datetime.now() - cache_entry['timestamp']).total_seconds()
        if idade < 3600:  # 1 hora
            return cache_entry['price_data'], cache_entry['dividend_data']
    
    return None, None


def limpar_cache():
    """Limpa todo o cache de dados"""
    if 'data_cache' in st.session_state:
        st.session_state.data_cache = {}


def info_cache():
    """
    Retorna informações sobre o cache atual
    
    Returns:
        Dict com estatísticas do cache
    """
    if 'data_cache' not in st.session_state:
        return {'entries': 0, 'oldest': None, 'newest': None}
    
    cache = st.session_state.data_cache
    
    if not cache:
        return {'entries': 0, 'oldest': None, 'newest': None}
    
    timestamps = [entry['timestamp'] for entry in cache.values()]
    
    return {
        'entries': len(cache),
        'oldest': min(timestamps),
        'newest': max(timestamps)
    }
