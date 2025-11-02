"""
Sistema de cache otimizado para dados de mercado
Usa decoradores nativos do Streamlit para máxima eficiência
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import hashlib
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURAÇÕES DE TTL POR TIPO DE DADO
# ==========================================

class CacheConfig:
    """Configurações de TTL para diferentes tipos de dados"""
    
    # Dados históricos (raramente mudam)
    HISTORICAL_DATA_TTL = 24 * 3600  # 24 horas
    
    # Preços atuais (mudam frequentemente)
    CURRENT_PRICE_TTL = 5 * 60  # 5 minutos
    
    # Dividendos (mudam raramente)
    DIVIDENDS_TTL = 12 * 3600  # 12 horas
    
    # Informações de ativos (quase nunca mudam)
    ASSET_INFO_TTL = 7 * 24 * 3600  # 7 dias
    
    # Cache de sessão (até recarregar página)
    SESSION_TTL = None  # Sem expiração


# ==========================================
# DECORADORES DE CACHE
# ==========================================

def cache_historical_data(func: Callable) -> Callable:
    """
    Decorator para cachear dados históricos
    TTL: 24 horas
    """
    @st.cache_data(ttl=CacheConfig.HISTORICAL_DATA_TTL, show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cache_current_price(func: Callable) -> Callable:
    """
    Decorator para cachear preços atuais
    TTL: 5 minutos
    """
    @st.cache_data(ttl=CacheConfig.CURRENT_PRICE_TTL, show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cache_dividends(func: Callable) -> Callable:
    """
    Decorator para cachear dividendos
    TTL: 12 horas
    """
    @st.cache_data(ttl=CacheConfig.DIVIDENDS_TTL, show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cache_asset_info(func: Callable) -> Callable:
    """
    Decorator para cachear informações de ativos
    TTL: 7 dias
    """
    @st.cache_data(ttl=CacheConfig.ASSET_INFO_TTL, show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cache_session(func: Callable) -> Callable:
    """
    Decorator para cachear durante a sessão
    Sem TTL (até recarregar página)
    """
    @st.cache_data(show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cache_resource(func: Callable) -> Callable:
    """
    Decorator para cachear recursos não-serializáveis
    (conexões, objetos complexos, etc)
    """
    @st.cache_resource(show_spinner=False)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ==========================================
# UTILITÁRIOS DE CACHE
# ==========================================

def criar_chave_cache(*args, **kwargs) -> str:
    """
    Cria chave única para cache baseada nos argumentos
    
    Args:
        *args: Argumentos posicionais
        **kwargs: Argumentos nomeados
        
    Returns:
        String com hash único
    """
    # Converter args e kwargs em string única
    cache_str = str(args) + str(sorted(kwargs.items()))
    
    # Gerar hash MD5
    return hashlib.md5(cache_str.encode()).hexdigest()


def limpar_cache_completo():
    """Limpa todo o cache do Streamlit"""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cache completo limpo")


def limpar_cache_dados():
    """Limpa apenas cache de dados"""
    st.cache_data.clear()
    logger.info("Cache de dados limpo")


def limpar_cache_recursos():
    """Limpa apenas cache de recursos"""
    st.cache_resource.clear()
    logger.info("Cache de recursos limpo")


# ==========================================
# ESTATÍSTICAS DE CACHE
# ==========================================

class CacheStats:
    """Classe para monitorar estatísticas de cache"""
    
    def __init__(self):
        self._inicializar_stats()
    
    def _inicializar_stats(self):
        """Inicializa stats no session_state se não existir"""
        if 'cache_stats' not in st.session_state:
            st.session_state.cache_stats = {
                'hits': 0,
                'misses': 0,
                'last_clear': None,
                'data_requests': 0
            }
    
    def registrar_hit(self):
        """Registra um cache hit"""
        self._inicializar_stats()
        st.session_state.cache_stats['hits'] += 1
    
    def registrar_miss(self):
        """Registra um cache miss"""
        self._inicializar_stats()
        st.session_state.cache_stats['misses'] += 1
    
    def registrar_request(self):
        """Registra uma requisição de dados"""
        self._inicializar_stats()
        st.session_state.cache_stats['data_requests'] += 1
    
    def obter_taxa_acerto(self) -> float:
        """Calcula taxa de acerto do cache"""
        self._inicializar_stats()
        total = st.session_state.cache_stats['hits'] + st.session_state.cache_stats['misses']
        if total == 0:
            return 0.0
        return (st.session_state.cache_stats['hits'] / total) * 100
    
    def obter_estatisticas(self) -> Dict[str, Any]:
        """Retorna estatísticas completas"""
        self._inicializar_stats()
        stats = st.session_state.cache_stats.copy()
        stats['hit_rate'] = self.obter_taxa_acerto()
        return stats
    
    def resetar(self):
        """Reseta estatísticas"""
        st.session_state.cache_stats = {
            'hits': 0,
            'misses': 0,
            'last_clear': datetime.now(),
            'data_requests': 0
        }
        logger.info("Estatísticas de cache resetadas")


# ==========================================
# GERENCIADOR DE CACHE
# ==========================================

class CacheManager:
    """Gerenciador centralizado de cache"""
    
    def __init__(self):
        self.stats = CacheStats()
    
    def limpar_por_tipo(self, tipo: str):
        """
        Limpa cache por tipo de dado
        
        Args:
            tipo: 'historical', 'prices', 'dividends', 'info', 'all'
        """
        if tipo == 'all':
            limpar_cache_completo()
        elif tipo == 'historical':
            limpar_cache_dados()
        elif tipo == 'prices':
            limpar_cache_dados()
        elif tipo == 'dividends':
            limpar_cache_dados()
        elif tipo == 'info':
            limpar_cache_dados()
        else:
            logger.warning(f"Tipo de cache desconhecido: {tipo}")
        
        self.stats.registrar_miss()
    
    def obter_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o cache"""
        return {
            'stats': self.stats.obter_estatisticas(),
            'config': {
                'historical_ttl': f"{CacheConfig.HISTORICAL_DATA_TTL / 3600:.1f}h",
                'current_price_ttl': f"{CacheConfig.CURRENT_PRICE_TTL / 60:.1f}min",
                'dividends_ttl': f"{CacheConfig.DIVIDENDS_TTL / 3600:.1f}h",
                'asset_info_ttl': f"{CacheConfig.ASSET_INFO_TTL / (24 * 3600):.1f}d"
            },
            # Campos para compatibilidade com código antigo
            'entries': self.stats.obter_estatisticas()['data_requests'],
            'oldest': self.stats.obter_estatisticas().get('last_clear', None),
            'newest': datetime.now()
        }
    
    def exibir_painel_controle(self):
        """Exibe painel de controle do cache no Streamlit"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Cache")
        
        info = self.obter_info()
        stats = info['stats']
        
        # Estatísticas
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Hits", stats['hits'])
        with col2:
            st.metric("Misses", stats['misses'])
        
        st.sidebar.metric("Taxa de Acerto", f"{stats['hit_rate']:.1f}%")
        st.sidebar.metric("Requisições", stats['data_requests'])
        
        # Botão de limpar cache
        if st.sidebar.button("🗑️ Limpar Cache", use_container_width=True, key="btn_limpar_cache_sidebar"):
            try:
                # Limpar cache do Streamlit
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Resetar estatísticas
                self.stats.resetar()
                
                # Mensagem de sucesso
                st.sidebar.success("✅ Cache limpo!")
                
                # Aguardar um pouco antes de recarregar
                import time
                time.sleep(0.5)
                
                # Forçar recarregamento
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Erro ao limpar: {str(e)}")




# ==========================================
# INSTÂNCIA GLOBAL
# ==========================================

cache_manager = CacheManager()


# ==========================================
# FUNÇÕES PÚBLICAS (COMPATIBILIDADE)
# ==========================================

def criar_chave_cache_legacy(tickers, start_date, end_date):
    """Mantém compatibilidade com código antigo"""
    tickers_sorted = sorted(tickers)
    cache_str = f"{','.join(tickers_sorted)}_{start_date}_{end_date}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def salvar_dados_cache(tickers, start_date, end_date, price_data, dividend_data=None):
    """Mantém compatibilidade - agora usa cache nativo"""
    logger.info("Função legada salvar_dados_cache() - usando cache nativo")
    pass


def carregar_dados_cache(tickers, start_date, end_date):
    """Mantém compatibilidade - agora usa cache nativo"""
    logger.info("Função legada carregar_dados_cache() - usando cache nativo")
    return None, None


def limpar_cache():
    """Mantém compatibilidade"""
    limpar_cache_completo()


def info_cache():
    """
    Retorna informações sobre o cache atual
    Mantém compatibilidade total com código antigo
    
    Returns:
        Dict com estatísticas do cache
    """
    info = cache_manager.obter_info()
    
    # Retornar no formato esperado pelo código antigo
    return {
        'entries': info['entries'],
        'oldest': info['oldest'],
        'newest': info['newest'],
        'stats': info['stats'],
        'config': info['config']
    }
