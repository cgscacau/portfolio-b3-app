"""
core/data.py
Sistema de coleta, cache e limpeza de dados da B3
VERSÃO SIMPLIFICADA - Apenas yfinance (sem mock)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional, Union
import time
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Diretórios
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

ASSETS_DIR = Path("assets")
B3_UNIVERSE_FILE = ASSETS_DIR / "b3_universe.csv"


class DataCache:
    """Gerenciador de cache em disco para dados históricos."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, tickers: List[str], start_date: datetime, 
                     end_date: datetime, data_type: str = "prices") -> str:
        """Gera chave única para combinação de parâmetros."""
        tickers_sorted = sorted(tickers)
        key_string = f"{data_type}_{tickers_sorted}_{start_date.date()}_{end_date.date()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Carrega dados do cache se ainda válidos."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        
        if file_age_hours > max_age_hours:
            logger.info(f"Cache expirado: {cache_key} ({file_age_hours:.1f}h)")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Cache carregado: {cache_key} ({file_age_hours:.1f}h)")
            return data
        except Exception as e:
            logger.error(f"Erro ao carregar cache {cache_key}: {e}")
            return None
    
    def save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Salva dados no cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cache salvo: {cache_key}")
        except Exception as e:
            logger.error(f"Erro ao salvar cache {cache_key}: {e}")


@st.cache_data(ttl=86400)
def load_ticker_universe() -> pd.DataFrame:
    """Carrega universo de tickers B3 com metadados."""
    try:
        if not B3_UNIVERSE_FILE.exists():
            logger.error(f"Arquivo não encontrado: {B3_UNIVERSE_FILE}")
            st.error(f"❌ Arquivo de universo não encontrado: {B3_UNIVERSE_FILE}")
            return pd.DataFrame()
        
        df = pd.read_csv(B3_UNIVERSE_FILE)
        
        expected_cols = ['ticker', 'nome', 'setor', 'subsetor', 'segmento_listagem', 'tipo']
        missing_cols = set(expected_cols) - set(df.columns)
        
        if missing_cols:
            logger.error(f"Colunas faltando: {missing_cols}")
            st.error(f"❌ Arquivo com formato inválido")
            return pd.DataFrame()
        
        logger.info(f"Universo carregado: {len(df)} tickers")
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        st.error(f"❌ Erro: {e}")
        return pd.DataFrame()


def download_ticker_data(ticker: str, start: datetime, end: datetime, 
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Baixa dados de um ticker usando yf.Ticker().history() (mais confiável).
    
    Args:
        ticker: Ticker do ativo
        start: Data inicial
        end: Data final
        max_retries: Número máximo de tentativas
    
    Returns:
        DataFrame com dados OHLCV ou None se falhar
    """
    for attempt in range(max_retries):
        try:
            # Usar yf.Ticker().history() ao invés de yf.download()
            stock = yf.Ticker(ticker)
            
            # Baixar histórico
            data = stock.history(start=start, end=end)
            
            if not data.empty:
                logger.info(f"✅ {ticker}: {len(data)} dias baixados")
                return data
            else:
                logger.warning(f"⚠️ {ticker}: sem dados no período")
                
        except Exception as e:
            logger.warning(f"❌ {ticker} (tentativa {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)  # Pausa antes de tentar novamente
    
    return None


def filter_traded_last_30d(df: pd.DataFrame, min_sessions: int = 5, 
                          min_avg_volume: float = 100000,
                          show_progress: bool = True) -> pd.DataFrame:
    """
    Filtra ativos negociados nos últimos 30 dias.
    VERSÃO OTIMIZADA usando yf.Ticker().history()
    """
    if df.empty:
        return df
    
    df = df.copy()
    df['is_traded_30d'] = False
    df['avg_volume_30d'] = 0.0
    df['sessions_traded_30d'] = 0
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    total = len(df)
    traded_count = 0
    
    # Calcular período
    end_date = datetime.now()
    start_date = end_date - timedelta(days=35)
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        
        if show_progress:
            status_text.text(f"Verificando: {ticker} ({idx+1}/{total})")
        
        try:
            # Usar função otimizada
            data = download_ticker_data(ticker, start_date, end_date, max_retries=2)
            
            if data is not None and not data.empty and 'Volume' in data.columns:
                valid_sessions = data[data['Volume'] > 0]
                
                sessions_traded = len(valid_sessions)
                avg_volume = valid_sessions['Volume'].mean() if len(valid_sessions) > 0 else 0
                
                df.at[idx, 'sessions_traded_30d'] = int(sessions_traded)
                df.at[idx, 'avg_volume_30d'] = float(avg_volume)
                
                if sessions_traded >= min_sessions and avg_volume >= min_avg_volume:
                    df.at[idx, 'is_traded_30d'] = True
                    traded_count += 1
        
        except Exception as e:
            logger.warning(f"Erro ao verificar {ticker}: {e}")
            continue
        
        if show_progress:
            progress_bar.progress((idx + 1) / total)
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    logger.info(f"Ativos líquidos: {traded_count}/{total}")
    
    return df


@st.cache_data(ttl=3600)
def get_price_history(tickers: List[str], start: datetime, end: datetime,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Obtém histórico de preços usando método otimizado.
    """
    if not tickers:
        return pd.DataFrame()
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "prices")
    
    # Tentar cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key)
        if cached_data is not None:
            st.success(f"✅ Dados carregados do cache ({len(cached_data)} dias)")
            return cached_data
    
    st.info(f"📥 Baixando histórico de {len(tickers)} ativos...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    prices_dict = {}
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Baixando: {ticker} ({idx+1}/{len(tickers)})")
        
        data = download_ticker_data(ticker, start, end)
        
        if data is not None and not data.empty:
            # Usar Close como preço ajustado (yfinance já ajusta automaticamente)
            if 'Close' in data.columns:
                prices_dict[ticker] = data['Close']
        
        progress_bar.progress((idx + 1) / len(tickers))
        
        # Pequena pausa para não sobrecarregar
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    if not prices_dict:
        st.error("❌ Nenhum dado obtido")
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(prices_dict)
    prices_df = prices_df.dropna(how='all')
    prices_df = prices_df.sort_index()
    
    # Salvar cache
    if use_cache and not prices_df.empty:
        cache_manager.save_to_cache(cache_key, prices_df)
    
    st.success(f"✅ Obtidos: {len(prices_df)} dias, {len(prices_df.columns)} ativos")
    
    return prices_df


@st.cache_data(ttl=3600)
def get_dividends(tickers: List[str], start: datetime, end: datetime,
                 use_cache: bool = True) -> Dict[str, pd.Series]:
    """
    Obtém histórico de dividendos usando método otimizado.
    """
    if not tickers:
        return {}
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "dividends")
    
    # Tentar cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key, max_age_hours=12)
        if cached_data is not None:
            st.success(f"✅ Dividendos carregados do cache")
            return {col: cached_data[col].dropna() for col in cached_data.columns}
    
    st.info(f"📥 Baixando dividendos de {len(tickers)} ativos...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dividends_dict = {}
    success_count = 0
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Obtendo dividendos: {ticker} ({idx+1}/{len(tickers)})")
        
        try:
            stock = yf.Ticker(ticker)
            divs = stock.dividends
            
            if not divs.empty:
                # Filtrar por período
                divs = divs[(divs.index >= start) & (divs.index <= end)]
                
                if not divs.empty:
                    dividends_dict[ticker] = divs
                    success_count += 1
        
        except Exception as e:
            logger.warning(f"Erro ao obter dividendos de {ticker}: {e}")
        
        progress_bar.progress((idx + 1) / len(tickers))
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    # Salvar cache
    if dividends_dict:
        all_dates = pd.DatetimeIndex([])
        for series in dividends_dict.values():
            all_dates = all_dates.union(series.index)
        
        divs_df = pd.DataFrame(index=all_dates.sort_values())
        for ticker, series in dividends_dict.items():
            divs_df[ticker] = series
        
        if use_cache:
            cache_manager.save_to_cache(cache_key, divs_df)
        
        st.success(f"✅ Dividendos: {success_count}/{len(tickers)} ativos")
    else:
        st.warning("⚠️ Nenhum dividendo encontrado")
    
    return dividends_dict


@st.cache_data(ttl=1800)
def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Obtém preços atuais."""
    if not tickers:
        return {}
    
    prices = {}
    end = datetime.now()
    start = end - timedelta(days=7)
    
    st.info("📥 Obtendo preços atuais...")
    
    for ticker in tickers:
        data = download_ticker_data(ticker, start, end)
        
        if data is not None and not data.empty and 'Close' in data.columns:
            last_price = data['Close'].iloc[-1]
            if not np.isnan(last_price):
                prices[ticker] = float(last_price)
    
    st.success(f"✅ Preços obtidos: {len(prices)} ativos")
    
    return prices


def validate_data_quality(prices_df: pd.DataFrame, 
                         min_data_points: int = 252,
                         max_missing_pct: float = 0.1) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Valida qualidade dos dados."""
    if prices_df.empty:
        return prices_df, [], {}
    
    removed_tickers = []
    removal_reasons = {}
    
    total_days = len(prices_df)
    
    for col in prices_df.columns:
        valid_points = prices_df[col].notna().sum()
        missing_pct = 1 - (valid_points / total_days)
        
        if valid_points < min_data_points:
            removed_tickers.append(col)
            removal_reasons[col] = f"Dados insuficientes: {valid_points} pontos"
            continue
        
        if missing_pct > max_missing_pct:
            removed_tickers.append(col)
            removal_reasons[col] = f"Muitos dados faltantes: {missing_pct*100:.1f}%"
            continue
    
    clean_df = prices_df.drop(columns=removed_tickers, errors='ignore')
    
    if clean_df.empty:
        st.error("❌ Todos os ativos removidos por dados insuficientes")
        return clean_df, removed_tickers, removal_reasons
    
    # Forward fill
    clean_df = clean_df.fillna(method='ffill', limit=5)
    clean_df = clean_df.dropna(how='any')
    
    if removed_tickers:
        st.warning(f"⚠️ {len(removed_tickers)} ativos removidos")
        
        with st.expander("Ver detalhes"):
            for ticker, reason in removal_reasons.items():
                st.text(f"• {ticker}: {reason}")
    
    return clean_df, removed_tickers, removal_reasons


def calculate_returns(prices_df: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """Calcula retornos diários."""
    if prices_df.empty:
        return pd.DataFrame()
    
    if method == 'log':
        returns = np.log(prices_df / prices_df.shift(1))
    else:
        returns = prices_df.pct_change()
    
    return returns.dropna()


def verify_module():
    """Verifica configuração do módulo."""
    checks = {
        'cache_dir_exists': CACHE_DIR.exists(),
        'assets_dir_exists': ASSETS_DIR.exists(),
        'universe_file_exists': B3_UNIVERSE_FILE.exists(),
    }
    
    all_ok = all(checks.values())
    
    if not all_ok:
        logger.warning(f"Verificação: {checks}")
    
    return all_ok


if __name__ != "__main__":
    verify_module()
