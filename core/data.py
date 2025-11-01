"""
Módulo de coleta, cache e processamento de dados da B3
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import List, Dict, Tuple, Optional
import hashlib
import pickle

logger = logging.getLogger(__name__)

# Diretórios
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

ASSETS_DIR = Path("assets")


class DataCache:
    """Gerenciador de cache em disco para dados históricos."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, tickers: List[str], start_date: datetime, 
                      end_date: datetime, data_type: str = "prices") -> str:
        """Gera chave única para combinação de parâmetros."""
        tickers_sorted = sorted(tickers)
        key_string = f"{data_type}_{tickers_sorted}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Carrega dados do cache se existirem e forem recentes."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            
            if age_hours < max_age_hours:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Cache hit: {cache_key} (idade: {age_hours:.1f}h)")
                    return data
                except Exception as e:
                    logger.warning(f"Erro ao carregar cache {cache_key}: {e}")
        
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


# Instância global do cache
data_cache = DataCache()


@st.cache_data(ttl=3600)
def load_ticker_universe() -> pd.DataFrame:
    """
    Carrega universo de tickers da B3 com metadados.
    
    Returns:
        DataFrame com colunas: ticker, nome, setor, subsetor, segmento_listagem, tipo
    """
    try:
        universe_file = ASSETS_DIR / "b3_universe.csv"
        
        if not universe_file.exists():
            logger.error(f"Arquivo {universe_file} não encontrado")
            return pd.DataFrame()
        
        df = pd.read_csv(universe_file)
        
        # Validação básica
        required_cols = ['ticker', 'nome', 'setor', 'subsetor', 'segmento_listagem', 'tipo']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Colunas faltando no arquivo: {set(required_cols) - set(df.columns)}")
            return pd.DataFrame()
        
        logger.info(f"Universo carregado: {len(df)} tickers")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        return pd.DataFrame()


def check_ticker_liquidity(ticker: str, min_sessions: int = 5, 
                           min_avg_volume: float = 10000) -> Tuple[bool, Dict]:
    """
    Verifica se um ticker teve negociação nos últimos 30 dias.
    
    Args:
        ticker: Código do ativo
        min_sessions: Número mínimo de sessões com volume > 0
        min_avg_volume: Volume médio mínimo
    
    Returns:
        (is_liquid, metrics_dict)
    """
    try:
        data = yf.download(ticker, period="35d", progress=False, show_errors=False)
        
        if data.empty or len(data) < 5:
            return False, {'sessions': 0, 'avg_volume': 0}
        
        sessions_traded = (data['Volume'] > 0).sum()
        avg_volume = data['Volume'].mean()
        
        is_liquid = (sessions_traded >= min_sessions) and (avg_volume >= min_avg_volume)
        
        metrics = {
            'sessions': int(sessions_traded),
            'avg_volume': float(avg_volume),
            'last_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0
        }
        
        return is_liquid, metrics
        
    except Exception as e:
        logger.warning(f"Erro ao verificar liquidez de {ticker}: {e}")
        return False, {'sessions': 0, 'avg_volume': 0}


@st.cache_data(ttl=21600)  # Cache de 6 horas
def filter_traded_last_30d(df: pd.DataFrame, min_sessions: int = 5,
                           min_avg_volume: float = 10000) -> pd.DataFrame:
    """
    Filtra DataFrame mantendo apenas tickers negociados nos últimos 30 dias.
    
    Args:
        df: DataFrame com coluna 'ticker'
        min_sessions: Sessões mínimas com negociação
        min_avg_volume: Volume médio mínimo
    
    Returns:
        DataFrame filtrado com colunas adicionais de liquidez
    """
    if df.empty:
        return df
    
    tickers = df['ticker'].tolist()
    
    logger.info(f"Verificando liquidez de {len(tickers)} tickers...")
    
    # Progress bar para UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    liquid_tickers = []
    liquidity_metrics = []
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Verificando {ticker}... ({i+1}/{len(tickers)})")
        
        is_liquid, metrics = check_ticker_liquidity(ticker, min_sessions, min_avg_volume)
        
        if is_liquid:
            liquid_tickers.append(ticker)
            liquidity_metrics.append(metrics)
        
        progress_bar.progress((i + 1) / len(tickers))
        
        # Rate limiting para não sobrecarregar API
        if (i + 1) % 10 == 0:
            time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    # Filtrar DataFrame
    df_filtered = df[df['ticker'].isin(liquid_tickers)].copy()
    
    # Adicionar métricas de liquidez
    metrics_df = pd.DataFrame(liquidity_metrics, index=liquid_tickers)
    df_filtered = df_filtered.set_index('ticker').join(metrics_df).reset_index()
    
    logger.info(f"Tickers líquidos: {len(df_filtered)} de {len(df)}")
    
    return df_filtered


def batch_download_history(tickers: List[str], start_date: datetime,
                           end_date: datetime, batch_size: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Download de histórico de preços em lotes para melhor performance.
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
        batch_size: Tamanho do lote
    
    Returns:
        Dict com ticker -> DataFrame de preços
    """
    all_data = {}
    
    # Verificar cache primeiro
    cache_key = data_cache.get_cache_key(tickers, start_date, end_date, "prices")
    cached_data = data_cache.load_from_cache(cache_key)
    
    if cached_data is not None:
        logger.info("Usando dados de preços do cache")
        return cached_data
    
    logger.info(f"Baixando histórico de {len(tickers)} tickers...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(tickers), batch_size):
        batch = tickers[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        status_text.text(f"Baixando lote {batch_num}/{total_batches}...")
        
        try:
            # Download em grupo
            ticker_string = " ".join(batch)
            data = yf.download(
                ticker_string,
                start=start_date,
                end=end_date,
                progress=False,
                show_errors=False,
                group_by='ticker'
            )
            
            # Processar dados
            if len(batch) == 1:
                # Caso especial: 1 ticker
                ticker = batch[0]
                if not data.empty:
                    all_data[ticker] = data
            else:
                # Múltiplos tickers
                for ticker in batch:
                    try:
                        ticker_data = data[ticker]
                        if not ticker_data.empty and len(ticker_data) > 10:
                            all_data[ticker] = ticker_data
                    except (KeyError, AttributeError):
                        logger.warning(f"Sem dados para {ticker}")
            
            progress_bar.progress(batch_num / total_batches)
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Erro no lote {batch_num}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Salvar no cache
    if all_data:
        data_cache.save_to_cache(cache_key, all_data)
    
    logger.info(f"Download concluído: {len(all_data)} tickers com dados")
    
    return all_data


@st.cache_data(ttl=3600)
def get_price_history(tickers: List[str], start_date: datetime, 
                      end_date: datetime) -> pd.DataFrame:
    """
    Obtém histórico de preços ajustados para lista de tickers.
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
    
    Returns:
        DataFrame com MultiIndex (Date, Ticker) e colunas OHLCV
    """
    if not tickers:
        return pd.DataFrame()
    
    # Download em lote
    data_dict = batch_download_history(tickers, start_date, end_date)
    
    if not data_dict:
        logger.warning("Nenhum dado de preço obtido")
        return pd.DataFrame()
    
    # Consolidar em DataFrame único
    dfs = []
    
    for ticker, data in data_dict.items():
        if data.empty:
            continue
        
        df = data.copy()
        df['Ticker'] = ticker
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, axis=0)
    result = result.reset_index()
    
    # Garantir colunas padronizadas
    result.columns = [col[0] if isinstance(col, tuple) else col for col in result.columns]
    
    logger.info(f"Histórico consolidado: {len(result)} registros, {len(data_dict)} tickers")
    
    return result


def get_dividends_history(tickers: List[str], start_date: datetime,
                          end_date: datetime) -> Dict[str, pd.Series]:
    """
    Obtém histórico de dividendos para lista de tickers.
    
    Args:
        tickers: Lista de tickers
        start_date: Data inicial
        end_date: Data final
    
    Returns:
        Dict com ticker -> Series de dividendos (index=Date, values=dividend)
    """
    # Verificar cache
    cache_key = data_cache.get_cache_key(tickers, start_date, end_date, "dividends")
    cached_data = data_cache.load_from_cache(cache_key)
    
    if cached_data is not None:
        logger.info("Usando dados de dividendos do cache")
        return cached_data
    
    logger.info(f"Baixando dividendos de {len(tickers)} tickers...")
    
    dividends_dict = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Baixando dividendos: {ticker} ({i+1}/{len(tickers)})")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            divs = ticker_obj.dividends
            
            if divs is not None and not divs.empty:
                # Filtrar por período
                divs = divs[(divs.index >= start_date) & (divs.index <= end_date)]
                
                if not divs.empty:
                    dividends_dict[ticker] = divs
            
        except Exception as e:
            logger.warning(f"Erro ao obter dividendos de {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(tickers))
        
        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    # Salvar no cache
    if dividends_dict:
        data_cache.save_to_cache(cache_key, dividends_dict)
    
    logger.info(f"Dividendos obtidos: {len(dividends_dict)} tickers")
    
    return dividends_dict


@st.cache_data(ttl=3600)
def get_latest_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Obtém preços mais recentes para lista de tickers.
    
    Args:
        tickers: Lista de tickers
    
    Returns:
        Dict com ticker -> preço_atual
    """
    prices = {}
    
    logger.info(f"Obtendo preços atuais de {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d")
            
            if not hist.empty:
                prices[ticker] = float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Erro ao obter preço de {ticker}: {e}")
    
    logger.info(f"Preços obtidos: {len(prices)} tickers")
    
    return prices


def validate_data_quality(price_data: pd.DataFrame, 
                         min_data_points: int = 252) -> Tuple[bool, List[str]]:
    """
    Valida qualidade dos dados de preços.
    
    Args:
        price_data: DataFrame com histórico de preços
        min_data_points: Mínimo de pontos necessários
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    if price_data.empty:
        issues.append("DataFrame vazio")
        return False, issues
    
    # Verificar quantidade de dados
    tickers = price_data['Ticker'].unique()
    
    for ticker in tickers:
        ticker_data = price_data[price_data['Ticker'] == ticker]
        
        if len(ticker_data) < min_data_points:
            issues.append(f"{ticker}: apenas {len(ticker_data)} pontos (mínimo {min_data_points})")
        
        # Verificar valores nulos
        null_count = ticker_data['Close'].isnull().sum()
        if null_count > 0:
            issues.append(f"{ticker}: {null_count} valores nulos em Close")
        
        # Verificar valores zerados
        zero_count = (ticker_data['Close'] == 0).sum()
        if zero_count > 0:
            issues.append(f"{ticker}: {zero_count} valores zerados em Close")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Problemas de qualidade encontrados: {len(issues)}")
    
    return is_valid, issues
