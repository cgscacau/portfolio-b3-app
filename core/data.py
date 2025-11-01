"""
core/data.py
Sistema completo de coleta, cache e limpeza de dados da B3
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
from . import data_mock  # Importar módulo de mock


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
        """
        Gera chave única para combinação de parâmetros.
        
        Args:
            tickers: Lista de tickers
            start_date: Data inicial
            end_date: Data final
            data_type: Tipo de dado (prices, dividends, volume)
        
        Returns:
            Hash MD5 único
        """
        tickers_sorted = sorted(tickers)
        key_string = f"{data_type}_{tickers_sorted}_{start_date.date()}_{end_date.date()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Carrega dados do cache se ainda válidos.
        
        Args:
            cache_key: Chave do cache
            max_age_hours: Idade máxima em horas
        
        Returns:
            DataFrame ou None se não encontrado/expirado
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Verifica idade do arquivo
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
        """
        Salva dados no cache.
        
        Args:
            cache_key: Chave do cache
            data: DataFrame a ser salvo
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cache salvo: {cache_key}")
        except Exception as e:
            logger.error(f"Erro ao salvar cache {cache_key}: {e}")
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """
        Limpa arquivos de cache.
        
        Args:
            older_than_hours: Se especificado, remove apenas caches mais antigos
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_hours:
                file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if file_age_hours < older_than_hours:
                    continue
            
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Erro ao remover {cache_file}: {e}")
        
        logger.info(f"Cache limpo: {count} arquivos removidos")
        return count


@st.cache_data(ttl=86400)  # Cache de 24 horas
def load_ticker_universe() -> pd.DataFrame:
    """
    Carrega universo de tickers B3 com metadados.
    
    Returns:
        DataFrame com colunas: ticker, nome, setor, subsetor, segmento_listagem, tipo
    """
    try:
        if not B3_UNIVERSE_FILE.exists():
            logger.error(f"Arquivo não encontrado: {B3_UNIVERSE_FILE}")
            st.error(f"❌ Arquivo de universo não encontrado: {B3_UNIVERSE_FILE}")
            return pd.DataFrame()
        
        df = pd.read_csv(B3_UNIVERSE_FILE)
        
        # Validar colunas esperadas
        expected_cols = ['ticker', 'nome', 'setor', 'subsetor', 'segmento_listagem', 'tipo']
        missing_cols = set(expected_cols) - set(df.columns)
        
        if missing_cols:
            logger.error(f"Colunas faltando no arquivo: {missing_cols}")
            st.error(f"❌ Arquivo de universo com formato inválido")
            return pd.DataFrame()
        
        logger.info(f"Universo carregado: {len(df)} tickers")
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar universo: {e}")
        st.error(f"❌ Erro ao carregar universo de tickers: {e}")
        return pd.DataFrame()


def filter_traded_last_30d(df: pd.DataFrame, min_sessions: int = 5, 
                          min_avg_volume: float = 100000,  # CORRIGIDO: valor mais realista
                          show_progress: bool = True) -> pd.DataFrame:
    """
    Filtra ativos negociados nos últimos 30 dias com liquidez mínima.
    
    Args:
        df: DataFrame com coluna 'ticker'
        min_sessions: Número mínimo de sessões com negociação (padrão: 5)
        min_avg_volume: Volume médio mínimo diário em ações (padrão: 100.000)
                       Valores típicos:
                       - 100.000 = baixa liquidez
                       - 1.000.000 = média liquidez  
                       - 10.000.000 = alta liquidez
                       - Blue chips: > 50.000.000
        show_progress: Se deve mostrar barra de progresso
    
    Returns:
        DataFrame filtrado com colunas adicionais:
        - is_traded_30d: bool
        - avg_volume_30d: float (volume médio em ações)
        - sessions_traded_30d: int
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
    failed_tickers = []
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        
        try:
            if show_progress:
                status_text.text(f"Verificando liquidez: {ticker} ({idx+1}/{total})")
            
            # Download dos últimos 35 dias para garantir 30 dias úteis
            data = yf.download(
                ticker, 
                period="35d", 
                progress=False, 
                show_errors=False,
                threads=False
            )
            
            if not data.empty and 'Volume' in data.columns:
                # Filtrar apenas sessões com volume > 0
                valid_sessions = data[data['Volume'] > 0]
                
                sessions_traded = len(valid_sessions)
                avg_volume = valid_sessions['Volume'].mean() if len(valid_sessions) > 0 else 0
                
                df.at[idx, 'sessions_traded_30d'] = int(sessions_traded)
                df.at[idx, 'avg_volume_30d'] = float(avg_volume)
                
                if sessions_traded >= min_sessions and avg_volume >= min_avg_volume:
                    df.at[idx, 'is_traded_30d'] = True
                    traded_count += 1
            else:
                failed_tickers.append(ticker)
            
        except Exception as e:
            logger.warning(f"Erro ao verificar {ticker}: {e}")
            failed_tickers.append(ticker)
            continue
        
        if show_progress:
            progress_bar.progress((idx + 1) / total)
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    logger.info(f"Ativos líquidos (30d): {traded_count}/{total}")
    
    if failed_tickers:
        logger.warning(f"Falhas ao verificar {len(failed_tickers)} tickers")
    
    return df


def batch_download_history(tickers: List[str], start: datetime, end: datetime,
                           interval: str = "1d", batch_size: int = 50,
                           show_progress: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Download em lotes para melhor performance.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        interval: Intervalo (1d, 1wk, 1mo)
        batch_size: Tamanho do lote
        show_progress: Se deve mostrar progresso
    
    Returns:
        Dicionário {ticker: DataFrame com OHLCV}
    """
    all_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        if show_progress:
            status_text.text(f"Baixando lote {batch_num}/{total_batches} ({len(batch)} ativos)...")
        
        try:
            # Download do lote
            if len(batch) == 1:
                # Caso especial: único ticker
                data = yf.download(
                    batch[0],
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    show_errors=False,
                    threads=False
                )
                if not data.empty:
                    all_data[batch[0]] = data
            else:
                # Múltiplos tickers
                ticker_string = " ".join(batch)
                data = yf.download(
                    ticker_string,
                    start=start,
                    end=end,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    show_errors=False,
                    threads=False
                )
                
                # Processar dados por ticker
                for ticker in batch:
                    try:
                        if ticker in data.columns.levels[0]:
                            ticker_data = data[ticker]
                            if not ticker_data.empty:
                                all_data[ticker] = ticker_data
                    except (KeyError, AttributeError):
                        # Ticker não tem dados ou estrutura diferente
                        continue
                    except Exception as e:
                        logger.warning(f"Erro ao processar {ticker}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Erro no lote {batch_num}: {e}")
            # Fallback: tentar individual
            for ticker in batch:
                try:
                    data = yf.download(
                        ticker, 
                        start=start, 
                        end=end, 
                        interval=interval, 
                        progress=False, 
                        show_errors=False,
                        threads=False
                    )
                    if not data.empty:
                        all_data[ticker] = data
                except Exception as e2:
                    logger.warning(f"Falha individual em {ticker}: {e2}")
                    continue
        
        if show_progress:
            progress_bar.progress(min((i + batch_size) / len(tickers), 1.0))
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    logger.info(f"Download concluído: {len(all_data)}/{len(tickers)} ativos")
    
    return all_data


@st.cache_data(ttl=3600)
def get_price_history(tickers: List[str], start: datetime, end: datetime,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Obtém histórico de preços ajustados.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        use_cache: Se deve usar cache em disco
    
    Returns:
        DataFrame com índice datetime e colunas = tickers (preços ajustados)
    """
    if not tickers:
        logger.warning("Lista de tickers vazia")
        return pd.DataFrame()
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "prices")
    
    # Tentar carregar do cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key)
        if cached_data is not None:
            st.success(f"✅ Dados de preços carregados do cache ({len(cached_data)} dias)")
            return cached_data
    
    # Download de dados
    st.info(f"📥 Baixando histórico de preços para {len(tickers)} ativos...")
    
    all_data = batch_download_history(tickers, start, end)
    
    if not all_data:
        st.error("❌ Nenhum dado disponível para os tickers selecionados")
        return pd.DataFrame()
    
    # Consolidar em DataFrame único (preços ajustados)
    prices_dict = {}
    
    for ticker, data in all_data.items():
        if not data.empty:
            # Tentar Adj Close, senão Close
            if 'Adj Close' in data.columns:
                prices_dict[ticker] = data['Adj Close']
            elif 'Close' in data.columns:
                prices_dict[ticker] = data['Close']
                logger.warning(f"{ticker}: usando Close (Adj Close não disponível)")
    
    if not prices_dict:
        st.warning("⚠️ Nenhum dado de preço disponível")
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(prices_dict)
    
    # Limpar dados
    prices_df = prices_df.dropna(how='all')  # Remove dias sem nenhum dado
    
    # Ordenar por data
    prices_df = prices_df.sort_index()
    
    # Salvar no cache
    if use_cache and not prices_df.empty:
        cache_manager.save_to_cache(cache_key, prices_df)
    
    st.success(f"✅ Histórico obtido: {len(prices_df)} dias, {len(prices_df.columns)} ativos")
    
    return prices_df


@st.cache_data(ttl=3600)
def get_volume_history(tickers: List[str], start: datetime, end: datetime,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Obtém histórico de volume negociado.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        use_cache: Se deve usar cache
    
    Returns:
        DataFrame com índice datetime e colunas = tickers (volume)
    """
    if not tickers:
        return pd.DataFrame()
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "volume")
    
    # Tentar carregar do cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
    
    st.info(f"📥 Baixando histórico de volume...")
    
    all_data = batch_download_history(tickers, start, end)
    
    volume_dict = {}
    
    for ticker, data in all_data.items():
        if not data.empty and 'Volume' in data.columns:
            volume_dict[ticker] = data['Volume']
    
    if not volume_dict:
        return pd.DataFrame()
    
    volume_df = pd.DataFrame(volume_dict)
    volume_df = volume_df.dropna(how='all')
    volume_df = volume_df.sort_index()
    
    # Salvar no cache
    if use_cache and not volume_df.empty:
        cache_manager.save_to_cache(cache_key, volume_df)
    
    return volume_df


@st.cache_data(ttl=3600)
def get_dividends(tickers: List[str], start: datetime, end: datetime,
                 use_cache: bool = True) -> Dict[str, pd.Series]:
    """
    Obtém histórico de dividendos pagos.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        use_cache: Se deve usar cache
    
    Returns:
        Dicionário {ticker: Series de dividendos com índice datetime}
    """
    if not tickers:
        return {}
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "dividends")
    
    # Tentar carregar do cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key, max_age_hours=12)
        if cached_data is not None:
            st.success(f"✅ Dados de dividendos carregados do cache")
            # Converter DataFrame de volta para dict de Series
            return {col: cached_data[col].dropna() for col in cached_data.columns}
    
    st.info(f"📥 Baixando histórico de dividendos para {len(tickers)} ativos...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dividends_dict = {}
    total = len(tickers)
    success_count = 0
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Obtendo dividendos: {ticker} ({idx+1}/{total})")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            divs = ticker_obj.dividends
            
            if not divs.empty:
                # Filtrar por período
                divs = divs[(divs.index >= start) & (divs.index <= end)]
                
                if not divs.empty:
                    dividends_dict[ticker] = divs
                    success_count += 1
        
        except Exception as e:
            logger.warning(f"Erro ao obter dividendos de {ticker}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    # Consolidar em DataFrame para cache
    if dividends_dict:
        # Criar DataFrame alinhado por data
        all_dates = pd.DatetimeIndex([])
        for series in dividends_dict.values():
            all_dates = all_dates.union(series.index)
        
        divs_df = pd.DataFrame(index=all_dates.sort_values())
        for ticker, series in dividends_dict.items():
            divs_df[ticker] = series
        
        if use_cache:
            cache_manager.save_to_cache(cache_key, divs_df)
        
        st.success(f"✅ Dividendos obtidos: {success_count}/{total} ativos com pagamentos")
    else:
        st.warning("⚠️ Nenhum dividendo encontrado no período selecionado")
    
    return dividends_dict


@st.cache_data(ttl=1800)  # Cache de 30 minutos
def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Obtém preços atuais (último fechamento disponível).
    
    Args:
        tickers: Lista de tickers
    
    Returns:
        Dicionário {ticker: preço}
    """
    if not tickers:
        return {}
    
    prices = {}
    
    # Usar período curto para pegar último preço
    end = datetime.now()
    start = end - timedelta(days=7)
    
    st.info("📥 Obtendo preços atuais...")
    
    all_data = batch_download_history(tickers, start, end, show_progress=False)
    
    for ticker, data in all_data.items():
        if not data.empty:
            # Tentar Adj Close, senão Close
            if 'Adj Close' in data.columns:
                last_price = data['Adj Close'].iloc[-1]
            elif 'Close' in data.columns:
                last_price = data['Close'].iloc[-1]
            else:
                continue
            
            if not np.isnan(last_price):
                prices[ticker] = float(last_price)
    
    st.success(f"✅ Preços obtidos para {len(prices)} ativos")
    
    return prices


def validate_data_quality(prices_df: pd.DataFrame, 
                         min_data_points: int = 252,
                         max_missing_pct: float = 0.1) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Valida qualidade dos dados e remove ativos com dados insuficientes.
    
    Args:
        prices_df: DataFrame de preços
        min_data_points: Número mínimo de pontos de dados
        max_missing_pct: Percentual máximo de dados faltantes permitido
    
    Returns:
        Tuple (DataFrame limpo, lista de tickers removidos, dict de razões)
    """
    if prices_df.empty:
        return prices_df, [], {}
    
    removed_tickers = []
    removal_reasons = {}
    
    total_days = len(prices_df)
    
    for col in prices_df.columns:
        valid_points = prices_df[col].notna().sum()
        missing_pct = 1 - (valid_points / total_days)
        
        # Verificar número mínimo de pontos
        if valid_points < min_data_points:
            removed_tickers.append(col)
            removal_reasons[col] = f"Dados insuficientes: {valid_points} pontos (mín: {min_data_points})"
            logger.warning(f"Removido {col}: apenas {valid_points} pontos válidos")
            continue
        
        # Verificar percentual de dados faltantes
        if missing_pct > max_missing_pct:
            removed_tickers.append(col)
            removal_reasons[col] = f"Muitos dados faltantes: {missing_pct*100:.1f}% (máx: {max_missing_pct*100:.1f}%)"
            logger.warning(f"Removido {col}: {missing_pct*100:.1f}% de dados faltantes")
            continue
    
    # Remover colunas com dados insuficientes
    clean_df = prices_df.drop(columns=removed_tickers, errors='ignore')
    
    if clean_df.empty:
        st.error("❌ Todos os ativos foram removidos por dados insuficientes")
        return clean_df, removed_tickers, removal_reasons
    
    # Forward fill para preencher gaps pequenos (máx 5 dias consecutivos)
    clean_df = clean_df.fillna(method='ffill', limit=5)
    
    # Remover linhas ainda com NaN (início da série)
    clean_df = clean_df.dropna(how='any')
    
    if removed_tickers:
        st.warning(f"⚠️ {len(removed_tickers)} ativos removidos por qualidade de dados insuficiente")
        
        with st.expander("Ver detalhes dos ativos removidos"):
            for ticker, reason in removal_reasons.items():
                st.text(f"• {ticker}: {reason}")
    
    return clean_df, removed_tickers, removal_reasons


@st.cache_data(ttl=3600)
def get_ticker_info(ticker: str) -> Dict:
    """
    Obtém informações detalhadas de um ticker.
    
    Args:
        ticker: Ticker do ativo
    
    Returns:
        Dicionário com informações do ativo
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Extrair campos relevantes
        relevant_info = {
            'shortName': info.get('shortName', ticker),
            'longName': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'marketCap': info.get('marketCap', 0),
            'currency': info.get('currency', 'BRL'),
            'exchange': info.get('exchange', 'SAO'),
            'quoteType': info.get('quoteType', ''),
            'dividendYield': info.get('dividendYield', 0),
            'trailingPE': info.get('trailingPE', None),
            'forwardPE': info.get('forwardPE', None),
            'beta': info.get('beta', None),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', None),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', None),
        }
        
        return relevant_info
    
    except Exception as e:
        logger.error(f"Erro ao obter info de {ticker}: {e}")
        return {'shortName': ticker, 'error': str(e)}


def get_multiple_ticker_info(tickers: List[str], show_progress: bool = True) -> pd.DataFrame:
    """
    Obtém informações de múltiplos tickers.
    
    Args:
        tickers: Lista de tickers
        show_progress: Se deve mostrar progresso
    
    Returns:
        DataFrame com informações dos ativos
    """
    if not tickers:
        return pd.DataFrame()
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    info_list = []
    
    for idx, ticker in enumerate(tickers):
        if show_progress:
            status_text.text(f"Obtendo informações: {ticker} ({idx+1}/{len(tickers)})")
        
        info = get_ticker_info(ticker)
        info['ticker'] = ticker
        info_list.append(info)
        
        if show_progress:
            progress_bar.progress((idx + 1) / len(tickers))
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    df = pd.DataFrame(info_list)
    
    return df


def calculate_returns(prices_df: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calcula retornos diários.
    
    Args:
        prices_df: DataFrame de preços
        method: 'simple' ou 'log'
    
    Returns:
        DataFrame de retornos
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    if method == 'log':
        returns = np.log(prices_df / prices_df.shift(1))
    else:  # simple
        returns = prices_df.pct_change()
    
    returns = returns.dropna()
    
    return returns


def resample_prices(prices_df: pd.DataFrame, frequency: str = 'W') -> pd.DataFrame:
    """
    Reamostra preços para frequência diferente.
    
    Args:
        prices_df: DataFrame de preços diários
        frequency: 'W' (semanal), 'M' (mensal), 'Q' (trimestral), 'Y' (anual)
    
    Returns:
        DataFrame reamostrado
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    # Usar último preço do período
    resampled = prices_df.resample(frequency).last()
    resampled = resampled.dropna(how='all')
    
    return resampled


def get_data_summary(prices_df: pd.DataFrame, dividends_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Cria resumo estatístico dos dados coletados.
    
    Args:
        prices_df: DataFrame de preços
        dividends_dict: Dicionário de dividendos
    
    Returns:
        DataFrame com resumo por ativo
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker].dropna()
        
        summary = {
            'ticker': ticker,
            'data_points': len(prices),
            'first_date': prices.index[0].strftime('%Y-%m-%d'),
            'last_date': prices.index[-1].strftime('%Y-%m-%d'),
            'first_price': prices.iloc[0],
            'last_price': prices.iloc[-1],
            'min_price': prices.min(),
            'max_price': prices.max(),
            'avg_price': prices.mean(),
            'price_std': prices.std(),
            'has_dividends': ticker in dividends_dict,
            'num_dividends': len(dividends_dict.get(ticker, [])),
        }
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df


# Funções auxiliares para debugging e manutenção

def check_data_integrity(prices_df: pd.DataFrame) -> Dict[str, any]:
    """
    Verifica integridade dos dados.
    
    Returns:
        Dicionário com estatísticas de integridade
    """
    if prices_df.empty:
        return {'status': 'empty'}
    
    integrity = {
        'total_tickers': len(prices_df.columns),
        'total_days': len(prices_df),
        'missing_values': prices_df.isna().sum().sum(),
        'missing_pct': (prices_df.isna().sum().sum() / prices_df.size) * 100,
        'tickers_with_missing': (prices_df.isna().any()).sum(),
        'date_range': f"{prices_df.index[0].date()} to {prices_df.index[-1].date()}",
        'duplicated_dates': prices_df.index.duplicated().sum(),
    }
    
    return integrity


def export_data_to_csv(prices_df: pd.DataFrame, filename: str = "prices_export.csv"):
    """
    Exporta dados para CSV.
    
    Args:
        prices_df: DataFrame a exportar
        filename: Nome do arquivo
    """
    try:
        prices_df.to_csv(filename)
        logger.info(f"Dados exportados para {filename}")
        return True
    except Exception as e:
        logger.error(f"Erro ao exportar dados: {e}")
        return False


# Inicialização e verificação do módulo
def verify_module():
    """Verifica se o módulo está configurado corretamente."""
    checks = {
        'cache_dir_exists': CACHE_DIR.exists(),
        'assets_dir_exists': ASSETS_DIR.exists(),
        'universe_file_exists': B3_UNIVERSE_FILE.exists(),
    }
    
    all_ok = all(checks.values())
    
    if not all_ok:
        logger.warning(f"Verificação do módulo data.py: {checks}")
    
    return all_ok


# Executar verificação ao importar
if __name__ != "__main__":
    verify_module()
