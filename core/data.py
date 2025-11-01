"""
core/data.py
Sistema completo de coleta, cache e limpeza de dados da B3
Inclui fallback para dados simulados quando yfinance falhar
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


# ============================================================================
# FUNÇÕES DE DADOS SIMULADOS (MOCK)
# ============================================================================

BLUE_CHIPS = [
    'PETR4.SA', 'PETR3.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 
    'BBDC3.SA', 'BBAS3.SA', 'ABEV3.SA', 'WEGE3.SA', 'RENT3.SA',
    'B3SA3.SA', 'SUZB3.SA', 'RAIL3.SA', 'ELET3.SA', 'CMIG4.SA'
]

def generate_mock_liquidity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera dados simulados de liquidez baseados em características conhecidas.
    """
    df = df.copy()
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        setor = row.get('setor', '')
        
        # Determinar perfil de liquidez
        if ticker in BLUE_CHIPS:
            # Blue chips: altíssima liquidez
            avg_volume = np.random.uniform(50e6, 300e6)
            sessions = np.random.randint(20, 23)
            is_traded = True
        
        elif setor in ['Financeiro', 'Petróleo e Gás', 'Mineração']:
            # Setores líquidos: alta liquidez
            avg_volume = np.random.uniform(5e6, 50e6)
            sessions = np.random.randint(18, 23)
            is_traded = np.random.random() > 0.1  # 90% líquidos
        
        elif setor in ['Energia Elétrica', 'Saneamento', 'Telecomunicações']:
            # Setores moderados: média liquidez
            avg_volume = np.random.uniform(1e6, 10e6)
            sessions = np.random.randint(15, 22)
            is_traded = np.random.random() > 0.2  # 80% líquidos
        
        else:
            # Outros setores: liquidez variável
            avg_volume = np.random.uniform(100e3, 5e6)
            sessions = np.random.randint(10, 22)
            is_traded = np.random.random() > 0.3  # 70% líquidos
        
        df.at[idx, 'is_traded_30d'] = is_traded
        df.at[idx, 'avg_volume_30d'] = avg_volume
        df.at[idx, 'sessions_traded_30d'] = sessions
    
    return df


def generate_mock_price_data(tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Gera série temporal simulada de preços.
    """
    dates = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    prices = {}
    
    for ticker in tickers:
        # Preço inicial aleatório
        initial_price = np.random.uniform(10, 100)
        
        # Simular retornos diários com drift positivo
        n_days = len(dates)
        returns = np.random.normal(0.0003, 0.015, n_days)  # drift=0.03%/dia, vol=1.5%/dia
        
        # Gerar série de preços
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        prices[ticker] = price_series
    
    df = pd.DataFrame(prices, index=dates)
    
    return df


def generate_mock_dividend_data(tickers: list, start: datetime, end: datetime) -> dict:
    """
    Gera histórico simulado de dividendos.
    """
    dividends_dict = {}
    
    for ticker in tickers:
        # Decidir se paga dividendos (75% pagam)
        if np.random.random() > 0.25:
            # Número de pagamentos no período (trimestral ou semestral)
            months = (end - start).days / 30
            n_payments = int(np.random.uniform(2, min(months/3, 12)))
            
            if n_payments > 0:
                # Datas aleatórias distribuídas ao longo do período
                date_range = pd.date_range(start=start, end=end, periods=n_payments)
                
                # Valores aleatórios (mais consistentes)
                base_value = np.random.uniform(0.2, 1.5)
                values = base_value * (1 + np.random.normal(0, 0.2, n_payments))
                values = np.abs(values)  # Garantir positivos
                
                dividends_dict[ticker] = pd.Series(values, index=date_range)
    
    return dividends_dict


# ============================================================================
# FUNÇÕES PRINCIPAIS DE DADOS
# ============================================================================

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
                          min_avg_volume: float = 100000,
                          show_progress: bool = True,
                          use_mock: bool = False) -> pd.DataFrame:
    """
    Filtra ativos negociados nos últimos 30 dias com liquidez mínima.
    
    Args:
        df: DataFrame com coluna 'ticker'
        min_sessions: Número mínimo de sessões com negociação
        min_avg_volume: Volume médio mínimo diário
        show_progress: Se deve mostrar barra de progresso
        use_mock: Se True, usa dados simulados
    
    Returns:
        DataFrame filtrado com colunas adicionais
    """
    if df.empty:
        return df
    
    # Modo mock
    if use_mock:
        logger.info("Usando dados simulados de liquidez")
        st.info("🎲 Usando dados simulados para teste")
        return generate_mock_liquidity_data(df)
    
    df = df.copy()
    df['is_traded_30d'] = False
    df['avg_volume_30d'] = 0.0
    df['sessions_traded_30d'] = 0
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    total = len(df)
    traded_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        ticker = row['ticker']
        
        try:
            if show_progress:
                status_text.text(f"Verificando: {ticker} ({idx+1}/{total})")
            
            # Tentar download com retry
            max_retries = 2
            data = None
            
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        ticker,
                        period="1mo",
                        progress=False,
                        show_errors=False
                    )
                    
                    if not data.empty:
                        break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(f"Falha após {max_retries} tentativas: {ticker}")
                    time.sleep(0.5)  # Pequena pausa entre tentativas
            
            if data is not None and not data.empty and 'Volume' in data.columns:
                valid_sessions = data[data['Volume'] > 0]
                
                sessions_traded = len(valid_sessions)
                avg_volume = valid_sessions['Volume'].mean() if len(valid_sessions) > 0 else 0
                
                df.at[idx, 'sessions_traded_30d'] = int(sessions_traded)
                df.at[idx, 'avg_volume_30d'] = float(avg_volume)
                
                if sessions_traded >= min_sessions and avg_volume >= min_avg_volume:
                    df.at[idx, 'is_traded_30d'] = True
                    traded_count += 1
            else:
                failed_count += 1
            
        except Exception as e:
            logger.warning(f"Erro ao verificar {ticker}: {e}")
            failed_count += 1
            continue
        
        if show_progress:
            progress_bar.progress((idx + 1) / total)
        
        # Se muitos erros consecutivos, oferecer fallback
        if failed_count > 20 and idx > 30 and traded_count == 0:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
            
            st.error(f"❌ Muitas falhas no download ({failed_count}/{idx+1}). yfinance pode estar indisponível.")
            
            use_mock_fallback = st.button(
                "🎲 Usar Dados Simulados em Vez Disso", 
                key="fallback_to_mock",
                type="primary"
            )
            
            if use_mock_fallback:
                st.info("Gerando dados simulados...")
                return generate_mock_liquidity_data(df)
            
            st.stop()
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    logger.info(f"Ativos líquidos: {traded_count}/{total}, Falhas: {failed_count}")
    
    # Se TODOS ou quase todos falharam, usar mock automaticamente
    if traded_count == 0 and failed_count > total * 0.8:
        st.warning("⚠️ yfinance não está respondendo. Usando dados simulados automaticamente.")
        time.sleep(1)
        return generate_mock_liquidity_data(df)
    
    return df


def batch_download_history(tickers: List[str], start: datetime, end: datetime,
                           interval: str = "1d", batch_size: int = 50,
                           show_progress: bool = True,
                           use_mock: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download em lotes para melhor performance.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        interval: Intervalo (1d, 1wk, 1mo)
        batch_size: Tamanho do lote
        show_progress: Se deve mostrar progresso
        use_mock: Se True, usa dados simulados
    
    Returns:
        Dicionário {ticker: DataFrame com OHLCV}
    """
    # Modo mock
    if use_mock:
        logger.info("Usando dados simulados de preços")
        mock_prices = generate_mock_price_data(tickers, start, end)
        
        # Converter para formato esperado
        result = {}
        for ticker in tickers:
            if ticker in mock_prices.columns:
                df = pd.DataFrame({
                    'Open': mock_prices[ticker],
                    'High': mock_prices[ticker] * 1.02,
                    'Low': mock_prices[ticker] * 0.98,
                    'Close': mock_prices[ticker],
                    'Adj Close': mock_prices[ticker],
                    'Volume': np.random.randint(1e6, 100e6, len(mock_prices))
                })
                result[ticker] = df
        
        return result
    
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
        
        # Pequena pausa entre lotes
        time.sleep(0.3)
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    logger.info(f"Download concluído: {len(all_data)}/{len(tickers)} ativos")
    
    return all_data


@st.cache_data(ttl=3600)
def get_price_history(tickers: List[str], start: datetime, end: datetime,
                     use_cache: bool = True, use_mock: bool = False) -> pd.DataFrame:
    """
    Obtém histórico de preços ajustados.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        use_cache: Se deve usar cache em disco
        use_mock: Se True, usa dados simulados
    
    Returns:
        DataFrame com índice datetime e colunas = tickers (preços ajustados)
    """
    if not tickers:
        logger.warning("Lista de tickers vazia")
        return pd.DataFrame()
    
    # Modo mock
    if use_mock:
        logger.info("Usando preços simulados")
        st.info("🎲 Usando dados de preços simulados")
        return generate_mock_price_data(tickers, start, end)
    
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
    
    try:
        all_data = batch_download_history(tickers, start, end, use_mock=False)
        
        if not all_data:
            st.warning("⚠️ Falha no download. Usando dados simulados.")
            return generate_mock_price_data(tickers, start, end)
        
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
            st.warning("⚠️ Nenhum dado de preço disponível. Usando dados simulados.")
            return generate_mock_price_data(tickers, start, end)
        
        prices_df = pd.DataFrame(prices_dict)
        
        # Limpar dados
        prices_df = prices_df.dropna(how='all')
        prices_df = prices_df.sort_index()
        
        # Salvar no cache
        if use_cache and not prices_df.empty:
            cache_manager.save_to_cache(cache_key, prices_df)
        
        st.success(f"✅ Histórico obtido: {len(prices_df)} dias, {len(prices_df.columns)} ativos")
        
        return prices_df
    
    except Exception as e:
        logger.error(f"Erro no download de preços: {e}")
        st.warning(f"⚠️ Erro no download: {e}. Usando dados simulados.")
        return generate_mock_price_data(tickers, start, end)


@st.cache_data(ttl=3600)
def get_volume_history(tickers: List[str], start: datetime, end: datetime,
                      use_cache: bool = True, use_mock: bool = False) -> pd.DataFrame:
    """
    Obtém histórico de volume negociado.
    """
    if not tickers:
        return pd.DataFrame()
    
    if use_mock:
        # Gerar volumes simulados
        dates = pd.date_range(start=start, end=end, freq='B')
        volumes = {}
        for ticker in tickers:
            volumes[ticker] = np.random.randint(1e6, 100e6, len(dates))
        return pd.DataFrame(volumes, index=dates)
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "volume")
    
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
    
    if use_cache and not volume_df.empty:
        cache_manager.save_to_cache(cache_key, volume_df)
    
    return volume_df


@st.cache_data(ttl=3600)
def get_dividends(tickers: List[str], start: datetime, end: datetime,
                 use_cache: bool = True, use_mock: bool = False) -> Dict[str, pd.Series]:
    """
    Obtém histórico de dividendos pagos.
    
    Args:
        tickers: Lista de tickers
        start: Data inicial
        end: Data final
        use_cache: Se deve usar cache
        use_mock: Se True, usa dados simulados
    
    Returns:
        Dicionário {ticker: Series de dividendos com índice datetime}
    """
    if not tickers:
        return {}
    
    # Modo mock
    if use_mock:
        logger.info("Usando dividendos simulados")
        st.info("🎲 Usando dados de dividendos simulados")
        return generate_mock_dividend_data(tickers, start, end)
    
    cache_manager = DataCache()
    cache_key = cache_manager.get_cache_key(tickers, start, end, "dividends")
    
    # Tentar carregar do cache
    if use_cache:
        cached_data = cache_manager.load_from_cache(cache_key, max_age_hours=12)
        if cached_data is not None:
            st.success(f"✅ Dados de dividendos carregados do cache")
            return {col: cached_data[col].dropna() for col in cached_data.columns}
    
    st.info(f"📥 Baixando histórico de dividendos para {len(tickers)} ativos...")
    
    try:
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
        
        # Se falhou muito, usar mock
        if success_count == 0 and total > 10:
            st.warning("⚠️ Falha ao obter dividendos. Usando dados simulados.")
            return generate_mock_dividend_data(tickers, start, end)
        
        # Consolidar em DataFrame para cache
        if dividends_dict:
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
    
    except Exception as e:
        logger.error(f"Erro no download de dividendos: {e}")
        st.warning(f"⚠️ Erro: {e}. Usando dados simulados.")
        return generate_mock_dividend_data(tickers, start, end)


@st.cache_data(ttl=1800)
def get_current_prices(tickers: List[str], use_mock: bool = False) -> Dict[str, float]:
    """
    Obtém preços atuais (último fechamento disponível).
    
    Args:
        tickers: Lista de tickers
        use_mock: Se True, usa dados simulados
    
    Returns:
        Dicionário {ticker: preço}
    """
    if not tickers:
        return {}
    
    if use_mock:
        # Preços simulados
        return {ticker: np.random.uniform(10, 100) for ticker in tickers}
    
    prices = {}
    
    # Usar período curto para pegar último preço
    end = datetime.now()
    start = end - timedelta(days=7)
    
    st.info("📥 Obtendo preços atuais...")
    
    all_data = batch_download_history(tickers, start, end, show_progress=False)
    
    for ticker, data in all_data.items():
        if not data.empty:
            if 'Adj Close' in data.columns:
                last_price = data['Adj Close'].iloc[-1]
            elif 'Close' in data.columns:
                last_price = data['Close'].iloc[-1]
            else:
                continue
            
            if not np.isnan(last_price):
                prices[ticker] = float(last_price)
    
    # Se falhou, usar mock
    if not prices and use_mock is False:
        st.warning("⚠️ Falha ao obter preços. Usando valores simulados.")
        return {ticker: np.random.uniform(10, 100) for ticker in tickers}
    
    st.success(f"✅ Preços obtidos para {len(prices)} ativos")
    
    return prices


def validate_data_quality(prices_df: pd.DataFrame, 
                         min_data_points: int = 252,
                         max_missing_pct: float = 0.1) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Valida qualidade dos dados e remove ativos com dados insuficientes.
    """
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
            removal_reasons[col] = f"Dados insuficientes: {valid_points} pontos (mín: {min_data_points})"
            logger.warning(f"Removido {col}: apenas {valid_points} pontos válidos")
            continue
        
        if missing_pct > max_missing_pct:
            removed_tickers.append(col)
            removal_reasons[col] = f"Muitos dados faltantes: {missing_pct*100:.1f}%"
            logger.warning(f"Removido {col}: {missing_pct*100:.1f}% de dados faltantes")
            continue
    
    clean_df = prices_df.drop(columns=removed_tickers, errors='ignore')
    
    if clean_df.empty:
        st.error("❌ Todos os ativos foram removidos por dados insuficientes")
        return clean_df, removed_tickers, removal_reasons
    
    # Forward fill para preencher gaps pequenos
    clean_df = clean_df.fillna(method='ffill', limit=5)
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
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
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


def calculate_returns(prices_df: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calcula retornos diários.
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    if method == 'log':
        returns = np.log(prices_df / prices_df.shift(1))
    else:
        returns = prices_df.pct_change()
    
    returns = returns.dropna()
    
    return returns


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
