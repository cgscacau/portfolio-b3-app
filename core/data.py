"""
Módulo de gerenciamento de dados com cache otimizado
Usa requisições HTTP diretas com cache nativo do Streamlit
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List
import streamlit as st

# Importar sistema de cache
from .cache import (
    cache_historical_data,
    cache_current_price,
    cache_dividends,
    cache_asset_info,
    cache_resource,
    cache_manager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# SESSÃO HTTP CACHEADA
# ==========================================

@cache_resource
def get_http_session() -> requests.Session:
    """
    Cria e cacheia sessão HTTP
    Reutiliza conexões para melhor performance
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    logger.info("Nova sessão HTTP criada e cacheada")
    return session


# ==========================================
# FUNÇÕES CACHEADAS (STANDALONE)
# ==========================================

@cache_historical_data
def _fetch_historical_data(
    ticker: str,
    start_timestamp: int,
    end_timestamp: int,
    interval: str = '1d'
) -> Optional[pd.DataFrame]:
    """
    Busca dados históricos com cache
    Cache: 24 horas
    """
    session = get_http_session()
    base_url = "https://query1.finance.yahoo.com"
    max_retries = 3
    timeout = 30
    
    # Normalizar ticker
    ticker_normalizado = ticker.upper().strip()
    if not ticker_normalizado.endswith('.SA'):
        ticker_normalizado = f"{ticker_normalizado}.SA"
    
    url = f"{base_url}/v8/finance/chart/{ticker_normalizado}"
    
    params = {
        'period1': start_timestamp,
        'period2': end_timestamp,
        'interval': interval,
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    
    for tentativa in range(max_retries):
        try:
            logger.info(f"    → Requisição HTTP tentativa {tentativa + 1}")
            
            response = session.get(url, params=params, timeout=timeout)
            
            logger.info(f"    ← Status: {response.status_code}")
            
            if response.status_code != 200:
                if tentativa < max_retries - 1:
                    time.sleep(2 ** tentativa)
                    continue
                logger.error(f"    ✗ Status code: {response.status_code}")
                return None
            
            data = response.json()
            
            if 'chart' not in data:
                logger.warning(f"    ⚠ Resposta sem 'chart'")
                return None
            
            if 'result' not in data['chart'] or not data['chart']['result']:
                logger.warning(f"    ⚠ Resposta sem 'result'")
                return None
            
            result = data['chart']['result'][0]
            timestamps = result.get('timestamp', [])
            
            if not timestamps:
                logger.warning(f"    ⚠ Sem timestamps")
                return None
            
            indicators = result.get('indicators', {})
            quote = indicators.get('quote', [{}])[0]
            adjclose = indicators.get('adjclose', [{}])[0]
            
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quote.get('open', []),
                'High': quote.get('high', []),
                'Low': quote.get('low', []),
                'Close': quote.get('close', []),
                'Volume': quote.get('volume', []),
                'Adj Close': adjclose.get('adjclose', quote.get('close', []))
            })
            
            df = df.set_index('Date')
            df = df.dropna(subset=['Close'])
            
            if not df.empty:
                logger.info(f"    ✓ {len(df)} registros obtidos")
                return df
            else:
                logger.warning(f"    ⚠ DataFrame vazio após limpeza")
                return None
                
        except Exception as e:
            logger.error(f"    ✗ Erro: {str(e)}")
            if tentativa < max_retries - 1:
                time.sleep(2 ** tentativa)
            continue
    
    return None


@cache_current_price
def _fetch_current_price(ticker: str) -> Optional[float]:
    """
    Busca preço atual com cache
    Cache: 5 minutos
    """
    logger.info(f"💰 Buscando preço de {ticker} (cache 5min)")
    
    # Buscar últimos 5 dias
    end = datetime.now()
    start = end - timedelta(days=5)
    
    end_ts = int(end.timestamp())
    start_ts = int(start.timestamp())
    
    df = _fetch_historical_data(ticker, start_ts, end_ts)
    
    if df is not None and not df.empty:
        preco = float(df['Close'].iloc[-1])
        logger.info(f"✓ {ticker}: R$ {preco:.2f}")
        return preco
    
    logger.error(f"✗ Não foi possível obter preço de {ticker}")
    return None


@cache_dividends
def _fetch_dividends(
    ticker: str,
    start_timestamp: int,
    end_timestamp: int
) -> pd.DataFrame:
    """
    Busca dividendos com cache
    Cache: 12 horas
    """
    session = get_http_session()
    base_url = "https://query1.finance.yahoo.com"
    timeout = 30
    
    # Normalizar ticker
    ticker_normalizado = ticker.upper().strip()
    if not ticker_normalizado.endswith('.SA'):
        ticker_normalizado = f"{ticker_normalizado}.SA"
    
    logger.info(f"💵 Buscando dividendos de {ticker_normalizado} (cache 12h)")
    
    url = f"{base_url}/v8/finance/chart/{ticker_normalizado}"
    
    params = {
        'period1': start_timestamp,
        'period2': end_timestamp,
        'interval': '1d',
        'events': 'div',
    }
    
    try:
        response = session.get(url, params=params, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                events = result.get('events', {})
                dividends = events.get('dividends', {})
                
                if dividends:
                    div_list = []
                    for timestamp, div_data in dividends.items():
                        div_list.append({
                            'data': pd.to_datetime(int(timestamp), unit='s'),
                            'valor': div_data.get('amount', 0)
                        })
                    
                    df = pd.DataFrame(div_list)
                    df = df.sort_values('data').reset_index(drop=True)
                    
                    logger.info(f"✓ {ticker}: {len(df)} dividendos")
                    return df
    
    except Exception as e:
        logger.error(f"✗ Erro ao buscar dividendos: {str(e)}")
    
    logger.warning(f"⚠ {ticker}: sem dividendos")
    return pd.DataFrame(columns=['data', 'valor'])


@cache_asset_info
def _fetch_asset_info(ticker: str) -> Dict[str, Any]:
    """
    Busca informações do ativo com cache
    Cache: 7 dias
    """
    logger.info(f"ℹ️ Buscando info de {ticker} (cache 7d)")
    
    info = {
        'ticker': ticker,
        'nome': ticker,
        'preco': None,
        'tipo': 'ACAO'
    }
    
    info['preco'] = _fetch_current_price(ticker)
    
    return info


# ==========================================
# DATA MANAGER (SEM CACHE NOS MÉTODOS)
# ==========================================

class DataManager:
    """Gerenciador de dados - delega cache para funções standalone"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.base_url = "https://query1.finance.yahoo.com"
        self.max_retries = 3
        self.timeout = 30
        logger.info("DataManager inicializado")
    
    @property
    def session(self) -> requests.Session:
        """Retorna sessão HTTP cacheada"""
        return get_http_session()
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normaliza ticker"""
        ticker = str(ticker).upper().strip()
        if not ticker.endswith('.SA'):
            ticker = f"{ticker}.SA"
        return ticker
    
    def obter_preco_atual(self, ticker: str) -> Optional[float]:
        """
        Busca preço atual
        Usa cache de 5 minutos
        """
        cache_manager.stats.registrar_request()
        return _fetch_current_price(ticker)
    
    def get_price_history(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Busca histórico de preços
        Usa cache de 24 horas por ticker
        """
        logger.info("=" * 60)
        logger.info(f"📊 BUSCA DE HISTÓRICO (Cache: {'ON' if use_cache else 'OFF'})")
        logger.info(f"Ativos: {len(tickers)}")
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        logger.info("=" * 60)
        
        if start_date >= end_date:
            logger.error("✗ Data inicial >= data final")
            return pd.DataFrame()
        
        if not tickers:
            logger.error("✗ Lista vazia")
            return pd.DataFrame()
        
        # Converter para timestamps
        start_ts = int(start_date.timestamp())
        end_ts = int((end_date + timedelta(days=1)).timestamp())
        
        all_data = {}
        sucessos = 0
        falhas = 0
        
        # Processar em lotes
        batch_size = 5
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"\n📦 Lote {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            for idx, ticker in enumerate(batch, 1):
                logger.info(f"[{i+idx}/{len(tickers)}] {ticker}")
                cache_manager.stats.registrar_request()
                
                if use_cache:
                    # Usa função cacheada
                    df = _fetch_historical_data(ticker, start_ts, end_ts)
                    if df is not None:
                        cache_manager.stats.registrar_hit()
                else:
                    # Limpa cache e busca novamente
                    st.cache_data.clear()
                    df = _fetch_historical_data(ticker, start_ts, end_ts)
                    cache_manager.stats.registrar_miss()
                
                if df is not None and not df.empty:
                    all_data[ticker] = df['Close']
                    sucessos += 1
                else:
                    falhas += 1
                    cache_manager.stats.registrar_miss()
                
                # Delay entre requisições
                time.sleep(0.2 if use_cache else 0.5)
            
            # Delay entre lotes
            if i + batch_size < len(tickers):
                logger.info("⏳ Aguardando antes do próximo lote...")
                time.sleep(1 if use_cache else 2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ RESUMO: {sucessos} sucessos, {falhas} falhas")
        logger.info(f"{'='*60}")
        
        if all_data:
            try:
                df = pd.DataFrame(all_data)
                df = df.sort_index()
                
                logger.info(f"✓ DataFrame: {len(df)} linhas x {len(df.columns)} colunas")
                
                if not df.empty:
                    logger.info(f"✓ Período: {df.index[0].date()} até {df.index[-1].date()}")
                
                return df
                
            except Exception as e:
                logger.error(f"✗ Erro ao criar DataFrame: {str(e)}")
                return pd.DataFrame()
        
        logger.error("✗ Nenhum dado obtido")
        return pd.DataFrame()
    
    def get_dividends(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Busca dividendos
        Usa cache de 12 horas
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        if end_date is None:
            end_date = datetime.now()
        
        cache_manager.stats.registrar_request()
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        df = _fetch_dividends(ticker, start_ts, end_ts)
        
        if not df.empty:
            cache_manager.stats.registrar_hit()
        else:
            cache_manager.stats.registrar_miss()
        
        return df
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Busca preços de múltiplos ativos
        Usa cache de 5 minutos por ticker
        """
        logger.info(f"💰 Buscando preços de {len(tickers)} ativos (cache 5min)")
        
        precos = {}
        for ticker in tickers:
            precos[ticker] = self.obter_preco_atual(ticker)
            time.sleep(0.2)
        
        return precos
    
    def obter_informacoes_ativo(self, ticker: str) -> Dict[str, Any]:
        """
        Busca informações do ativo
        Usa cache de 7 dias
        """
        cache_manager.stats.registrar_request()
        info = _fetch_asset_info(ticker)
        cache_manager.stats.registrar_hit()
        return info
    
    def testar_conexao(self) -> Dict[str, bool]:
        """Testa conexão"""
        logger.info("🔌 Testando conexão HTTP...")
        
        resultado = {'yahoo_finance': False}
        
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            
            df = _fetch_historical_data(
                'PETR4',
                int(start.timestamp()),
                int(end.timestamp())
            )
            
            if df is not None and not df.empty:
                resultado['yahoo_finance'] = True
                logger.info("✓ Yahoo Finance HTTP: OK")
            else:
                logger.error("✗ Yahoo Finance HTTP: sem dados")
        except Exception as e:
            logger.error(f"✗ Yahoo Finance HTTP: {str(e)}")
        
        return resultado


# ==========================================
# INSTÂNCIA GLOBAL
# ==========================================

_data_manager = DataManager()


# ==========================================
# FUNÇÕES PÚBLICAS
# ==========================================

def get_price_history(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True
) -> pd.DataFrame:
    """Busca histórico (com cache)"""
    return _data_manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Busca dividendos (com cache)"""
    return _data_manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Busca preços (com cache)"""
    return _data_manager.get_current_prices(tickers)


def obter_preco_atual(ticker: str) -> Optional[float]:
    """Busca preço (com cache)"""
    return _data_manager.obter_preco_atual(ticker)


def obter_informacoes_ativo(ticker: str) -> Dict[str, Any]:
    """Busca info (com cache)"""
    return _data_manager.obter_informacoes_ativo(ticker)


def testar_conexao() -> Dict[str, bool]:
    """Testa conexão"""
    return _data_manager.testar_conexao()


# Aliases
obter_preco = obter_preco_atual
obter_info = obter_informacoes_ativo
obter_dividendos = get_dividends
testar_apis = testar_conexao
