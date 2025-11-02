"""
Módulo de gerenciamento de dados usando requisições HTTP diretas
Contorna problemas do yfinance com User-Agent
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Gerenciador de dados com requisições HTTP diretas"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.base_url = "https://query1.finance.yahoo.com"
        self.max_retries = 3
        self.timeout = 30
        logger.info("DataManager inicializado com requisições HTTP diretas")
    
    # ==========================================
    # NORMALIZAÇÃO
    # ==========================================
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normaliza ticker"""
        ticker = str(ticker).upper().strip()
        if not ticker.endswith('.SA'):
            ticker = f"{ticker}.SA"
        return ticker
    
    # ==========================================
    # REQUISIÇÃO DIRETA AO YAHOO
    # ==========================================
    
    def _fetch_yahoo_data(
        self,
        ticker: str,
        start_timestamp: int,
        end_timestamp: int,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Faz requisição HTTP direta ao Yahoo Finance
        
        Args:
            ticker: Código do ativo
            start_timestamp: Timestamp de início
            end_timestamp: Timestamp de fim
            interval: Intervalo (1d, 1wk, 1mo)
            
        Returns:
            DataFrame com dados ou None
        """
        ticker_normalizado = self._normalize_ticker(ticker)
        
        url = f"{self.base_url}/v8/finance/chart/{ticker_normalizado}"
        
        params = {
            'period1': start_timestamp,
            'period2': end_timestamp,
            'interval': interval,
            'events': 'history',
            'includeAdjustedClose': 'true'
        }
        
        for tentativa in range(self.max_retries):
            try:
                logger.info(f"  Tentativa {tentativa + 1}: requisição HTTP para {ticker_normalizado}")
                
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                
                logger.info(f"    Status: {response.status_code}")
                
                if response.status_code != 200:
                    if tentativa < self.max_retries - 1:
                        time.sleep(2 ** tentativa)
                        continue
                    logger.error(f"    ✗ Status code: {response.status_code}")
                    return None
                
                data = response.json()
                
                # Verificar estrutura da resposta
                if 'chart' not in data:
                    logger.warning(f"    ⚠ Resposta sem 'chart'")
                    return None
                
                if 'result' not in data['chart'] or not data['chart']['result']:
                    logger.warning(f"    ⚠ Resposta sem 'result'")
                    return None
                
                result = data['chart']['result'][0]
                
                # Extrair timestamps
                timestamps = result.get('timestamp', [])
                if not timestamps:
                    logger.warning(f"    ⚠ Sem timestamps")
                    return None
                
                # Extrair indicadores
                indicators = result.get('indicators', {})
                quote = indicators.get('quote', [{}])[0]
                adjclose = indicators.get('adjclose', [{}])[0]
                
                # Criar DataFrame
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
                if tentativa < self.max_retries - 1:
                    time.sleep(2 ** tentativa)
                continue
        
        return None
    
    # ==========================================
    # BUSCA DE PREÇO ATUAL
    # ==========================================
    
    def obter_preco_atual(self, ticker: str) -> Optional[float]:
        """
        Busca preço atual
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço atual ou None
        """
        logger.info(f"Buscando preço de {ticker}")
        
        # Buscar últimos 5 dias
        end = datetime.now()
        start = end - timedelta(days=5)
        
        end_ts = int(end.timestamp())
        start_ts = int(start.timestamp())
        
        df = self._fetch_yahoo_data(ticker, start_ts, end_ts)
        
        if df is not None and not df.empty:
            preco = float(df['Close'].iloc[-1])
            logger.info(f"✓ {ticker}: R$ {preco:.2f}")
            return preco
        
        logger.error(f"✗ Não foi possível obter preço de {ticker}")
        return None
    
    # ==========================================
    # BUSCA DE HISTÓRICO
    # ==========================================
    
    def get_price_history(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Busca histórico de preços
        
        Args:
            tickers: Lista de códigos
            start_date: Data inicial
            end_date: Data final
            use_cache: Ignorado
            
        Returns:
            DataFrame com histórico
        """
        logger.info("=" * 60)
        logger.info(f"BUSCA DE HISTÓRICO (HTTP DIRETO)")
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
            logger.info(f"\nLote {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            for idx, ticker in enumerate(batch, 1):
                logger.info(f"[{i+idx}/{len(tickers)}] {ticker}")
                
                df = self._fetch_yahoo_data(ticker, start_ts, end_ts)
                
                if df is not None and not df.empty:
                    all_data[ticker] = df['Close']
                    sucessos += 1
                else:
                    falhas += 1
                
                # Delay entre requisições
                time.sleep(0.5)
            
            # Delay entre lotes
            if i + batch_size < len(tickers):
                logger.info("Aguardando antes do próximo lote...")
                time.sleep(2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMO: {sucessos} sucessos, {falhas} falhas")
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
    
    # ==========================================
    # BUSCA DE DIVIDENDOS
    # ==========================================
    
    def get_dividends(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Busca dividendos via requisição HTTP
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com dividendos
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        if end_date is None:
            end_date = datetime.now()
        
        ticker_normalizado = self._normalize_ticker(ticker)
        logger.info(f"Buscando dividendos de {ticker_normalizado}")
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        url = f"{self.base_url}/v8/finance/chart/{ticker_normalizado}"
        
        params = {
            'period1': start_ts,
            'period2': end_ts,
            'interval': '1d',
            'events': 'div',
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
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
    
    # ==========================================
    # BUSCA EM LOTE
    # ==========================================
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Busca preços de múltiplos ativos"""
        logger.info(f"Buscando preços de {len(tickers)} ativos")
        
        precos = {}
        for ticker in tickers:
            precos[ticker] = self.obter_preco_atual(ticker)
            time.sleep(0.5)
        
        return precos
    
    # ==========================================
    # INFORMAÇÕES
    # ==========================================
    
    def obter_informacoes_ativo(self, ticker: str) -> Dict[str, Any]:
        """Busca informações do ativo"""
        info = {
            'ticker': ticker,
            'nome': ticker,
            'preco': None,
            'tipo': 'ACAO'
        }
        
        info['preco'] = self.obter_preco_atual(ticker)
        
        return info
    
    # ==========================================
    # TESTE
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """Testa conexão"""
        logger.info("Testando conexão HTTP...")
        
        resultado = {'yahoo_finance': False}
        
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            
            df = self._fetch_yahoo_data('PETR4', int(start.timestamp()), int(end.timestamp()))
            
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
    """Busca histórico"""
    return _data_manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Busca dividendos"""
    return _data_manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Busca preços"""
    return _data_manager.get_current_prices(tickers)


def obter_preco_atual(ticker: str) -> Optional[float]:
    """Busca preço"""
    return _data_manager.obter_preco_atual(ticker)


def obter_informacoes_ativo(ticker: str) -> Dict[str, Any]:
    """Busca info"""
    return _data_manager.obter_informacoes_ativo(ticker)


def testar_conexao() -> Dict[str, bool]:
    """Testa conexão"""
    return _data_manager.testar_conexao()


# Aliases
obter_preco = obter_preco_atual
obter_info = obter_informacoes_ativo
obter_dividendos = get_dividends
testar_apis = testar_conexao
