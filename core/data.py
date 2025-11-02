"""
Módulo principal de gerenciamento de dados
Responsável por buscar cotações, dividendos e informações de ativos
"""

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List
import streamlit as st

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# CLASSE PRINCIPAL
# ==========================================

class DataManager:
    """Gerenciador de dados de ativos financeiros"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.brapi_url = "https://brapi.dev/api"
        self.max_retries = 3
        self.timeout = 15
        
    # ==========================================
    # NORMALIZAÇÃO DE TICKERS
    # ==========================================
    
    def _add_sa(self, ticker: str) -> str:
        """Adiciona .SA ao ticker se necessário"""
        ticker = str(ticker).upper().strip()
        return ticker if ticker.endswith('.SA') else f"{ticker}.SA"
    
    def _remove_sa(self, ticker: str) -> str:
        """Remove .SA do ticker"""
        return str(ticker).upper().strip().replace('.SA', '')
    
    # ==========================================
    # BUSCA DE PREÇO ATUAL
    # ==========================================
    
    def obter_preco_atual(self, ticker: str) -> Optional[float]:
        """
        Busca preço atual do ativo
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço atual ou None
        """
        # Tentar Yahoo Finance
        preco = self._preco_yahoo(ticker)
        if preco:
            return preco
        
        # Tentar BRAPI
        preco = self._preco_brapi(ticker)
        if preco:
            return preco
        
        logger.error(f"Não foi possível obter preço para {ticker}")
        return None
    
    def _preco_yahoo(self, ticker: str) -> Optional[float]:
        """Busca preço no Yahoo Finance"""
        try:
            yahoo_ticker = self._add_sa(ticker)
            stock = yf.Ticker(yahoo_ticker)
            
            # Tentar pelo histórico recente (mais confiável)
            hist = stock.history(period='5d')
            if not hist.empty:
                preco = float(hist['Close'].iloc[-1])
                if preco > 0:
                    logger.info(f"✓ {ticker}: R$ {preco:.2f} (Yahoo)")
                    return preco
            
        except Exception as e:
            logger.debug(f"Yahoo falhou para {ticker}: {e}")
        
        return None
    
    def _preco_brapi(self, ticker: str) -> Optional[float]:
        """Busca preço na BRAPI"""
        try:
            ticker_limpo = self._remove_sa(ticker)
            url = f"{self.brapi_url}/quote/{ticker_limpo}"
            
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    preco = data['results'][0].get('regularMarketPrice')
                    if preco and preco > 0:
                        logger.info(f"✓ {ticker}: R$ {preco:.2f} (BRAPI)")
                        return float(preco)
            
        except Exception as e:
            logger.debug(f"BRAPI falhou para {ticker}: {e}")
        
        return None
    
    # ==========================================
    # BUSCA DE HISTÓRICO DE PREÇOS
    # ==========================================
    
    def get_price_history(
        self, 
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Busca histórico de preços para múltiplos ativos
        
        Args:
            tickers: Lista de códigos de ativos
            start_date: Data inicial
            end_date: Data final
            use_cache: Usar cache (padrão: True)
            
        Returns:
            DataFrame com histórico (índice: data, colunas: tickers)
        """
        logger.info(f"Buscando histórico de {len(tickers)} ativos")
        logger.info(f"Período: {start_date.date()} a {end_date.date()}")
        
        # Validar datas
        if start_date >= end_date:
            logger.error("Data inicial deve ser anterior à final")
            return pd.DataFrame()
        
        # Processar ativos
        result = {}
        total = len(tickers)
        
        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"Processando {ticker} ({idx}/{total})")
            
            series = self._buscar_historico_ativo(ticker, start_date, end_date)
            
            if series is not None and not series.empty:
                result[ticker] = series
                logger.info(f"  ✓ {len(series)} registros")
            else:
                logger.warning(f"  ✗ Sem dados")
            
            # Delay para evitar rate limit
            if idx < total:
                time.sleep(0.3)
        
        # Criar DataFrame
        if result:
            df = pd.DataFrame(result)
            logger.info(f"✓ DataFrame: {len(df)} linhas x {len(df.columns)} colunas")
            return df
        
        logger.error("✗ Nenhum dado obtido")
        return pd.DataFrame()
    
    def _buscar_historico_ativo(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.Series]:
        """
        Busca histórico de um único ativo
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            Series com preços ou None
        """
        yahoo_ticker = self._add_sa(ticker)
        
        for tentativa in range(self.max_retries):
            try:
                # Método 1: usando Ticker
                stock = yf.Ticker(yahoo_ticker)
                hist = stock.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    auto_adjust=True
                )
                
                if not hist.empty and 'Close' in hist.columns:
                    return hist['Close']
                
                # Método 2: usando download
                df = yf.download(
                    yahoo_ticker,
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    progress=False,
                    auto_adjust=True
                )
                
                if not df.empty:
                    if 'Close' in df.columns:
                        return df['Close']
                    elif isinstance(df.columns, pd.MultiIndex):
                        return df['Close'][yahoo_ticker]
                
            except Exception as e:
                logger.debug(f"Tentativa {tentativa + 1} falhou para {ticker}: {e}")
                if tentativa < self.max_retries - 1:
                    time.sleep(1)
        
        return None
    
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
        Busca histórico de dividendos
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial (padrão: 2 anos atrás)
            end_date: Data final (opcional)
            
        Returns:
            DataFrame com colunas ['data', 'valor']
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        yahoo_ticker = self._add_sa(ticker)
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            divs = stock.dividends
            
            if not divs.empty:
                df = pd.DataFrame({
                    'data': divs.index,
                    'valor': divs.values
                })
                
                # Filtrar por data
                df = df[df['data'] >= start_date]
                
                if end_date:
                    df = df[df['data'] <= end_date]
                
                df = df.reset_index(drop=True)
                
                logger.info(f"✓ {ticker}: {len(df)} dividendos")
                return df
        
        except Exception as e:
            logger.error(f"Erro ao buscar dividendos de {ticker}: {e}")
        
        return pd.DataFrame(columns=['data', 'valor'])
    
    # ==========================================
    # BUSCA EM LOTE
    # ==========================================
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Busca preços de múltiplos ativos
        
        Args:
            tickers: Lista de códigos
            
        Returns:
            Dicionário {ticker: preço}
        """
        logger.info(f"Buscando preços de {len(tickers)} ativos")
        
        precos = {}
        for ticker in tickers:
            precos[ticker] = self.obter_preco_atual(ticker)
            time.sleep(0.3)
        
        return precos
    
    # ==========================================
    # INFORMAÇÕES DO ATIVO
    # ==========================================
    
    def obter_informacoes_ativo(self, ticker: str) -> Dict[str, Any]:
        """
        Busca informações do ativo
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Dicionário com informações
        """
        yahoo_ticker = self._add_sa(ticker)
        
        info = {
            'ticker': ticker,
            'nome': ticker,
            'preco': None,
            'tipo': self._identificar_tipo(ticker)
        }
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            stock_info = stock.info
            
            if stock_info:
                info['nome'] = (
                    stock_info.get('longName') or 
                    stock_info.get('shortName') or 
                    ticker
                )
                info['preco'] = (
                    stock_info.get('currentPrice') or 
                    stock_info.get('regularMarketPrice')
                )
        
        except Exception as e:
            logger.debug(f"Erro ao buscar info de {ticker}: {e}")
        
        # Buscar preço separadamente se necessário
        if not info['preco']:
            info['preco'] = self.obter_preco_atual(ticker)
        
        return info
    
    def _identificar_tipo(self, ticker: str) -> str:
        """Identifica tipo do ativo"""
        ticker_limpo = self._remove_sa(ticker)
        
        if ticker_limpo.endswith('11'):
            if any(x in ticker_limpo for x in ['BOVA', 'SMAL', 'IVVB']):
                return 'ETF'
            return 'FII'
        
        if ticker_limpo[-1].isdigit():
            return 'ACAO'
        
        return 'OUTRO'
    
    # ==========================================
    # TESTE DE CONEXÃO
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """
        Testa conectividade com as APIs
        
        Returns:
            Dicionário com status
        """
        resultados = {
            'yahoo_finance': False,
            'brapi': False
        }
        
        # Testar Yahoo
        try:
            stock = yf.Ticker('PETR4.SA')
            hist = stock.history(period='1d')
            if not hist.empty:
                resultados['yahoo_finance'] = True
                logger.info("✓ Yahoo Finance: OK")
        except Exception as e:
            logger.error(f"✗ Yahoo Finance: {e}")
        
        # Testar BRAPI
        try:
            response = requests.get(
                f"{self.brapi_url}/quote/PETR4",
                timeout=self.timeout
            )
            if response.status_code == 200:
                resultados['brapi'] = True
                logger.info("✓ BRAPI: OK")
        except Exception as e:
            logger.error(f"✗ BRAPI: {e}")
        
        return resultados


# ==========================================
# INSTÂNCIA GLOBAL
# ==========================================

_manager = DataManager()


# ==========================================
# FUNÇÕES DE CONVENIÊNCIA
# ==========================================

def get_price_history(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime,
    use_cache: bool = True
) -> pd.DataFrame:
    """Busca histórico de preços"""
    return _manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(
    ticker: str, 
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Busca dividendos"""
    return _manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Busca preços atuais"""
    return _manager.get_current_prices(tickers)


def obter_preco_atual(ticker: str) -> Optional[float]:
    """Busca preço atual"""
    return _manager.obter_preco_atual(ticker)


def obter_informacoes_ativo(ticker: str) -> Dict[str, Any]:
    """Busca informações do ativo"""
    return _manager.obter_informacoes_ativo(ticker)


def testar_conexao() -> Dict[str, bool]:
    """Testa conexão com APIs"""
    return _manager.testar_conexao()


# Aliases para compatibilidade
obter_preco = obter_preco_atual
obter_info = obter_informacoes_ativo
obter_dividendos = get_dividends
obter_historico = get_price_history
testar_apis = testar_conexao
