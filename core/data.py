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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# CLASSE PRINCIPAL DE GERENCIAMENTO DE DADOS
# ==========================================

class DataManager:
    """Gerenciador de dados de ativos financeiros"""
    
    def __init__(self):
        """Inicializa o gerenciador de dados"""
        self.brapi_base_url = "https://brapi.dev/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.max_retries = 3
        self.timeout = 10
    
    # ==========================================
    # FUNÇÕES DE NORMALIZAÇÃO DE TICKERS
    # ==========================================
    
    def normalizar_ticker_yahoo(self, ticker: str) -> str:
        """
        Normaliza ticker para formato Yahoo Finance (.SA)
        
        Args:
            ticker: Código do ativo (ex: PETR4, MXRF11)
            
        Returns:
            Ticker formatado (ex: PETR4.SA)
        """
        ticker = ticker.upper().strip()
        if not ticker.endswith('.SA'):
            ticker += '.SA'
        return ticker
    
    def normalizar_ticker_brapi(self, ticker: str) -> str:
        """
        Normaliza ticker para formato BRAPI (sem .SA)
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Ticker sem sufixo .SA
        """
        return ticker.upper().strip().replace('.SA', '')
    
    # ==========================================
    # FUNÇÕES DE BUSCA DE PREÇO ATUAL
    # ==========================================
    
    def obter_preco_atual(self, ticker: str) -> Optional[float]:
        """
        Busca preço atual do ativo com fallback entre APIs
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço atual ou None se falhar
        """
        logger.info(f"Buscando preço para {ticker}")
        
        # Tentativa 1: Yahoo Finance
        preco = self._buscar_preco_yahoo(ticker)
        if preco is not None:
            logger.info(f"✓ Preço obtido via Yahoo Finance: R$ {preco:.2f}")
            return preco
        
        # Tentativa 2: BRAPI (fallback)
        logger.warning(f"⚠ Yahoo Finance falhou, tentando BRAPI para {ticker}")
        preco = self._buscar_preco_brapi(ticker)
        if preco is not None:
            logger.info(f"✓ Preço obtido via BRAPI: R$ {preco:.2f}")
            return preco
        
        logger.error(f"✗ Não foi possível obter preço para {ticker}")
        return None
    
    def _buscar_preco_yahoo(self, ticker: str) -> Optional[float]:
        """
        Busca preço usando Yahoo Finance com retry
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço ou None
        """
        yahoo_ticker = self.normalizar_ticker_yahoo(ticker)
        
        for tentativa in range(self.max_retries):
            try:
                stock = yf.Ticker(yahoo_ticker)
                
                # Método 1: Tentar via info
                info = stock.info
                if info:
                    preco = info.get('currentPrice') or info.get('regularMarketPrice')
                    if preco and preco > 0:
                        return float(preco)
                
                # Método 2: Tentar via histórico recente
                hist = stock.history(period='5d')
                if not hist.empty and len(hist) > 0:
                    ultimo_preco = hist['Close'].iloc[-1]
                    if ultimo_preco > 0:
                        return float(ultimo_preco)
                
                # Método 3: Tentar fast_info
                try:
                    fast_info = stock.fast_info
                    if hasattr(fast_info, 'last_price') and fast_info.last_price > 0:
                        return float(fast_info.last_price)
                except:
                    pass
                
            except Exception as e:
                logger.warning(f"Yahoo tentativa {tentativa + 1}/{self.max_retries}: {str(e)}")
                if tentativa < self.max_retries - 1:
                    time.sleep(2 ** tentativa)  # Exponential backoff
                continue
        
        return None
    
    def _buscar_preco_brapi(self, ticker: str) -> Optional[float]:
        """
        Busca preço usando BRAPI com retry
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço ou None
        """
        brapi_ticker = self.normalizar_ticker_brapi(ticker)
        
        for tentativa in range(self.max_retries):
            try:
                url = f"{self.brapi_base_url}/quote/{brapi_ticker}"
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and 'results' in data and len(data['results']) > 0:
                        result = data['results'][0]
                        preco = result.get('regularMarketPrice')
                        
                        if preco and preco > 0:
                            return float(preco)
                
            except Exception as e:
                logger.warning(f"BRAPI tentativa {tentativa + 1}/{self.max_retries}: {str(e)}")
                if tentativa < self.max_retries - 1:
                    time.sleep(2 ** tentativa)
                continue
        
        return None
    
    # ==========================================
    # FUNÇÕES DE BUSCA DE INFORMAÇÕES DO ATIVO
    # ==========================================
    
    def obter_informacoes_ativo(self, ticker: str) -> Dict[str, Any]:
        """
        Busca informações detalhadas do ativo
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Dicionário com informações do ativo
        """
        logger.info(f"Buscando informações para {ticker}")
        
        info_data = {
            'ticker': ticker,
            'nome': ticker,
            'preco': None,
            'moeda': 'BRL',
            'valor_mercado': None,
            'setor': None,
            'industria': None,
            'tipo': self._identificar_tipo_ativo(ticker)
        }
        
        yahoo_ticker = self.normalizar_ticker_yahoo(ticker)
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            stock_info = stock.info
            
            if stock_info:
                info_data['nome'] = stock_info.get('longName') or stock_info.get('shortName') or ticker
                info_data['valor_mercado'] = stock_info.get('marketCap')
                info_data['setor'] = stock_info.get('sector')
                info_data['industria'] = stock_info.get('industry')
                
                # Preço
                preco = stock_info.get('currentPrice') or stock_info.get('regularMarketPrice')
                if preco:
                    info_data['preco'] = float(preco)
            
        except Exception as e:
            logger.warning(f"Erro ao buscar info via Yahoo: {str(e)}")
        
        # Se não conseguiu preço, busca separadamente
        if info_data['preco'] is None:
            info_data['preco'] = self.obter_preco_atual(ticker)
        
        return info_data
    
    def _identificar_tipo_ativo(self, ticker: str) -> str:
        """
        Identifica o tipo de ativo pelo ticker
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Tipo do ativo (ACAO, FII, ETF, etc)
        """
        ticker_limpo = ticker.replace('.SA', '').upper()
        
        # FIIs geralmente terminam com 11
        if ticker_limpo.endswith('11'):
            # Verificar se é ETF
            etf_patterns = ['BOVA', 'SMAL', 'IVVB', 'PIBB']
            if any(pattern in ticker_limpo for pattern in etf_patterns):
                return 'ETF'
            return 'FII'
        
        # Ações terminam com 3, 4, 5, 6, etc
        if ticker_limpo[-1].isdigit():
            return 'ACAO'
        
        return 'OUTRO'
    
    # ==========================================
    # FUNÇÕES DE BUSCA DE DIVIDENDOS
    # ==========================================
    
    def obter_dividendos(self, ticker: str, data_inicio: Optional[datetime] = None) -> pd.DataFrame:
        """
        Busca histórico de dividendos/proventos
        
        Args:
            ticker: Código do ativo
            data_inicio: Data inicial (padrão: 2 anos atrás)
            
        Returns:
            DataFrame com colunas ['data', 'valor']
        """
        if data_inicio is None:
            data_inicio = datetime.now() - timedelta(days=730)  # 2 anos
        
        logger.info(f"Buscando dividendos para {ticker} desde {data_inicio.date()}")
        
        yahoo_ticker = self.normalizar_ticker_yahoo(ticker)
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            dividendos = stock.dividends
            
            if not dividendos.empty:
                # Converter para DataFrame
                df = pd.DataFrame({
                    'data': dividendos.index,
                    'valor': dividendos.values
                })
                
                # Filtrar por data
                df = df[df['data'] >= data_inicio]
                df = df.reset_index(drop=True)
                
                logger.info(f"✓ Encontrados {len(df)} dividendos para {ticker}")
                return df
            
        except Exception as e:
            logger.error(f"✗ Erro ao buscar dividendos de {ticker}: {str(e)}")
        
        logger.warning(f"⚠ Nenhum dividendo encontrado para {ticker}")
        return pd.DataFrame(columns=['data', 'valor'])
    
    def get_dividends(self, ticker: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Busca dividendos para compatibilidade com código existente
        Alias para obter_dividendos com suporte a end_date
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final (ignorado por enquanto)
            
        Returns:
            DataFrame com dividendos
        """
        return self.obter_dividendos(ticker, start_date)
    
    # ==========================================
    # FUNÇÕES DE BUSCA DE HISTÓRICO
    # ==========================================
    
    def obter_historico(self, ticker: str, periodo: str = '1y', intervalo: str = '1d') -> pd.DataFrame:
        """
        Busca dados históricos de preços
        
        Args:
            ticker: Código do ativo
            periodo: Período (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            intervalo: Intervalo (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame com histórico de preços
        """
        logger.info(f"Buscando histórico de {ticker} - período: {periodo}")
        
        yahoo_ticker = self.normalizar_ticker_yahoo(ticker)
        
        try:
            stock = yf.Ticker(yahoo_ticker)
            hist = stock.history(period=periodo, interval=intervalo)
            
            if not hist.empty:
                logger.info(f"✓ Histórico obtido: {len(hist)} registros")
                return hist
            
        except Exception as e:
            logger.error(f"✗ Erro ao buscar histórico de {ticker}: {str(e)}")
        
        logger.warning(f"⚠ Nenhum histórico encontrado para {ticker}")
        return pd.DataFrame()
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_price_history_cached(_self, tickers_tuple: tuple, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Versão com cache da busca de histórico
        
        Args:
            tickers_tuple: Tupla de códigos de ativos (tuple para ser hashable)
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com histórico de preços
        """
        tickers = list(tickers_tuple)
        return _self._get_price_history_no_cache(tickers, start_date, end_date)
    
    def _get_price_history_no_cache(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Busca histórico sem cache
        
        Args:
            tickers: Lista de códigos de ativos
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com histórico de preços
        """
        logger.info(f"Buscando histórico para {len(tickers)} ativos de {start_date.date()} até {end_date.date()}")
        
        all_data = {}
        
        for ticker in tickers:
            yahoo_ticker = self.normalizar_ticker_yahoo(ticker)
            
            try:
                stock = yf.Ticker(yahoo_ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Usar apenas a coluna Close
                    all_data[ticker] = hist['Close']
                    logger.info(f"✓ {ticker}: {len(hist)} registros")
                else:
                    logger.warning(f"⚠ {ticker}: sem dados")
                    all_data[ticker] = pd.Series(dtype=float)
                
            except Exception as e:
                logger.error(f"✗ Erro ao buscar {ticker}: {str(e)}")
                all_data[ticker] = pd.Series(dtype=float)
            
            time.sleep(0.3)  # Evitar rate limiting
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"✓ DataFrame criado com {len(df)} linhas e {len(df.columns)} colunas")
            return df
        
        logger.warning("⚠ Nenhum dado histórico foi obtido")
        return pd.DataFrame()
    
    def get_price_history(self, tickers: List[str], start_date: datetime, end_date: datetime, use_cache: bool = True) -> pd.DataFrame:
        """
        Busca histórico de preços para múltiplos ativos
        Função para compatibilidade com código existente
        
        Args:
            tickers: Lista de códigos de ativos
            start_date: Data inicial
            end_date: Data final
            use_cache: Se deve usar cache (padrão: True)
            
        Returns:
            DataFrame com histórico de preços (índice: data, colunas: tickers)
        """
        if use_cache:
            # Converter lista para tupla para ser hashable no cache
            tickers_tuple = tuple(sorted(tickers))
            return self._get_price_history_cached(tickers_tuple, start_date, end_date)
        else:
            return self._get_price_history_no_cache(tickers, start_date, end_date)
    
    # ==========================================
    # FUNÇÕES DE BUSCA EM LOTE
    # ==========================================
    
    def obter_precos_lote(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Busca preços de múltiplos ativos
        
        Args:
            tickers: Lista de códigos de ativos
            
        Returns:
            Dicionário {ticker: preço}
        """
        logger.info(f"Buscando preços para {len(tickers)} ativos")
        
        precos = {}
        for ticker in tickers:
            precos[ticker] = self.obter_preco_atual(ticker)
            time.sleep(0.5)  # Evitar rate limiting
        
        return precos
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Alias para obter_precos_lote (compatibilidade)
        
        Args:
            tickers: Lista de códigos de ativos
            
        Returns:
            Dicionário {ticker: preço}
        """
        return self.obter_precos_lote(tickers)
    
    # ==========================================
    # FUNÇÕES DE TESTE
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """
        Testa conectividade com as APIs
        
        Returns:
            Dicionário com status de cada API
        """
        logger.info("Testando conexão com APIs...")
        
        resultados = {
            'yahoo_finance': False,
            'brapi': False
        }
        
        # Testar Yahoo Finance
        try:
            test_ticker = yf.Ticker('PETR4.SA')
            info = test_ticker.info
            if info and len(info) > 0:
                resultados['yahoo_finance'] = True
                logger.info("✓ Yahoo Finance: OK")
        except Exception as e:
            logger.error(f"✗ Yahoo Finance: FALHOU - {str(e)}")
        
        # Testar BRAPI
        try:
            response = self.session.get(
                f"{self.brapi_base_url}/quote/PETR4",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data and 'results' in data:
                    resultados['brapi'] = True
                    logger.info("✓ BRAPI: OK")
        except Exception as e:
            logger.error(f"✗ BRAPI: FALHOU - {str(e)}")
        
        return resultados


# ==========================================
# INSTÂNCIA GLOBAL DO GERENCIADOR
# ==========================================

# Criar instância global para uso em todo o app
_data_manager = DataManager()


# ==========================================
# FUNÇÕES DE CONVENIÊNCIA (COMPATIBILIDADE)
# ==========================================

def obter_preco(ticker: str) -> Optional[float]:
    """Função de conveniência para obter preço"""
    return _data_manager.obter_preco_atual(ticker)


def obter_info(ticker: str) -> Dict[str, Any]:
    """Função de conveniência para obter informações"""
    return _data_manager.obter_informacoes_ativo(ticker)


def obter_dividendos(ticker: str) -> pd.DataFrame:
    """Função de conveniência para obter dividendos"""
    return _data_manager.obter_dividendos(ticker)


def obter_historico(ticker: str, periodo: str = '1y') -> pd.DataFrame:
    """Função de conveniência para obter histórico"""
    return _data_manager.obter_historico(ticker, periodo)


def testar_apis() -> Dict[str, bool]:
    """Função de conveniência para testar APIs"""
    return _data_manager.testar_conexao()


# Funções para compatibilidade com código existente
def get_price_history(tickers: List[str], start_date: datetime, end_date: datetime, use_cache: bool = True) -> pd.DataFrame:
    """Compatibilidade: busca histórico de preços"""
    return _data_manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(ticker: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Compatibilidade: busca dividendos"""
    return _data_manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Compatibilidade: busca preços atuais"""
    return _data_manager.get_current_prices(tickers)
