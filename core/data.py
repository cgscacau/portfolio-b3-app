"""
Módulo principal de gerenciamento de dados
Corrigido para funcionar com rate limiting do Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURAÇÃO GLOBAL DO YFINANCE
# ==========================================

# Força User-Agent para evitar erro 429
yf.utils.user_agent_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


class DataManager:
    """Gerenciador de dados de ativos financeiros"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.max_retries = 3
        self.delay_between_requests = 0.5
        logger.info("DataManager inicializado com User-Agent configurado")
    
    # ==========================================
    # NORMALIZAÇÃO
    # ==========================================
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normaliza ticker para formato Yahoo Finance"""
        ticker = str(ticker).upper().strip()
        if not ticker.endswith('.SA'):
            ticker = f"{ticker}.SA"
        return ticker
    
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
        ticker_normalizado = self._normalize_ticker(ticker)
        logger.info(f"Buscando preço de {ticker_normalizado}")
        
        for tentativa in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker_normalizado)
                hist = stock.history(period='5d')
                
                if not hist.empty and len(hist) > 0:
                    preco = float(hist['Close'].iloc[-1])
                    logger.info(f"✓ {ticker}: R$ {preco:.2f}")
                    return preco
                
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1}: {str(e)}")
                if tentativa < self.max_retries - 1:
                    time.sleep(2 ** tentativa)
                continue
        
        logger.error(f"✗ Não foi possível obter preço de {ticker}")
        return None
    
    # ==========================================
    # BUSCA DE HISTÓRICO
    # ==========================================
    
    def _download_single_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """
        Baixa histórico de um único ticker
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            Series com preços ou None
        """
        ticker_normalizado = self._normalize_ticker(ticker)
        
        for tentativa in range(self.max_retries):
            try:
                logger.info(f"  Tentativa {tentativa + 1}: {ticker_normalizado}")
                
                # Usar yf.download com sessão configurada
                data = yf.download(
                    ticker_normalizado,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    progress=False,
                    show_errors=False,
                    threads=False  # Importante: desabilitar threads
                )
                
                if data.empty:
                    logger.warning(f"    ⚠ Dados vazios")
                    if tentativa < self.max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Extrair Close
                if 'Close' in data.columns:
                    series = data['Close']
                elif isinstance(data.columns, pd.MultiIndex):
                    if ('Close', ticker_normalizado) in data.columns:
                        series = data[('Close', ticker_normalizado)]
                    else:
                        series = data['Close'].iloc[:, 0]
                else:
                    logger.warning(f"    ⚠ Estrutura inesperada")
                    return None
                
                series = series.dropna()
                
                if len(series) > 0:
                    logger.info(f"    ✓ {len(series)} registros")
                    return series
                
            except Exception as e:
                logger.error(f"    ✗ Erro: {str(e)}")
                if tentativa < self.max_retries - 1:
                    time.sleep(2 ** tentativa)
                continue
        
        return None
    
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
            tickers: Lista de códigos
            start_date: Data inicial
            end_date: Data final
            use_cache: Usar cache (ignorado)
            
        Returns:
            DataFrame com histórico
        """
        logger.info("=" * 60)
        logger.info(f"BUSCA DE HISTÓRICO")
        logger.info(f"Ativos: {len(tickers)}")
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        logger.info("=" * 60)
        
        if start_date >= end_date:
            logger.error("✗ Data inicial >= data final")
            return pd.DataFrame()
        
        if not tickers:
            logger.error("✗ Lista de tickers vazia")
            return pd.DataFrame()
        
        all_data = {}
        sucessos = 0
        falhas = 0
        
        # Processar em lotes pequenos
        batch_size = 5
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"\nLote {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            for idx, ticker in enumerate(batch, 1):
                logger.info(f"[{i+idx}/{len(tickers)}] {ticker}")
                
                series = self._download_single_ticker(ticker, start_date, end_date)
                
                if series is not None and not series.empty:
                    all_data[ticker] = series
                    sucessos += 1
                else:
                    falhas += 1
                
                # Delay entre requisições
                time.sleep(self.delay_between_requests)
            
            # Delay maior entre lotes
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
        Busca histórico de dividendos
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com dividendos
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        ticker_normalizado = self._normalize_ticker(ticker)
        logger.info(f"Buscando dividendos de {ticker_normalizado}")
        
        for tentativa in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker_normalizado)
                dividends = stock.dividends
                
                if not dividends.empty:
                    df = pd.DataFrame({
                        'data': dividends.index,
                        'valor': dividends.values
                    })
                    
                    df = df[df['data'] >= start_date]
                    if end_date:
                        df = df[df['data'] <= end_date]
                    
                    df = df.reset_index(drop=True)
                    
                    logger.info(f"✓ {ticker}: {len(df)} dividendos")
                    return df
                
            except Exception as e:
                logger.warning(f"Tentativa {tentativa + 1}: {str(e)}")
                if tentativa < self.max_retries - 1:
                    time.sleep(2)
                continue
        
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
            time.sleep(self.delay_between_requests)
        
        return precos
    
    # ==========================================
    # INFORMAÇÕES
    # ==========================================
    
    def obter_informacoes_ativo(self, ticker: str) -> Dict[str, Any]:
        """Busca informações do ativo"""
        ticker_normalizado = self._normalize_ticker(ticker)
        
        info = {
            'ticker': ticker,
            'nome': ticker,
            'preco': None,
            'tipo': 'ACAO'
        }
        
        try:
            stock = yf.Ticker(ticker_normalizado)
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
        except:
            pass
        
        if not info['preco']:
            info['preco'] = self.obter_preco_atual(ticker)
        
        return info
    
    # ==========================================
    # TESTE
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """Testa conexão"""
        logger.info("Testando conexão...")
        
        resultado = {'yahoo_finance': False}
        
        try:
            stock = yf.Ticker('PETR4.SA')
            hist = stock.history(period='5d')
            
            if not hist.empty:
                resultado['yahoo_finance'] = True
                logger.info("✓ Yahoo Finance: OK")
            else:
                logger.error("✗ Yahoo Finance: sem dados")
        except Exception as e:
            logger.error(f"✗ Yahoo Finance: {str(e)}")
        
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
    """Busca histórico de preços"""
    return _data_manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Busca dividendos"""
    return _data_manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Busca preços atuais"""
    return _data_manager.get_current_prices(tickers)


def obter_preco_atual(ticker: str) -> Optional[float]:
    """Busca preço atual"""
    return _data_manager.obter_preco_atual(ticker)


def obter_informacoes_ativo(ticker: str) -> Dict[str, Any]:
    """Busca informações"""
    return _data_manager.obter_informacoes_ativo(ticker)


def testar_conexao() -> Dict[str, bool]:
    """Testa conexão"""
    return _data_manager.testar_conexao()


# Aliases
obter_preco = obter_preco_atual
obter_info = obter_informacoes_ativo
obter_dividendos = get_dividends
testar_apis = testar_conexao
