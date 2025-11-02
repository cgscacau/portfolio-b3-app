"""
Módulo principal de gerenciamento de dados
Responsável por buscar cotações, dividendos e informações de ativos
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List

# Configuração de logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Gerenciador de dados de ativos financeiros"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.max_retries = 3
        logger.info("DataManager inicializado")
    
    # ==========================================
    # NORMALIZAÇÃO DE TICKERS
    # ==========================================
    
    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normaliza ticker para formato Yahoo Finance
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Ticker com .SA
        """
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
            Preço atual ou None se falhar
        """
        ticker_normalizado = self._normalize_ticker(ticker)
        logger.info(f"Buscando preço de {ticker_normalizado}")
        
        try:
            # Criar ticker
            stock = yf.Ticker(ticker_normalizado)
            
            # Tentar obter histórico recente (mais confiável)
            hist = stock.history(period='5d')
            
            if not hist.empty and len(hist) > 0:
                preco = float(hist['Close'].iloc[-1])
                logger.info(f"✓ {ticker}: R$ {preco:.2f}")
                return preco
            else:
                logger.warning(f"⚠ {ticker}: histórico vazio")
                return None
                
        except Exception as e:
            logger.error(f"✗ Erro ao buscar {ticker}: {str(e)}")
            return None
    
    # ==========================================
    # BUSCA DE HISTÓRICO - MÉTODO SIMPLIFICADO
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
            Series com preços de fechamento ou None
        """
        ticker_normalizado = self._normalize_ticker(ticker)
        
        try:
            logger.info(f"  Baixando {ticker_normalizado}...")
            
            # Usar yf.download que é mais robusto
            data = yf.download(
                ticker_normalizado,
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False,
                show_errors=False
            )
            
            # Verificar se obteve dados
            if data.empty:
                logger.warning(f"    ⚠ {ticker}: sem dados retornados")
                return None
            
            # Extrair coluna Close
            if 'Close' in data.columns:
                series = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                # Caso tenha MultiIndex (quando baixa múltiplos tickers)
                if ('Close', ticker_normalizado) in data.columns:
                    series = data[('Close', ticker_normalizado)]
                else:
                    series = data['Close'].iloc[:, 0]
            else:
                logger.warning(f"    ⚠ {ticker}: estrutura de dados inesperada")
                return None
            
            # Remover NaN
            series = series.dropna()
            
            if len(series) > 0:
                logger.info(f"    ✓ {ticker}: {len(series)} registros")
                return series
            else:
                logger.warning(f"    ⚠ {ticker}: série vazia após limpeza")
                return None
                
        except Exception as e:
            logger.error(f"    ✗ {ticker}: erro - {str(e)}")
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
            tickers: Lista de códigos de ativos
            start_date: Data inicial
            end_date: Data final
            use_cache: Usar cache (ignorado nesta versão)
            
        Returns:
            DataFrame com histórico (índice: data, colunas: tickers)
        """
        logger.info("=" * 60)
        logger.info(f"INICIANDO BUSCA DE HISTÓRICO")
        logger.info(f"Ativos: {len(tickers)}")
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        logger.info("=" * 60)
        
        # Validar datas
        if start_date >= end_date:
            logger.error("✗ Data inicial deve ser anterior à data final")
            return pd.DataFrame()
        
        # Validar tickers
        if not tickers or len(tickers) == 0:
            logger.error("✗ Lista de tickers vazia")
            return pd.DataFrame()
        
        # Dicionário para armazenar os dados
        all_data = {}
        sucessos = 0
        falhas = 0
        
        # Processar cada ticker individualmente
        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{idx}/{len(tickers)}] Processando {ticker}")
            
            # Baixar dados
            series = self._download_single_ticker(ticker, start_date, end_date)
            
            if series is not None and not series.empty:
                all_data[ticker] = series
                sucessos += 1
            else:
                falhas += 1
            
            # Pequeno delay para evitar rate limiting
            if idx < len(tickers):
                time.sleep(0.5)
        
        # Resumo
        logger.info("\n" + "=" * 60)
        logger.info(f"RESUMO: {sucessos} sucessos, {falhas} falhas")
        logger.info("=" * 60)
        
        # Criar DataFrame
        if all_data:
            try:
                df = pd.DataFrame(all_data)
                
                # Ordenar por data
                df = df.sort_index()
                
                logger.info(f"✓ DataFrame criado: {len(df)} linhas x {len(df.columns)} colunas")
                logger.info(f"✓ Período real: {df.index[0].date()} até {df.index[-1].date()}")
                
                return df
                
            except Exception as e:
                logger.error(f"✗ Erro ao criar DataFrame: {str(e)}")
                return pd.DataFrame()
        else:
            logger.error("✗ Nenhum dado foi obtido para nenhum ativo")
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
            start_date: Data inicial (padrão: 2 anos atrás)
            end_date: Data final (opcional)
            
        Returns:
            DataFrame com colunas ['data', 'valor']
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        ticker_normalizado = self._normalize_ticker(ticker)
        logger.info(f"Buscando dividendos de {ticker_normalizado}")
        
        try:
            stock = yf.Ticker(ticker_normalizado)
            dividends = stock.dividends
            
            if not dividends.empty:
                # Converter para DataFrame
                df = pd.DataFrame({
                    'data': dividends.index,
                    'valor': dividends.values
                })
                
                # Filtrar por data
                df = df[df['data'] >= start_date]
                
                if end_date:
                    df = df[df['data'] <= end_date]
                
                df = df.reset_index(drop=True)
                
                logger.info(f"✓ {ticker}: {len(df)} dividendos encontrados")
                return df
            else:
                logger.warning(f"⚠ {ticker}: sem dividendos")
                return pd.DataFrame(columns=['data', 'valor'])
                
        except Exception as e:
            logger.error(f"✗ Erro ao buscar dividendos de {ticker}: {str(e)}")
            return pd.DataFrame(columns=['data', 'valor'])
    
    # ==========================================
    # BUSCA EM LOTE
    # ==========================================
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """
        Busca preços atuais de múltiplos ativos
        
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
        Busca informações básicas do ativo
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Dicionário com informações
        """
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
        except Exception as e:
            logger.debug(f"Erro ao buscar info de {ticker}: {e}")
        
        # Buscar preço separadamente se necessário
        if not info['preco']:
            info['preco'] = self.obter_preco_atual(ticker)
        
        return info
    
    # ==========================================
    # TESTE DE CONEXÃO
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """
        Testa conectividade com Yahoo Finance
        
        Returns:
            Dicionário com status
        """
        logger.info("Testando conexão com Yahoo Finance...")
        
        resultado = {'yahoo_finance': False}
        
        try:
            # Testar com PETR4
            stock = yf.Ticker('PETR4.SA')
            hist = stock.history(period='5d')
            
            if not hist.empty:
                resultado['yahoo_finance'] = True
                logger.info("✓ Yahoo Finance: OK")
            else:
                logger.error("✗ Yahoo Finance: sem dados")
                
        except Exception as e:
            logger.error(f"✗ Yahoo Finance: erro - {str(e)}")
        
        return resultado


# ==========================================
# INSTÂNCIA GLOBAL
# ==========================================

_data_manager = DataManager()


# ==========================================
# FUNÇÕES PÚBLICAS (COMPATIBILIDADE)
# ==========================================

def get_price_history(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Busca histórico de preços
    
    Args:
        tickers: Lista de códigos de ativos
        start_date: Data inicial
        end_date: Data final
        use_cache: Usar cache (ignorado)
        
    Returns:
        DataFrame com histórico
    """
    return _data_manager.get_price_history(tickers, start_date, end_date, use_cache)


def get_dividends(
    ticker: str, 
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Busca dividendos
    
    Args:
        ticker: Código do ativo
        start_date: Data inicial
        end_date: Data final
        
    Returns:
        DataFrame com dividendos
    """
    return _data_manager.get_dividends(ticker, start_date, end_date)


def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Busca preços atuais
    
    Args:
        tickers: Lista de códigos
        
    Returns:
        Dicionário {ticker: preço}
    """
    return _data_manager.get_current_prices(tickers)


def obter_preco_atual(ticker: str) -> Optional[float]:
    """Busca preço atual de um ativo"""
    return _data_manager.obter_preco_atual(ticker)


def obter_informacoes_ativo(ticker: str) -> Dict[str, Any]:
    """Busca informações de um ativo"""
    return _data_manager.obter_informacoes_ativo(ticker)


def testar_conexao() -> Dict[str, bool]:
    """Testa conexão com APIs"""
    return _data_manager.testar_conexao()


# Aliases adicionais
obter_preco = obter_preco_atual
obter_info = obter_informacoes_ativo
obter_dividendos = get_dividends
testar_apis = testar_conexao
