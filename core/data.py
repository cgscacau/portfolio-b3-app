"""
Módulo de gerenciamento de dados usando BRAPI
API brasileira para dados da B3
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Gerenciador de dados usando BRAPI"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        self.base_url = "https://brapi.dev/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
        self.timeout = 30
        logger.info("DataManager inicializado com BRAPI")
    
    # ==========================================
    # NORMALIZAÇÃO
    # ==========================================
    
    def _clean_ticker(self, ticker: str) -> str:
        """Remove .SA do ticker"""
        return str(ticker).upper().strip().replace('.SA', '')
    
    # ==========================================
    # BUSCA DE PREÇO ATUAL
    # ==========================================
    
    def obter_preco_atual(self, ticker: str) -> Optional[float]:
        """
        Busca preço atual via BRAPI
        
        Args:
            ticker: Código do ativo
            
        Returns:
            Preço atual ou None
        """
        ticker_limpo = self._clean_ticker(ticker)
        
        try:
            url = f"{self.base_url}/quote/{ticker_limpo}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    result = data['results'][0]
                    preco = result.get('regularMarketPrice')
                    
                    if preco and preco > 0:
                        logger.info(f"✓ {ticker}: R$ {preco:.2f}")
                        return float(preco)
            
            logger.warning(f"⚠ {ticker}: sem preço disponível")
            return None
            
        except Exception as e:
            logger.error(f"✗ {ticker}: erro - {str(e)}")
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
        Busca histórico de preços via BRAPI
        
        Args:
            tickers: Lista de códigos
            start_date: Data inicial
            end_date: Data final
            use_cache: Ignorado (mantido para compatibilidade)
            
        Returns:
            DataFrame com histórico
        """
        logger.info("=" * 60)
        logger.info(f"BUSCA DE HISTÓRICO VIA BRAPI")
        logger.info(f"Ativos: {len(tickers)}")
        logger.info(f"Período: {start_date.date()} até {end_date.date()}")
        logger.info("=" * 60)
        
        # Calcular range (BRAPI usa range em texto)
        dias = (end_date - start_date).days
        
        if dias <= 7:
            range_param = "5d"
        elif dias <= 30:
            range_param = "1mo"
        elif dias <= 90:
            range_param = "3mo"
        elif dias <= 180:
            range_param = "6mo"
        elif dias <= 365:
            range_param = "1y"
        elif dias <= 730:
            range_param = "2y"
        else:
            range_param = "5y"
        
        logger.info(f"Range calculado: {range_param}")
        
        all_data = {}
        sucessos = 0
        falhas = 0
        
        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"[{idx}/{len(tickers)}] Processando {ticker}")
            
            series = self._buscar_historico_ativo(ticker, range_param, start_date, end_date)
            
            if series is not None and not series.empty:
                all_data[ticker] = series
                sucessos += 1
                logger.info(f"  ✓ {len(series)} registros")
            else:
                falhas += 1
                logger.warning(f"  ✗ Sem dados")
            
            # Delay entre requisições
            if idx < len(tickers):
                time.sleep(0.5)
        
        logger.info(f"\nRESUMO: {sucessos} sucessos, {falhas} falhas")
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_index()
            
            logger.info(f"✓ DataFrame: {len(df)} linhas x {len(df.columns)} colunas")
            return df
        
        logger.error("✗ Nenhum dado obtido")
        return pd.DataFrame()
    
    def _buscar_historico_ativo(
        self,
        ticker: str,
        range_param: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.Series]:
        """
        Busca histórico de um ativo via BRAPI
        
        Args:
            ticker: Código do ativo
            range_param: Range (5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            start_date: Data inicial para filtro
            end_date: Data final para filtro
            
        Returns:
            Series com preços ou None
        """
        ticker_limpo = self._clean_ticker(ticker)
        
        try:
            url = f"{self.base_url}/quote/{ticker_limpo}"
            params = {
                'range': range_param,
                'interval': '1d',
                'fundamental': 'false'
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"  Status code: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data.get('results') or len(data['results']) == 0:
                logger.warning(f"  Sem resultados na resposta")
                return None
            
            result = data['results'][0]
            historical_data = result.get('historicalDataPrice', [])
            
            if not historical_data:
                logger.warning(f"  Sem dados históricos")
                return None
            
            # Converter para DataFrame
            df = pd.DataFrame(historical_data)
            
            # Converter timestamp para datetime
            df['date'] = pd.to_datetime(df['date'], unit='s')
            
            # Filtrar por período solicitado
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if df.empty:
                logger.warning(f"  Dados vazios após filtro de data")
                return None
            
            # Criar Series com preço de fechamento
            df = df.set_index('date')
            series = df['close'].sort_index()
            
            return series
            
        except Exception as e:
            logger.error(f"  Erro: {str(e)}")
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
        Busca dividendos via BRAPI
        
        Args:
            ticker: Código do ativo
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            DataFrame com dividendos
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        
        ticker_limpo = self._clean_ticker(ticker)
        
        try:
            url = f"{self.base_url}/quote/{ticker_limpo}"
            params = {'dividends': 'true'}
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    result = data['results'][0]
                    dividends_data = result.get('dividendsData', {})
                    cash_dividends = dividends_data.get('cashDividends', [])
                    
                    if cash_dividends:
                        df = pd.DataFrame(cash_dividends)
                        
                        # Converter paymentDate para datetime
                        df['data'] = pd.to_datetime(df['paymentDate'])
                        df['valor'] = df['rate']
                        
                        # Filtrar por data
                        df = df[df['data'] >= start_date]
                        if end_date:
                            df = df[df['data'] <= end_date]
                        
                        df = df[['data', 'valor']].reset_index(drop=True)
                        
                        logger.info(f"✓ {ticker}: {len(df)} dividendos")
                        return df
            
            logger.warning(f"⚠ {ticker}: sem dividendos")
            
        except Exception as e:
            logger.error(f"✗ {ticker}: erro - {str(e)}")
        
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
        
        # BRAPI permite buscar múltiplos de uma vez
        tickers_limpos = [self._clean_ticker(t) for t in tickers]
        tickers_str = ','.join(tickers_limpos)
        
        try:
            url = f"{self.base_url}/quote/{tickers_str}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results'):
                    precos = {}
                    
                    for result in data['results']:
                        ticker_original = result['symbol']
                        preco = result.get('regularMarketPrice')
                        
                        # Encontrar ticker original na lista
                        for t in tickers:
                            if self._clean_ticker(t) == ticker_original:
                                precos[t] = float(preco) if preco else None
                                break
                    
                    return precos
        
        except Exception as e:
            logger.error(f"Erro na busca em lote: {str(e)}")
        
        # Fallback: buscar um por um
        logger.warning("Fallback: buscando preços individualmente")
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
        ticker_limpo = self._clean_ticker(ticker)
        
        info = {
            'ticker': ticker,
            'nome': ticker,
            'preco': None,
            'tipo': 'ACAO'
        }
        
        try:
            url = f"{self.base_url}/quote/{ticker_limpo}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    result = data['results'][0]
                    
                    info['nome'] = result.get('longName') or result.get('shortName') or ticker
                    info['preco'] = result.get('regularMarketPrice')
                    info['tipo'] = result.get('type', 'ACAO')
        
        except Exception as e:
            logger.error(f"Erro ao buscar info de {ticker}: {str(e)}")
        
        return info
    
    # ==========================================
    # TESTE
    # ==========================================
    
    def testar_conexao(self) -> Dict[str, bool]:
        """Testa conexão com BRAPI"""
        logger.info("Testando BRAPI...")
        
        resultado = {'brapi': False}
        
        try:
            url = f"{self.base_url}/quote/PETR4"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    resultado['brapi'] = True
                    logger.info("✓ BRAPI: OK")
                    return resultado
            
            logger.error("✗ BRAPI: sem dados")
            
        except Exception as e:
            logger.error(f"✗ BRAPI: erro - {str(e)}")
        
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
