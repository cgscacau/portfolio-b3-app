"""
core/data_mock.py
Dados simulados para fallback quando yfinance falhar
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Lista de blue chips conhecidos da B3
BLUE_CHIPS = [
    'PETR4.SA', 'PETR3.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 
    'BBDC3.SA', 'BBAS3.SA', 'ABEV3.SA', 'WEGE3.SA', 'RENT3.SA',
    'B3SA3.SA', 'SUZB3.SA', 'RAIL3.SA', 'ELET3.SA', 'CMIG4.SA',
    'CSAN3.SA', 'GGBR4.SA', 'CSNA3.SA', 'USIM5.SA', 'VIVT3.SA'
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
        
        # Simular retornos diários
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_days)  # drift=0.05%/dia, vol=2%/dia
        
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
        # Decidir se paga dividendos (80% pagam)
        if np.random.random() > 0.2:
            # Número de pagamentos no período
            months = (end - start).days / 30
            n_payments = int(np.random.uniform(1, min(months, 12)))
            
            # Datas aleatórias
            dates = pd.date_range(start=start, end=end, freq='M')[:n_payments]
            
            # Valores aleatórios
            values = np.random.uniform(0.1, 2.0, n_payments)
            
            dividends_dict[ticker] = pd.Series(values, index=dates)
    
    return dividends_dict
