"""
core/metrics.py
Cálculo de métricas de desempenho, risco e dividendos
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Constantes
TRADING_DAYS_YEAR = 252
MONTHS_YEAR = 12
WEEKS_YEAR = 52


class PerformanceMetrics:
    """Classe para cálculo de métricas de performance e risco."""
    
    def __init__(self, prices_df: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Inicializa com dados de preços.
        
        Args:
            prices_df: DataFrame com preços (índice datetime, colunas = tickers)
            risk_free_rate: Taxa livre de risco anual
        """
        self.prices_df = prices_df
        self.risk_free_rate = risk_free_rate
        self.returns_df = self._calculate_returns()
        self.log_returns_df = self._calculate_log_returns()
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calcula retornos simples diários."""
        if self.prices_df.empty:
            return pd.DataFrame()
        
        returns = self.prices_df.pct_change().dropna()
        return returns
    
    def _calculate_log_returns(self) -> pd.DataFrame:
        """Calcula retornos logarítmicos diários."""
        if self.prices_df.empty:
            return pd.DataFrame()
        
        log_returns = np.log(self.prices_df / self.prices_df.shift(1)).dropna()
        return log_returns
    
    def get_returns(self, log_returns: bool = False) -> pd.DataFrame:
        """
        Retorna DataFrame de retornos.
        
        Args:
            log_returns: Se True, retorna retornos logarítmicos
        
        Returns:
            DataFrame de retornos
        """
        return self.log_returns_df if log_returns else self.returns_df
    
    def calculate_cumulative_returns(self, ticker: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calcula retornos cumulativos.
        
        Args:
            ticker: Se especificado, retorna apenas para esse ticker
        
        Returns:
            Series ou DataFrame com retornos cumulativos
        """
        if self.returns_df.empty:
            return pd.Series() if ticker else pd.DataFrame()
        
        cum_returns = (1 + self.returns_df).cumprod() - 1
        
        if ticker and ticker in cum_returns.columns:
            return cum_returns[ticker]
        
        return cum_returns
    
    def calculate_annualized_return(self, ticker: str) -> float:
        """
        Calcula retorno anualizado.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Retorno anualizado (decimal)
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        returns = self.returns_df[ticker].dropna()
        
        if len(returns) == 0:
            return np.nan
        
        # Retorno total do período
        total_return = (1 + returns).prod() - 1
        
        # Anualizar
        n_days = len(returns)
        annualized = (1 + total_return) ** (TRADING_DAYS_YEAR / n_days) - 1
        
        return annualized
    
    def calculate_annualized_volatility(self, ticker: str) -> float:
        """
        Calcula volatilidade anualizada.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Volatilidade anualizada (decimal)
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        returns = self.returns_df[ticker].dropna()
        
        if len(returns) == 0:
            return np.nan
        
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_YEAR)
        
        return annualized_vol
    
    def calculate_sharpe_ratio(self, ticker: str) -> float:
        """
        Calcula Índice de Sharpe.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Sharpe ratio
        """
        ann_return = self.calculate_annualized_return(ticker)
        ann_vol = self.calculate_annualized_volatility(ticker)
        
        if np.isnan(ann_return) or np.isnan(ann_vol) or ann_vol == 0:
            return np.nan
        
        sharpe = (ann_return - self.risk_free_rate) / ann_vol
        
        return sharpe
    
    def calculate_sortino_ratio(self, ticker: str, target_return: float = 0.0) -> float:
        """
        Calcula Índice de Sortino (considera apenas downside risk).
        
        Args:
            ticker: Ticker do ativo
            target_return: Retorno alvo (anualizado)
        
        Returns:
            Sortino ratio
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        returns = self.returns_df[ticker].dropna()
        
        if len(returns) == 0:
            return np.nan
        
        ann_return = self.calculate_annualized_return(ticker)
        
        # Downside deviation
        target_daily = target_return / TRADING_DAYS_YEAR
        downside_returns = returns[returns < target_daily]
        
        if len(downside_returns) == 0:
            return np.nan
        
        downside_std = downside_returns.std()
        downside_vol = downside_std * np.sqrt(TRADING_DAYS_YEAR)
        
        if downside_vol == 0:
            return np.nan
        
        sortino = (ann_return - target_return) / downside_vol
        
        return sortino
    
    def calculate_max_drawdown(self, ticker: str) -> Dict[str, float]:
        """
        Calcula máximo drawdown.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Dict com max_drawdown, peak_date, trough_date, recovery_date
        """
        if ticker not in self.prices_df.columns:
            return {'max_drawdown': np.nan}
        
        prices = self.prices_df[ticker].dropna()
        
        if len(prices) == 0:
            return {'max_drawdown': np.nan}
        
        # Calcular running maximum
        running_max = prices.expanding().max()
        
        # Drawdown em cada ponto
        drawdown = (prices - running_max) / running_max
        
        # Máximo drawdown
        max_dd = drawdown.min()
        
        # Encontrar datas
        trough_date = drawdown.idxmin()
        peak_date = prices[:trough_date].idxmax()
        
        # Recovery (se houver)
        recovery_date = None
        if trough_date < prices.index[-1]:
            future_prices = prices[trough_date:]
            peak_price = prices[peak_date]
            recovered = future_prices[future_prices >= peak_price]
            if len(recovered) > 0:
                recovery_date = recovered.index[0]
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'drawdown_days': (trough_date - peak_date).days if peak_date else None,
            'recovery_days': (recovery_date - trough_date).days if recovery_date else None
        }
    
    def calculate_calmar_ratio(self, ticker: str) -> float:
        """
        Calcula Calmar Ratio (retorno anualizado / max drawdown).
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Calmar ratio
        """
        ann_return = self.calculate_annualized_return(ticker)
        max_dd_info = self.calculate_max_drawdown(ticker)
        max_dd = abs(max_dd_info['max_drawdown'])
        
        if np.isnan(ann_return) or np.isnan(max_dd) or max_dd == 0:
            return np.nan
        
        calmar = ann_return / max_dd
        
        return calmar
    
    def calculate_var(self, ticker: str, confidence: float = 0.95) -> float:
        """
        Calcula Value at Risk (VaR) paramétrico.
        
        Args:
            ticker: Ticker do ativo
            confidence: Nível de confiança (0.95 = 95%)
        
        Returns:
            VaR diário (valor positivo = perda)
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        returns = self.returns_df[ticker].dropna()
        
        if len(returns) == 0:
            return np.nan
        
        mean = returns.mean()
        std = returns.std()
        
        # Z-score para o nível de confiança
        z_score = stats.norm.ppf(1 - confidence)
        
        var = -(mean + z_score * std)
        
        return var
    
    def calculate_cvar(self, ticker: str, confidence: float = 0.95) -> float:
        """
        Calcula Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            ticker: Ticker do ativo
            confidence: Nível de confiança
        
        Returns:
            CVaR diário (valor positivo = perda esperada)
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        returns = self.returns_df[ticker].dropna()
        
        if len(returns) == 0:
            return np.nan
        
        var = self.calculate_var(ticker, confidence)
        
        # Média dos retornos piores que o VaR
        worst_returns = returns[returns <= -var]
        
        if len(worst_returns) == 0:
            return var
        
        cvar = -worst_returns.mean()
        
        return cvar
    
    def calculate_beta(self, ticker: str, market_returns: pd.Series) -> float:
        """
        Calcula Beta em relação a um benchmark.
        
        Args:
            ticker: Ticker do ativo
            market_returns: Series com retornos do mercado
        
        Returns:
            Beta
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        asset_returns = self.returns_df[ticker].dropna()
        
        # Alinhar datas
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 30:  # Mínimo de observações
            return np.nan
        
        covariance = aligned['asset'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        if market_variance == 0:
            return np.nan
        
        beta = covariance / market_variance
        
        return beta
    
    def calculate_alpha(self, ticker: str, market_returns: pd.Series) -> float:
        """
        Calcula Alpha (Jensen's Alpha).
        
        Args:
            ticker: Ticker do ativo
            market_returns: Series com retornos do mercado
        
        Returns:
            Alpha anualizado
        """
        ann_return = self.calculate_annualized_return(ticker)
        beta = self.calculate_beta(ticker, market_returns)
        
        if np.isnan(ann_return) or np.isnan(beta):
            return np.nan
        
        # Retorno anualizado do mercado
        market_ann_return = (1 + market_returns.mean()) ** TRADING_DAYS_YEAR - 1
        
        # CAPM expected return
        expected_return = self.risk_free_rate + beta * (market_ann_return - self.risk_free_rate)
        
        alpha = ann_return - expected_return
        
        return alpha
    
    def calculate_information_ratio(self, ticker: str, benchmark_returns: pd.Series) -> float:
        """
        Calcula Information Ratio.
        
        Args:
            ticker: Ticker do ativo
            benchmark_returns: Series com retornos do benchmark
        
        Returns:
            Information ratio
        """
        if ticker not in self.returns_df.columns:
            return np.nan
        
        asset_returns = self.returns_df[ticker].dropna()
        
        # Alinhar datas
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 30:
            return np.nan
        
        # Excess returns
        excess_returns = aligned['asset'] - aligned['benchmark']
        
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        
        if std_excess == 0:
            return np.nan
        
        # Anualizar
        ir = (mean_excess / std_excess) * np.sqrt(TRADING_DAYS_YEAR)
        
        return ir
    
    def calculate_all_metrics(self, ticker: str) -> Dict[str, float]:
        """
        Calcula todas as métricas para um ticker.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Dicionário com todas as métricas
        """
        metrics = {
            'ticker': ticker,
            'annualized_return': self.calculate_annualized_return(ticker),
            'annualized_volatility': self.calculate_annualized_volatility(ticker),
            'sharpe_ratio': self.calculate_sharpe_ratio(ticker),
            'sortino_ratio': self.calculate_sortino_ratio(ticker),
            'calmar_ratio': self.calculate_calmar_ratio(ticker),
            'max_drawdown': self.calculate_max_drawdown(ticker)['max_drawdown'],
            'var_95': self.calculate_var(ticker, 0.95),
            'cvar_95': self.calculate_cvar(ticker, 0.95),
        }
        
        return metrics
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calcula matriz de correlação dos retornos.
        
        Returns:
            DataFrame com correlações
        """
        if self.returns_df.empty:
            return pd.DataFrame()
        
        corr_matrix = self.returns_df.corr()
        
        return corr_matrix
    
    def get_covariance_matrix(self, annualized: bool = True) -> pd.DataFrame:
        """
        Calcula matriz de covariância dos retornos.
        
        Args:
            annualized: Se True, retorna covariância anualizada
        
        Returns:
            DataFrame com covariâncias
        """
        if self.returns_df.empty:
            return pd.DataFrame()
        
        cov_matrix = self.returns_df.cov()
        
        if annualized:
            cov_matrix = cov_matrix * TRADING_DAYS_YEAR
        
        return cov_matrix


class DividendMetrics:
    """Classe para cálculo de métricas de dividendos."""
    
    def __init__(self, dividends_dict: Dict[str, pd.Series], prices_df: pd.DataFrame):
        """
        Inicializa com dados de dividendos e preços.
        
        Args:
            dividends_dict: Dicionário {ticker: Series de dividendos}
            prices_df: DataFrame com preços
        """
        self.dividends_dict = dividends_dict
        self.prices_df = prices_df
    
    def calculate_dividend_yield(self, ticker: str, method: str = 'trailing') -> float:
        """
        Calcula dividend yield.
        
        Args:
            ticker: Ticker do ativo
            method: 'trailing' (últimos 12 meses) ou 'average' (média do período)
        
        Returns:
            Dividend yield anual (decimal)
        """
        if ticker not in self.dividends_dict or ticker not in self.prices_df.columns:
            return np.nan
        
        divs = self.dividends_dict[ticker]
        prices = self.prices_df[ticker].dropna()
        
        if divs.empty or prices.empty:
            return np.nan
        
        if method == 'trailing':
            # Últimos 12 meses
            end_date = divs.index[-1]
            start_date = end_date - pd.DateOffset(months=12)
            recent_divs = divs[divs.index >= start_date]
            
            if recent_divs.empty:
                return np.nan
            
            total_divs = recent_divs.sum()
            
            # Preço médio do período
            recent_prices = prices[prices.index >= start_date]
            if recent_prices.empty:
                avg_price = prices.iloc[-1]
            else:
                avg_price = recent_prices.mean()
        
        else:  # average
            total_divs = divs.sum()
            n_years = (divs.index[-1] - divs.index[0]).days / 365.25
            
            if n_years == 0:
                return np.nan
            
            annual_divs = total_divs / n_years
            avg_price = prices.mean()
            total_divs = annual_divs
        
        if avg_price == 0:
            return np.nan
        
        dy = total_divs / avg_price
        
        return dy
    
    def calculate_dividend_regularity(self, ticker: str) -> Dict[str, float]:
        """
        Calcula índice de regularidade de dividendos (0-100).
        
        Componentes:
        - Consistência: % de meses com pagamento
        - Uniformidade: inverso do CV (coeficiente de variação)
        - Previsibilidade: correlação com média móvel
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Dict com regularity_index e componentes
        """
        if ticker not in self.dividends_dict:
            return {'regularity_index': np.nan}
        
        divs = self.dividends_dict[ticker]
        
        if divs.empty:
            return {'regularity_index': np.nan}
        
        # Reamostrar para mensal
        monthly_divs = divs.resample('M').sum()
        
        # Período de análise (mínimo 12 meses)
        period_months = len(monthly_divs)
        
        if period_months < 12:
            return {'regularity_index': np.nan}
        
        # 1. Consistência (peso 40%)
        months_with_payment = (monthly_divs > 0).sum()
        consistency_score = (months_with_payment / period_months) * 100
        
        # 2. Uniformidade (peso 40%)
        non_zero_divs = monthly_divs[monthly_divs > 0]
        
        if len(non_zero_divs) > 1:
            cv = non_zero_divs.std() / non_zero_divs.mean()
            uniformity_score = max(0, (1 - cv) * 100)
        else:
            uniformity_score = 0
        
        # 3. Previsibilidade (peso 20%)
        if len(monthly_divs) >= 6:
            ma = monthly_divs.rolling(window=3, min_periods=1).mean()
            correlation = monthly_divs.corr(ma)
            predictability_score = max(0, correlation * 100)
        else:
            predictability_score = 0
        
        # Índice composto
        regularity_index = (
            0.4 * consistency_score +
            0.4 * uniformity_score +
            0.2 * predictability_score
        )
        
        return {
            'regularity_index': regularity_index,
            'consistency_score': consistency_score,
            'uniformity_score': uniformity_score,
            'predictability_score': predictability_score,
            'months_with_payment': months_with_payment,
            'total_months': period_months
        }
    
    def calculate_dividend_growth_rate(self, ticker: str, periods: int = 5) -> float:
        """
        Calcula taxa de crescimento anual composta (CAGR) dos dividendos.
        
        Args:
            ticker: Ticker do ativo
            periods: Número de anos a considerar
        
        Returns:
            CAGR dos dividendos (decimal)
        """
        if ticker not in self.dividends_dict:
            return np.nan
        
        divs = self.dividends_dict[ticker]
        
        if divs.empty:
            return np.nan
        
        # Reamostrar para anual
        annual_divs = divs.resample('Y').sum()
        
        if len(annual_divs) < 2:
            return np.nan
        
        # Usar últimos N anos
        recent_divs = annual_divs.tail(periods)
        
        if len(recent_divs) < 2:
            return np.nan
        
        first_value = recent_divs.iloc[0]
        last_value = recent_divs.iloc[-1]
        n_years = len(recent_divs) - 1
        
        if first_value == 0 or n_years == 0:
            return np.nan
        
        cagr = (last_value / first_value) ** (1 / n_years) - 1
        
        return cagr
    
    def calculate_payout_ratio(self, ticker: str, earnings_per_share: float) -> float:
        """
        Calcula payout ratio (dividendos / lucro por ação).
        
        Args:
            ticker: Ticker do ativo
            earnings_per_share: LPA anual
        
        Returns:
            Payout ratio (decimal)
        """
        if ticker not in self.dividends_dict or earnings_per_share == 0:
            return np.nan
        
        divs = self.dividends_dict[ticker]
        
        if divs.empty:
            return np.nan
        
        # Dividendos dos últimos 12 meses
        end_date = divs.index[-1]
        start_date = end_date - pd.DateOffset(months=12)
        recent_divs = divs[divs.index >= start_date]
        
        total_divs = recent_divs.sum()
        
        payout = total_divs / earnings_per_share
        
        return payout
    
    def get_monthly_dividend_calendar(self, ticker: str) -> pd.DataFrame:
        """
        Cria calendário mensal de dividendos.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            DataFrame com dividendos por mês
        """
        if ticker not in self.dividends_dict:
            return pd.DataFrame()
        
        divs = self.dividends_dict[ticker]
        
        if divs.empty:
            return pd.DataFrame()
        
        # Reamostrar para mensal
        monthly = divs.resample('M').sum()
        
        # Criar DataFrame formatado
        calendar = pd.DataFrame({
            'month': monthly.index.strftime('%Y-%m'),
            'dividend': monthly.values
        })
        
        return calendar
    
    def calculate_all_dividend_metrics(self, ticker: str) -> Dict[str, any]:
        """
        Calcula todas as métricas de dividendos para um ticker.
        
        Args:
            ticker: Ticker do ativo
        
        Returns:
            Dicionário com todas as métricas
        """
        regularity = self.calculate_dividend_regularity(ticker)
        
        metrics = {
            'ticker': ticker,
            'dividend_yield': self.calculate_dividend_yield(ticker, 'trailing'),
            'dividend_yield_avg': self.calculate_dividend_yield(ticker, 'average'),
            'regularity_index': regularity.get('regularity_index', np.nan),
            'consistency_score': regularity.get('consistency_score', np.nan),
            'uniformity_score': regularity.get('uniformity_score', np.nan),
            'dividend_growth_rate': self.calculate_dividend_growth_rate(ticker),
            'num_payments': len(self.dividends_dict.get(ticker, [])),
        }
        
        return metrics
    
    def get_portfolio_monthly_dividends(self, weights: Dict[str, float]) -> pd.Series:
        """
        Calcula fluxo mensal de dividendos de um portfólio.
        
        Args:
            weights: Dicionário {ticker: peso}
        
        Returns:
            Series com dividendos mensais do portfólio
        """
        if not weights:
            return pd.Series()
        
        # Coletar todos os dividendos mensais
        all_monthly = []
        
        for ticker, weight in weights.items():
            if ticker in self.dividends_dict:
                divs = self.dividends_dict[ticker]
                monthly = divs.resample('M').sum() * weight
                all_monthly.append(monthly)
        
        if not all_monthly:
            return pd.Series()
        
        # Consolidar
        portfolio_monthly = pd.concat(all_monthly, axis=1).sum(axis=1)
        
        return portfolio_monthly


def create_metrics_summary_table(metrics_list: List[Dict]) -> pd.DataFrame:
    """
    Cria tabela resumo de métricas para múltiplos ativos.
    
    Args:
        metrics_list: Lista de dicionários com métricas
    
    Returns:
        DataFrame formatado
    """
    if not metrics_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_list)
    
    # Ordenar por Sharpe Ratio (descendente)
    if 'sharpe_ratio' in df.columns:
        df = df.sort_values('sharpe_ratio', ascending=False)
    
    return df


def calculate_portfolio_metrics(weights: Dict[str, float], 
                                returns_df: pd.DataFrame,
                                cov_matrix: pd.DataFrame,
                                risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calcula métricas de um portfólio.
    
    Args:
        weights: Dicionário {ticker: peso}
        returns_df: DataFrame de retornos
        cov_matrix: Matriz de covariância anualizada
        risk_free_rate: Taxa livre de risco
    
    Returns:
        Dicionário com métricas do portfólio
    """
    if not weights or returns_df.empty:
        return {}
    
    # Converter pesos para array alinhado
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    
    # Retornos individuais anualizados
    individual_returns = {}
    for ticker in tickers:
        if ticker in returns_df.columns:
            ret = (1 + returns_df[ticker].mean()) ** TRADING_DAYS_YEAR - 1
            individual_returns[ticker] = ret
    
    # Retorno do portfólio
    portfolio_return = sum(weights[t] * individual_returns.get(t, 0) for t in tickers)
    
    # Volatilidade do portfólio
    if not cov_matrix.empty:
        # Alinhar matriz de covariância
        aligned_tickers = [t for t in tickers if t in cov_matrix.columns]
        w_aligned = np.array([weights[t] for t in aligned_tickers])
        cov_aligned = cov_matrix.loc[aligned_tickers, aligned_tickers].values
        
        portfolio_variance = np.dot(w_aligned, np.dot(cov_aligned, w_aligned))
        portfolio_volatility = np.sqrt(portfolio_variance)
    else:
        portfolio_volatility = np.nan
    
    # Sharpe Ratio
    if not np.isnan(portfolio_volatility) and portfolio_volatility > 0:
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    else:
        sharpe = np.nan
    
    metrics = {
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe,
        'num_assets': len(weights),
        'max_weight': max(weights.values()),
        'min_weight': min(weights.values()),
    }
    
    return metrics


def winsorize_returns(returns_df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Aplica winsorização nos retornos para tratar outliers.
    
    Args:
        returns_df: DataFrame de retornos
        lower: Percentil inferior
        upper: Percentil superior
    
    Returns:
        DataFrame winsorizado
    """
    if returns_df.empty:
        return returns_df
    
    winsorized = returns_df.copy()
    
    for col in winsorized.columns:
        lower_bound = winsorized[col].quantile(lower)
        upper_bound = winsorized[col].quantile(upper)
        
        winsorized[col] = winsorized[col].clip(lower=lower_bound, upper=upper_bound)
    
    return winsorized


def calculate_rolling_metrics(prices_df: pd.DataFrame, ticker: str, 
                              window: int = 252, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Calcula métricas em janela móvel.
    
    Args:
        prices_df: DataFrame de preços
        ticker: Ticker do ativo
        window: Tamanho da janela em dias
        risk_free_rate: Taxa livre de risco
    
    Returns:
        DataFrame com métricas ao longo do tempo
    """
    if ticker not in prices_df.columns:
        return pd.DataFrame()
    
    prices = prices_df[ticker].dropna()
    returns = prices.pct_change().dropna()
    
    if len(returns) < window:
        return pd.DataFrame()
    
    # Calcular métricas móveis
    rolling_return = returns.rolling(window).apply(
        lambda x: (1 + x).prod() ** (TRADING_DAYS_YEAR / len(x)) - 1
    )
    
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR)
    
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol
    
    # Consolidar
    rolling_metrics = pd.DataFrame({
        'return': rolling_return,
        'volatility': rolling_vol,
        'sharpe': rolling_sharpe
    }, index=prices.index[window:])
    
    return rolling_metrics


# Funções auxiliares de formatação

def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata valor como percentual."""
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "R$") -> str:
    """Formata valor como moeda."""
    if np.isnan(value):
        return "N/A"
    return f"{currency} {value:,.2f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """Formata ratio."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"
