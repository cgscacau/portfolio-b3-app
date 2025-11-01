"""
core/opt.py
Otimização de portfólios: Markowitz, Sharpe Máximo, Mínima Volatilidade e Dividendos Regulares
"""

import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

TRADING_DAYS_YEAR = 252


class PortfolioOptimizer:
    """Classe base para otimização de portfólios."""
    
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.0):
        """
        Inicializa o otimizador.
        
        Args:
            expected_returns: Series com retornos esperados anualizados
            cov_matrix: Matriz de covariância anualizada
            risk_free_rate: Taxa livre de risco anual
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.tickers = expected_returns.index.tolist()
        self.n_assets = len(self.tickers)
        
        # Validar dados
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Valida consistência dos dados de entrada."""
        if self.expected_returns.empty or self.cov_matrix.empty:
            raise ValueError("Retornos esperados ou matriz de covariância vazios")
        
        if len(self.expected_returns) != len(self.cov_matrix):
            raise ValueError("Dimensões incompatíveis entre retornos e covariância")
        
        if not all(self.expected_returns.index == self.cov_matrix.index):
            raise ValueError("Índices não correspondem entre retornos e covariância")
        
        # Verificar se matriz é positiva semi-definida
        eigenvalues = np.linalg.eigvals(self.cov_matrix.values)
        if np.any(eigenvalues < -1e-8):
            logger.warning("Matriz de covariância não é positiva semi-definida. Aplicando correção.")
            self.cov_matrix = self._nearest_psd(self.cov_matrix)
    
    def _nearest_psd(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Encontra a matriz positiva semi-definida mais próxima.
        
        Args:
            matrix: Matriz original
        
        Returns:
            Matriz corrigida
        """
        # Decomposição espectral
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.values)
        
        # Zerar autovalores negativos
        eigenvalues[eigenvalues < 0] = 1e-8
        
        # Reconstruir matriz
        corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return pd.DataFrame(corrected, index=matrix.index, columns=matrix.columns)
    
    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calcula retorno e volatilidade do portfólio.
        
        Args:
            weights: Array de pesos
        
        Returns:
            Tuple (retorno, volatilidade)
        """
        portfolio_return = np.dot(weights, self.expected_returns.values)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def _sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calcula Sharpe ratio do portfólio.
        
        Args:
            weights: Array de pesos
        
        Returns:
            Sharpe ratio (negativo para minimização)
        """
        ret, vol = self._portfolio_performance(weights)
        
        if vol == 0:
            return -np.inf
        
        sharpe = (ret - self.risk_free_rate) / vol
        
        return -sharpe  # Negativo para minimização


class MarkowitzOptimizer(PortfolioOptimizer):
    """Otimizador de Markowitz para fronteira eficiente."""
    
    def optimize_for_return(self, target_return: float, 
                           max_weight: float = 1.0,
                           min_weight: float = 0.0,
                           sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Otimiza portfólio para retorno alvo (minimiza risco).
        
        Args:
            target_return: Retorno alvo anualizado
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            sector_constraints: Dict {setor: (lista_tickers, peso_max)}
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Variáveis de decisão
        weights = cp.Variable(self.n_assets)
        
        # Função objetivo: minimizar variância
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Restrições
        constraints = [
            cp.sum(weights) == 1,  # Pesos somam 1
            weights >= min_weight,  # Peso mínimo
            weights <= max_weight,  # Peso máximo
            self.expected_returns.values @ weights >= target_return  # Retorno alvo
        ]
        
        # Restrições setoriais
        if sector_constraints:
            for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
                sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
                if sector_indices:
                    constraints.append(cp.sum(weights[sector_indices]) <= max_sector_weight)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=1000)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização falhou: {problem.status}")
                return {}
            
            # Extrair pesos
            optimal_weights = weights.value
            
            # Filtrar pesos muito pequenos
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização: {e}")
            return {}
    
    def optimize_for_risk(self, target_volatility: float,
                         max_weight: float = 1.0,
                         min_weight: float = 0.0,
                         sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Otimiza portfólio para volatilidade alvo (maximiza retorno).
        
        Args:
            target_volatility: Volatilidade alvo anualizada
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            sector_constraints: Dict {setor: (lista_tickers, peso_max)}
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Variáveis de decisão
        weights = cp.Variable(self.n_assets)
        
        # Função objetivo: maximizar retorno
        portfolio_return = self.expected_returns.values @ weights
        objective = cp.Maximize(portfolio_return)
        
        # Restrições
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight,
            portfolio_variance <= target_volatility ** 2  # Volatilidade alvo
        ]
        
        # Restrições setoriais
        if sector_constraints:
            for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
                sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
                if sector_indices:
                    constraints.append(cp.sum(weights[sector_indices]) <= max_sector_weight)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=1000)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização falhou: {problem.status}")
                return {}
            
            optimal_weights = weights.value
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização: {e}")
            return {}
    
    def compute_efficient_frontier(self, n_points: int = 50,
                                  max_weight: float = 1.0,
                                  min_weight: float = 0.0) -> pd.DataFrame:
        """
        Computa fronteira eficiente.
        
        Args:
            n_points: Número de pontos na fronteira
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
        
        Returns:
            DataFrame com retorno, volatilidade e Sharpe de cada ponto
        """
        # Range de retornos
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, target_ret in enumerate(target_returns):
            status_text.text(f"Calculando fronteira eficiente: {idx+1}/{n_points}")
            
            weights = self.optimize_for_return(target_ret, max_weight, min_weight)
            
            if weights:
                ret, vol = self._portfolio_performance(np.array([weights.get(t, 0) for t in self.tickers]))
                sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else np.nan
                
                frontier_results.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
            
            progress_bar.progress((idx + 1) / n_points)
        
        progress_bar.empty()
        status_text.empty()
        
        if not frontier_results:
            return pd.DataFrame()
        
        frontier_df = pd.DataFrame(frontier_results)
        
        return frontier_df


class MaxSharpeOptimizer(PortfolioOptimizer):
    """Otimizador para máximo Sharpe ratio."""
    
    def optimize(self, max_weight: float = 1.0,
                min_weight: float = 0.0,
                sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Otimiza para máximo Sharpe ratio.
        
        Args:
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            sector_constraints: Dict {setor: (lista_tickers, peso_max)}
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Usar scipy.optimize para Sharpe
        def negative_sharpe(weights):
            return self._sharpe_ratio(weights)
        
        # Restrições
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Soma = 1
        ]
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Ponto inicial (equal weight)
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Otimizar
        try:
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning(f"Otimização Sharpe: {result.message}")
            
            optimal_weights = result.x
            
            # Aplicar restrições setoriais se necessário (pós-otimização)
            if sector_constraints:
                optimal_weights = self._apply_sector_constraints(
                    optimal_weights, sector_constraints, max_weight
                )
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização Sharpe: {e}")
            return {}
    
    def _apply_sector_constraints(self, weights: np.ndarray,
                                  sector_constraints: Dict[str, Tuple[List[str], float]],
                                  max_weight: float) -> np.ndarray:
        """
        Aplica restrições setoriais ajustando pesos.
        
        Args:
            weights: Array de pesos
            sector_constraints: Restrições setoriais
            max_weight: Peso máximo por ativo
        
        Returns:
            Array de pesos ajustados
        """
        adjusted = weights.copy()
        
        for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
            sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
            
            if sector_indices:
                sector_total = adjusted[sector_indices].sum()
                
                if sector_total > max_sector_weight:
                    # Escalar proporcionalmente
                    scale_factor = max_sector_weight / sector_total
                    adjusted[sector_indices] *= scale_factor
        
        # Renormalizar
        adjusted = adjusted / adjusted.sum()
        
        # Aplicar max_weight individual
        adjusted = np.minimum(adjusted, max_weight)
        adjusted = adjusted / adjusted.sum()
        
        return adjusted


class MinVolatilityOptimizer(PortfolioOptimizer):
    """Otimizador para mínima volatilidade."""
    
    def optimize(self, max_weight: float = 1.0,
                min_weight: float = 0.0,
                sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Otimiza para mínima volatilidade.
        
        Args:
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            sector_constraints: Dict {setor: (lista_tickers, peso_max)}
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Variáveis de decisão
        weights = cp.Variable(self.n_assets)
        
        # Função objetivo: minimizar variância
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Restrições
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight
        ]
        
        # Restrições setoriais
        if sector_constraints:
            for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
                sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
                if sector_indices:
                    constraints.append(cp.sum(weights[sector_indices]) <= max_sector_weight)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=1000)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização MinVol falhou: {problem.status}")
                return {}
            
            optimal_weights = weights.value
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização MinVol: {e}")
            return {}


class DividendRegularityOptimizer:
    """Otimizador para dividendos regulares."""
    
    def __init__(self, expected_monthly_divs: pd.Series,
                 div_covariance: pd.DataFrame,
                 expected_returns: pd.Series,
                 price_cov_matrix: pd.DataFrame):
        """
        Inicializa otimizador de dividendos.
        
        Args:
            expected_monthly_divs: Series com dividend yield mensal médio por ativo
            div_covariance: Matriz de covariância dos fluxos mensais de dividendos
            expected_returns: Series com retornos esperados (para validação)
            price_cov_matrix: Matriz de covariância de preços (para risco)
        """
        self.expected_monthly_divs = expected_monthly_divs
        self.div_covariance = div_covariance
        self.expected_returns = expected_returns
        self.price_cov_matrix = price_cov_matrix
        self.tickers = expected_monthly_divs.index.tolist()
        self.n_assets = len(self.tickers)
    
    def optimize(self, lambda_penalty: float = 0.5,
                max_weight: float = 1.0,
                min_weight: float = 0.0,
                min_yield: Optional[float] = None,
                sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Otimiza para dividendos regulares.
        
        Objetivo: Maximizar yield mensal - λ * variância mensal
        
        Args:
            lambda_penalty: Penalização da variância (0-1)
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            min_yield: Yield mínimo do portfólio
            sector_constraints: Restrições setoriais
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Variáveis de decisão
        weights = cp.Variable(self.n_assets)
        
        # Função objetivo
        expected_yield = self.expected_monthly_divs.values @ weights
        portfolio_variance = cp.quad_form(weights, self.div_covariance.values)
        
        objective = cp.Maximize(expected_yield - lambda_penalty * portfolio_variance)
        
        # Restrições
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight
        ]
        
        # Restrição de yield mínimo
        if min_yield is not None:
            constraints.append(expected_yield >= min_yield)
        
        # Restrições setoriais
        if sector_constraints:
            for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
                sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
                if sector_indices:
                    constraints.append(cp.sum(weights[sector_indices]) <= max_sector_weight)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=1000)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização Dividendos falhou: {problem.status}")
                return {}
            
            optimal_weights = weights.value
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização Dividendos: {e}")
            return {}
    
    def optimize_max_yield_with_risk_limit(self, max_volatility: float,
                                          max_weight: float = 1.0,
                                          min_weight: float = 0.0,
                                          sector_constraints: Optional[Dict[str, Tuple[List[str], float]]] = None) -> Dict[str, float]:
        """
        Maximiza dividend yield com limite de volatilidade de preços.
        
        Args:
            max_volatility: Volatilidade máxima permitida (anualizada)
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
            sector_constraints: Restrições setoriais
        
        Returns:
            Dicionário {ticker: peso}
        """
        # Variáveis de decisão
        weights = cp.Variable(self.n_assets)
        
        # Função objetivo: maximizar yield
        expected_yield = self.expected_monthly_divs.values @ weights
        objective = cp.Maximize(expected_yield)
        
        # Restrições
        portfolio_variance = cp.quad_form(weights, self.price_cov_matrix.values)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight,
            portfolio_variance <= max_volatility ** 2  # Risco limitado
        ]
        
        # Restrições setoriais
        if sector_constraints:
            for sector, (sector_tickers, max_sector_weight) in sector_constraints.items():
                sector_indices = [i for i, t in enumerate(self.tickers) if t in sector_tickers]
                if sector_indices:
                    constraints.append(cp.sum(weights[sector_indices]) <= max_sector_weight)
        
        # Resolver
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, max_iters=1000)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.error(f"Otimização MaxYield falhou: {problem.status}")
                return {}
            
            optimal_weights = weights.value
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização MaxYield: {e}")
            return {}


class EqualWeightOptimizer:
    """Otimizador para carteira equally weighted (baseline)."""
    
    def __init__(self, tickers: List[str]):
        """
        Inicializa com lista de tickers.
        
        Args:
            tickers: Lista de tickers
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
    
    def optimize(self) -> Dict[str, float]:
        """
        Retorna pesos iguais.
        
        Returns:
            Dicionário {ticker: peso}
        """
        if self.n_assets == 0:
            return {}
        
        weight = 1.0 / self.n_assets
        
        return {ticker: weight for ticker in self.tickers}


class RiskParityOptimizer(PortfolioOptimizer):
    """Otimizador para Risk Parity (contribuição igual de risco)."""
    
    def optimize(self, max_weight: float = 1.0,
                min_weight: float = 0.0) -> Dict[str, float]:
        """
        Otimiza para risk parity.
        
        Args:
            max_weight: Peso máximo por ativo
            min_weight: Peso mínimo por ativo
        
        Returns:
            Dicionário {ticker: peso}
        """
        def risk_parity_objective(weights):
            """Minimiza diferença nas contribuições de risco."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))
            
            # Contribuição marginal de risco
            marginal_contrib = np.dot(self.cov_matrix.values, weights) / portfolio_vol
            
            # Contribuição de risco
            risk_contrib = weights * marginal_contrib
            
            # Minimizar variância das contribuições
            target_contrib = portfolio_vol / self.n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Restrições
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Ponto inicial (inverse volatility)
        vols = np.sqrt(np.diag(self.cov_matrix.values))
        initial_weights = (1 / vols) / (1 / vols).sum()
        
        # Otimizar
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning(f"Otimização Risk Parity: {result.message}")
            
            optimal_weights = result.x
            
            weights_dict = {
                ticker: float(w) for ticker, w in zip(self.tickers, optimal_weights)
                if w > 1e-4
            }
            
            return weights_dict
        
        except Exception as e:
            logger.error(f"Erro na otimização Risk Parity: {e}")
            return {}


def create_sector_constraints(universe_df: pd.DataFrame,
                              tickers: List[str],
                              max_sector_weight: float = 0.40) -> Dict[str, Tuple[List[str], float]]:
    """
    Cria dicionário de restrições setoriais.
    
    Args:
        universe_df: DataFrame com metadados dos ativos
        tickers: Lista de tickers no portfólio
        max_sector_weight: Peso máximo por setor
    
    Returns:
        Dict {setor: (lista_tickers, peso_max)}
    """
    if universe_df.empty or 'setor' not in universe_df.columns:
        return {}
    
    # Filtrar apenas tickers relevantes
    relevant = universe_df[universe_df['ticker'].isin(tickers)]
    
    sector_constraints = {}
    
    for sector in relevant['setor'].unique():
        sector_tickers = relevant[relevant['setor'] == sector]['ticker'].tolist()
        sector_constraints[sector] = (sector_tickers, max_sector_weight)
    
    return sector_constraints


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza pesos para somarem 1.
    
    Args:
        weights: Dicionário de pesos
    
    Returns:
        Dicionário normalizado
    """
    if not weights:
        return {}
    
    total = sum(weights.values())
    
    if total == 0:
        return weights
    
    return {ticker: w / total for ticker, w in weights.items()}


def filter_small_weights(weights: Dict[str, float], threshold: float = 0.01) -> Dict[str, float]:
    """
    Remove pesos menores que threshold e renormaliza.
    
    Args:
        weights: Dicionário de pesos
        threshold: Threshold mínimo
    
    Returns:
        Dicionário filtrado e normalizado
    """
    filtered = {ticker: w for ticker, w in weights.items() if w >= threshold}
    
    return normalize_weights(filtered)


def calculate_portfolio_stats(weights: Dict[str, float],
                              expected_returns: pd.Series,
                              cov_matrix: pd.DataFrame,
                              risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calcula estatísticas de um portfólio otimizado.
    
    Args:
        weights: Dicionário de pesos
        expected_returns: Series de retornos esperados
        cov_matrix: Matriz de covariância
        risk_free_rate: Taxa livre de risco
    
    Returns:
        Dicionário com estatísticas
    """
    if not weights:
        return {}
    
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    
    # Alinhar com expected_returns e cov_matrix
    aligned_tickers = [t for t in tickers if t in expected_returns.index and t in cov_matrix.columns]
    
    if not aligned_tickers:
        return {}
    
    w_aligned = np.array([weights[t] for t in aligned_tickers])
    w_aligned = w_aligned / w_aligned.sum()  # Renormalizar
    
    # Retorno
    portfolio_return = np.dot(w_aligned, expected_returns[aligned_tickers].values)
    
    # Volatilidade
    cov_aligned = cov_matrix.loc[aligned_tickers, aligned_tickers].values
    portfolio_variance = np.dot(w_aligned, np.dot(cov_aligned, w_aligned))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else np.nan
    
    stats = {
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe,
        'num_assets': len(weights),
        'max_weight': max(weights.values()),
        'min_weight': min(weights.values()),
        'effective_n': 1 / np.sum(w_aligned ** 2),  # Número efetivo de ativos
    }
    
    return stats


def compare_portfolios(portfolios: Dict[str, Dict[str, float]],
                      expected_returns: pd.Series,
                      cov_matrix: pd.DataFrame,
                      risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compara múltiplos portfólios.
    
    Args:
        portfolios: Dict {nome_portfolio: weights_dict}
        expected_returns: Series de retornos esperados
        cov_matrix: Matriz de covariância
        risk_free_rate: Taxa livre de risco
    
    Returns:
        DataFrame comparativo
    """
    comparison = []
    
    for name, weights in portfolios.items():
        stats = calculate_portfolio_stats(weights, expected_returns, cov_matrix, risk_free_rate)
        stats['portfolio'] = name
        comparison.append(stats)
    
    if not comparison:
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison)
    df = df.set_index('portfolio')
    
    return df


def backtest_portfolio(weights: Dict[str, float],
                      prices_df: pd.DataFrame,
                      rebalance_frequency: str = 'M') -> pd.Series:
    """
    Simples backtest de um portfólio.
    
    Args:
        weights: Dicionário de pesos
        prices_df: DataFrame de preços históricos
        rebalance_frequency: 'D', 'W', 'M', 'Q', 'Y'
    
    Returns:
        Series com valor do portfólio ao longo do tempo
    """
    if not weights or prices_df.empty:
        return pd.Series()
    
    # Filtrar apenas tickers com pesos
    tickers = list(weights.keys())
    portfolio_prices = prices_df[tickers].copy()
    
    # Retornos
    returns = portfolio_prices.pct_change().fillna(0)
    
    # Valor inicial
    initial_value = 1.0
    portfolio_value = [initial_value]
    
    # Simular
    for date, row in returns.iterrows():
        daily_return = sum(weights[ticker] * row[ticker] for ticker in tickers)
        new_value = portfolio_value[-1] * (1 + daily_return)
        portfolio_value.append(new_value)
    
    # Criar série
    portfolio_series = pd.Series(portfolio_value[1:], index=returns.index)
    
    return portfolio_series
