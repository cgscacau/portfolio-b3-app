"""
Módulo de gerenciamento de portfólios
Permite criar, salvar, carregar e comparar múltiplos portfólios
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURAÇÕES
# ==========================================

PORTFOLIOS_DIR = "data/portfolios"
PORTFOLIOS_FILE = os.path.join(PORTFOLIOS_DIR, "portfolios.json")


# ==========================================
# INICIALIZAÇÃO GLOBAL
# ==========================================

def _garantir_session_state():
    """Garante que session_state está inicializado"""
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
    
    if 'portfolio_ativo' not in st.session_state:
        st.session_state.portfolio_ativo = None


def _garantir_diretorio():
    """Garante que diretório existe"""
    os.makedirs(PORTFOLIOS_DIR, exist_ok=True)


# ==========================================
# CLASSE PORTFOLIO
# ==========================================

class Portfolio:
    """Representa um portfólio de investimentos"""
    
    def __init__(
        self,
        nome: str,
        tickers: List[str],
        pesos: List[float],
        data_inicio: datetime,
        data_fim: datetime,
        descricao: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa um portfólio
        
        Args:
            nome: Nome do portfólio
            tickers: Lista de códigos dos ativos
            pesos: Lista de pesos (devem somar 1.0 ou 100)
            data_inicio: Data inicial de análise
            data_fim: Data final de análise
            descricao: Descrição opcional
            metadata: Metadados adicionais
        """
        self.nome = nome
        self.tickers = tickers
        self.pesos = pesos
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.descricao = descricao
        self.metadata = metadata or {}
        self.criado_em = datetime.now()
        self.modificado_em = datetime.now()
        
        # Validar
        self._validar()
    
    def _validar(self):
        """Valida dados do portfólio"""
        if not self.nome:
            raise ValueError("Nome do portfólio é obrigatório")
        
        if not self.tickers:
            raise ValueError("Lista de tickers vazia")
        
        if len(self.tickers) != len(self.pesos):
            raise ValueError("Número de tickers diferente de número de pesos")
        
        # Normalizar pesos (se soma 100, converter para 1.0)
        soma_pesos = sum(self.pesos)
        if abs(soma_pesos - 100) < 0.01:
            self.pesos = [p / 100 for p in self.pesos]
            soma_pesos = sum(self.pesos)
        
        if abs(soma_pesos - 1.0) > 0.01:
            raise ValueError(f"Soma dos pesos deve ser 1.0 ou 100 (atual: {soma_pesos})")
        
        if self.data_inicio >= self.data_fim:
            raise ValueError("Data inicial deve ser anterior à data final")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'nome': self.nome,
            'tickers': self.tickers,
            'pesos': self.pesos,
            'data_inicio': self.data_inicio.isoformat(),
            'data_fim': self.data_fim.isoformat(),
            'descricao': self.descricao,
            'metadata': self.metadata,
            'criado_em': self.criado_em.isoformat(),
            'modificado_em': self.modificado_em.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Cria portfólio a partir de dicionário"""
        portfolio = cls(
            nome=data['nome'],
            tickers=data['tickers'],
            pesos=data['pesos'],
            data_inicio=datetime.fromisoformat(data['data_inicio']),
            data_fim=datetime.fromisoformat(data['data_fim']),
            descricao=data.get('descricao', ''),
            metadata=data.get('metadata', {})
        )
        
        if 'criado_em' in data:
            portfolio.criado_em = datetime.fromisoformat(data['criado_em'])
        if 'modificado_em' in data:
            portfolio.modificado_em = datetime.fromisoformat(data['modificado_em'])
        
        return portfolio
    
    def atualizar(
        self,
        tickers: Optional[List[str]] = None,
        pesos: Optional[List[float]] = None,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
        descricao: Optional[str] = None
    ):
        """Atualiza dados do portfólio"""
        if tickers is not None:
            self.tickers = tickers
        if pesos is not None:
            self.pesos = pesos
        if data_inicio is not None:
            self.data_inicio = data_inicio
        if data_fim is not None:
            self.data_fim = data_fim
        if descricao is not None:
            self.descricao = descricao
        
        self.modificado_em = datetime.now()
        self._validar()
    
    def __repr__(self) -> str:
        return f"Portfolio(nome='{self.nome}', ativos={len(self.tickers)})"


# ==========================================
# FUNÇÕES DE ARQUIVO
# ==========================================

def _carregar_arquivo() -> Dict[str, Dict]:
    """Carrega portfólios do arquivo"""
    _garantir_diretorio()
    
    if not os.path.exists(PORTFOLIOS_FILE):
        return {}
    
    try:
        with open(PORTFOLIOS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {str(e)}")
        return {}


def _salvar_arquivo(portfolios_dict: Dict[str, Dict]) -> bool:
    """Salva portfólios no arquivo"""
    _garantir_diretorio()
    
    try:
        with open(PORTFOLIOS_FILE, 'w', encoding='utf-8') as f:
            json.dump(portfolios_dict, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo: {str(e)}")
        return False


# ==========================================
# GERENCIADOR DE PORTFÓLIOS
# ==========================================

class PortfolioManager:
    """Gerenciador de portfólios"""
    
    def __init__(self):
        """Inicializa o gerenciador"""
        _garantir_session_state()
        _garantir_diretorio()
    
    # ==========================================
    # OPERAÇÕES CRUD
    # ==========================================
    
    def criar(self, portfolio: Portfolio) -> bool:
        """
        Cria novo portfólio
        
        Args:
            portfolio: Objeto Portfolio
            
        Returns:
            True se criado com sucesso
        """
        _garantir_session_state()
        
        if portfolio.nome in st.session_state.portfolios:
            logger.warning(f"Portfólio '{portfolio.nome}' já existe")
            return False
        
        st.session_state.portfolios[portfolio.nome] = portfolio
        logger.info(f"Portfólio '{portfolio.nome}' criado")
        return True
    
    def salvar(self, nome: str) -> bool:
        """
        Salva portfólio em arquivo
        
        Args:
            nome: Nome do portfólio
            
        Returns:
            True se salvo com sucesso
        """
        _garantir_session_state()
        
        if nome not in st.session_state.portfolios:
            logger.error(f"Portfólio '{nome}' não encontrado na sessão")
            return False
        
        # Carregar portfólios existentes
        portfolios_salvos = _carregar_arquivo()
        
        # Adicionar/atualizar
        portfolio = st.session_state.portfolios[nome]
        portfolios_salvos[nome] = portfolio.to_dict()
        
        # Salvar arquivo
        if _salvar_arquivo(portfolios_salvos):
            logger.info(f"Portfólio '{nome}' salvo em arquivo")
            return True
        
        return False
    
    def carregar(self, nome: str) -> Optional[Portfolio]:
        """
        Carrega portfólio (da sessão ou arquivo)
        
        Args:
            nome: Nome do portfólio
            
        Returns:
            Portfolio ou None
        """
        _garantir_session_state()
        
        # Tentar carregar da sessão primeiro
        if nome in st.session_state.portfolios:
            return st.session_state.portfolios[nome]
        
        # Tentar carregar do arquivo
        portfolios_salvos = _carregar_arquivo()
        if nome in portfolios_salvos:
            try:
                portfolio = Portfolio.from_dict(portfolios_salvos[nome])
                st.session_state.portfolios[nome] = portfolio
                return portfolio
            except Exception as e:
                logger.error(f"Erro ao criar Portfolio de '{nome}': {str(e)}")
                return None
        
        logger.warning(f"Portfólio '{nome}' não encontrado")
        return None
    
    def deletar(self, nome: str, deletar_arquivo: bool = True) -> bool:
        """
        Deleta portfólio
        
        Args:
            nome: Nome do portfólio
            deletar_arquivo: Se True, deleta também do arquivo
            
        Returns:
            True se deletado com sucesso
        """
        _garantir_session_state()
        
        # Deletar da sessão
        if nome in st.session_state.portfolios:
            del st.session_state.portfolios[nome]
            logger.info(f"Portfólio '{nome}' removido da sessão")
        
        # Deletar do arquivo
        if deletar_arquivo:
            portfolios_salvos = _carregar_arquivo()
            if nome in portfolios_salvos:
                del portfolios_salvos[nome]
                _salvar_arquivo(portfolios_salvos)
                logger.info(f"Portfólio '{nome}' removido do arquivo")
        
        # Se era o ativo, limpar
        if st.session_state.portfolio_ativo == nome:
            st.session_state.portfolio_ativo = None
        
        return True
    
    def listar(self, incluir_arquivo: bool = True) -> List[str]:
        """
        Lista todos os portfólios
        
        Args:
            incluir_arquivo: Se True, inclui portfólios salvos em arquivo
            
        Returns:
            Lista de nomes de portfólios
        """
        _garantir_session_state()
        
        nomes = set(st.session_state.portfolios.keys())
        
        if incluir_arquivo:
            portfolios_salvos = _carregar_arquivo()
            nomes.update(portfolios_salvos.keys())
        
        return sorted(list(nomes))
    
    def obter_detalhes(self, nome: str) -> Optional[Dict[str, Any]]:
        """
        Obtém detalhes de um portfólio
        
        Args:
            nome: Nome do portfólio
            
        Returns:
            Dicionário com detalhes ou None
        """
        portfolio = self.carregar(nome)
        if portfolio:
            return portfolio.to_dict()
        return None
    
    # ==========================================
    # PORTFÓLIO ATIVO
    # ==========================================
    
    def definir_ativo(self, nome: str) -> bool:
        """
        Define portfólio ativo
        
        Args:
            nome: Nome do portfólio
            
        Returns:
            True se definido com sucesso
        """
        _garantir_session_state()
        
        portfolio = self.carregar(nome)
        if portfolio:
            st.session_state.portfolio_ativo = nome
            logger.info(f"Portfólio ativo: '{nome}'")
            return True
        return False
    
    def obter_ativo(self) -> Optional[Portfolio]:
        """
        Obtém portfólio ativo
        
        Returns:
            Portfolio ativo ou None
        """
        _garantir_session_state()
        
        if st.session_state.portfolio_ativo:
            return self.carregar(st.session_state.portfolio_ativo)
        return None
    
    # ==========================================
    # OPERAÇÕES EM LOTE
    # ==========================================
    
    def salvar_todos(self) -> bool:
        """Salva todos os portfólios da sessão em arquivo"""
        _garantir_session_state()
        
        portfolios_dict = {}
        for nome, portfolio in st.session_state.portfolios.items():
            portfolios_dict[nome] = portfolio.to_dict()
        
        if _salvar_arquivo(portfolios_dict):
            logger.info(f"{len(portfolios_dict)} portfólios salvos")
            return True
        
        return False
    
    def carregar_todos(self) -> bool:
        """Carrega todos os portfólios do arquivo para sessão"""
        _garantir_session_state()
        
        portfolios_salvos = _carregar_arquivo()
        
        for nome, data in portfolios_salvos.items():
            try:
                portfolio = Portfolio.from_dict(data)
                st.session_state.portfolios[nome] = portfolio
            except Exception as e:
                logger.error(f"Erro ao carregar '{nome}': {str(e)}")
        
        logger.info(f"{len(portfolios_salvos)} portfólios carregados")
        return True
    
    # ==========================================
    # COMPARAÇÃO
    # ==========================================
    
    def comparar(self, nomes: List[str]) -> pd.DataFrame:
        """
        Compara múltiplos portfólios
        
        Args:
            nomes: Lista de nomes de portfólios
            
        Returns:
            DataFrame com comparação
        """
        dados = []
        
        for nome in nomes:
            portfolio = self.carregar(nome)
            if portfolio:
                dados.append({
                    'Nome': portfolio.nome,
                    'Ativos': len(portfolio.tickers),
                    'Período': f"{portfolio.data_inicio.date()} - {portfolio.data_fim.date()}",
                    'Criado em': portfolio.criado_em.strftime('%d/%m/%Y %H:%M'),
                    'Modificado em': portfolio.modificado_em.strftime('%d/%m/%Y %H:%M')
                })
        
        return pd.DataFrame(dados)


# ==========================================
# INSTÂNCIA GLOBAL
# ==========================================

portfolio_manager = PortfolioManager()


# ==========================================
# FUNÇÕES PÚBLICAS
# ==========================================

def criar_portfolio(
    nome: str,
    tickers: List[str],
    pesos: List[float],
    data_inicio: datetime,
    data_fim: datetime,
    descricao: str = ""
) -> bool:
    """Cria novo portfólio"""
    try:
        portfolio = Portfolio(nome, tickers, pesos, data_inicio, data_fim, descricao)
        return portfolio_manager.criar(portfolio)
    except Exception as e:
        logger.error(f"Erro ao criar portfólio: {str(e)}")
        return False


def salvar_portfolio(nome: str) -> bool:
    """Salva portfólio em arquivo"""
    return portfolio_manager.salvar(nome)


def carregar_portfolio(nome: str) -> Optional[Portfolio]:
    """Carrega portfólio"""
    return portfolio_manager.carregar(nome)


def deletar_portfolio(nome: str, deletar_arquivo: bool = True) -> bool:
    """Deleta portfólio"""
    return portfolio_manager.deletar(nome, deletar_arquivo)


def listar_portfolios(incluir_arquivo: bool = True) -> List[str]:
    """Lista portfólios"""
    return portfolio_manager.listar(incluir_arquivo)


def definir_portfolio_ativo(nome: str) -> bool:
    """Define portfólio ativo"""
    return portfolio_manager.definir_ativo(nome)


def obter_portfolio_ativo() -> Optional[Portfolio]:
    """Obtém portfólio ativo"""
    return portfolio_manager.obter_ativo()
