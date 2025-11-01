"""
core/filters.py
Sistema de filtros e segmentação de ativos por setor, segmento e outros critérios
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class AssetFilter:
    """Classe para filtrar e segmentar ativos."""
    
    def __init__(self, universe_df: pd.DataFrame):
        """
        Inicializa o filtro com o universo de ativos.
        
        Args:
            universe_df: DataFrame com universo de tickers e metadados
        """
        self.universe_df = universe_df.copy()
        self.filtered_df = universe_df.copy()
        self.applied_filters = []
    
    def reset_filters(self):
        """Reseta todos os filtros aplicados."""
        self.filtered_df = self.universe_df.copy()
        self.applied_filters = []
        logger.info("Filtros resetados")
    
    def get_unique_values(self, column: str) -> List[str]:
        """
        Obtém valores únicos de uma coluna.
        
        Args:
            column: Nome da coluna
        
        Returns:
            Lista de valores únicos ordenados
        """
        if column not in self.universe_df.columns:
            logger.warning(f"Coluna {column} não encontrada")
            return []
        
        values = self.universe_df[column].dropna().unique()
        return sorted(values.tolist())
    
    def filter_by_sector(self, sectors: List[str]) -> 'AssetFilter':
        """
        Filtra por setores específicos.
        
        Args:
            sectors: Lista de setores
        
        Returns:
            Self para encadeamento
        """
        if not sectors:
            return self
        
        self.filtered_df = self.filtered_df[self.filtered_df['setor'].isin(sectors)]
        self.applied_filters.append(f"Setores: {', '.join(sectors)}")
        logger.info(f"Filtrado por setores: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_subsector(self, subsectors: List[str]) -> 'AssetFilter':
        """
        Filtra por subsetores específicos.
        
        Args:
            subsectors: Lista de subsetores
        
        Returns:
            Self para encadeamento
        """
        if not subsectors:
            return self
        
        self.filtered_df = self.filtered_df[self.filtered_df['subsetor'].isin(subsectors)]
        self.applied_filters.append(f"Subsetores: {', '.join(subsectors)}")
        logger.info(f"Filtrado por subsetores: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_segment(self, segments: List[str]) -> 'AssetFilter':
        """
        Filtra por segmentos de listagem.
        
        Args:
            segments: Lista de segmentos (Novo Mercado, Nível 1, etc.)
        
        Returns:
            Self para encadeamento
        """
        if not segments:
            return self
        
        self.filtered_df = self.filtered_df[self.filtered_df['segmento_listagem'].isin(segments)]
        self.applied_filters.append(f"Segmentos: {', '.join(segments)}")
        logger.info(f"Filtrado por segmentos: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_type(self, types: List[str]) -> 'AssetFilter':
        """
        Filtra por tipo de ação (ON, PN, UNIT, etc.).
        
        Args:
            types: Lista de tipos
        
        Returns:
            Self para encadeamento
        """
        if not types:
            return self
        
        self.filtered_df = self.filtered_df[self.filtered_df['tipo'].isin(types)]
        self.applied_filters.append(f"Tipos: {', '.join(types)}")
        logger.info(f"Filtrado por tipos: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_liquidity(self, min_volume: float = 0, min_sessions: int = 0) -> 'AssetFilter':
        """
        Filtra por liquidez (requer colunas avg_volume_30d e sessions_traded_30d).
        
        Args:
            min_volume: Volume médio mínimo
            min_sessions: Número mínimo de sessões negociadas
        
        Returns:
            Self para encadeamento
        """
        if 'avg_volume_30d' not in self.filtered_df.columns:
            logger.warning("Coluna avg_volume_30d não encontrada. Execute filter_traded_last_30d primeiro.")
            return self
        
        if min_volume > 0:
            self.filtered_df = self.filtered_df[self.filtered_df['avg_volume_30d'] >= min_volume]
            self.applied_filters.append(f"Volume mínimo: {min_volume:,.0f}")
        
        if min_sessions > 0 and 'sessions_traded_30d' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[self.filtered_df['sessions_traded_30d'] >= min_sessions]
            self.applied_filters.append(f"Sessões mínimas: {min_sessions}")
        
        logger.info(f"Filtrado por liquidez: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_traded_status(self, traded_only: bool = True) -> 'AssetFilter':
        """
        Filtra apenas ativos negociados nos últimos 30 dias.
        
        Args:
            traded_only: Se True, mantém apenas ativos negociados
        
        Returns:
            Self para encadeamento
        """
        if 'is_traded_30d' not in self.filtered_df.columns:
            logger.warning("Coluna is_traded_30d não encontrada. Execute filter_traded_last_30d primeiro.")
            return self
        
        if traded_only:
            self.filtered_df = self.filtered_df[self.filtered_df['is_traded_30d'] == True]
            self.applied_filters.append("Apenas negociados (30d)")
            logger.info(f"Filtrado por negociação: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_tickers(self, tickers: List[str], exclude: bool = False) -> 'AssetFilter':
        """
        Filtra por lista específica de tickers.
        
        Args:
            tickers: Lista de tickers
            exclude: Se True, exclui os tickers; se False, mantém apenas eles
        
        Returns:
            Self para encadeamento
        """
        if not tickers:
            return self
        
        if exclude:
            self.filtered_df = self.filtered_df[~self.filtered_df['ticker'].isin(tickers)]
            self.applied_filters.append(f"Excluídos {len(tickers)} tickers")
        else:
            self.filtered_df = self.filtered_df[self.filtered_df['ticker'].isin(tickers)]
            self.applied_filters.append(f"Selecionados {len(tickers)} tickers específicos")
        
        logger.info(f"Filtrado por tickers: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def filter_by_search(self, search_term: str) -> 'AssetFilter':
        """
        Filtra por termo de busca (nome ou ticker).
        
        Args:
            search_term: Termo a buscar
        
        Returns:
            Self para encadeamento
        """
        if not search_term:
            return self
        
        search_term = search_term.upper()
        mask = (
            self.filtered_df['ticker'].str.contains(search_term, case=False, na=False) |
            self.filtered_df['nome'].str.contains(search_term, case=False, na=False)
        )
        
        self.filtered_df = self.filtered_df[mask]
        self.applied_filters.append(f"Busca: '{search_term}'")
        logger.info(f"Filtrado por busca: {len(self.filtered_df)} ativos restantes")
        
        return self
    
    def get_filtered_tickers(self) -> List[str]:
        """
        Retorna lista de tickers filtrados.
        
        Returns:
            Lista de tickers
        """
        return self.filtered_df['ticker'].tolist()
    
    def get_filtered_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame filtrado.
        
        Returns:
            DataFrame filtrado
        """
        return self.filtered_df.copy()
    
    def get_filter_summary(self) -> Dict[str, any]:
        """
        Retorna resumo dos filtros aplicados.
        
        Returns:
            Dicionário com estatísticas
        """
        summary = {
            'total_universe': len(self.universe_df),
            'total_filtered': len(self.filtered_df),
            'filtered_pct': (len(self.filtered_df) / len(self.universe_df)) * 100 if len(self.universe_df) > 0 else 0,
            'applied_filters': self.applied_filters.copy(),
            'sectors_count': self.filtered_df['setor'].nunique(),
            'subsectors_count': self.filtered_df['subsetor'].nunique(),
        }
        
        return summary


def create_sector_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria distribuição de ativos por setor.
    
    Args:
        df: DataFrame com coluna 'setor'
    
    Returns:
        DataFrame com contagem por setor
    """
    if df.empty or 'setor' not in df.columns:
        return pd.DataFrame()
    
    distribution = df['setor'].value_counts().reset_index()
    distribution.columns = ['setor', 'count']
    distribution['percentage'] = (distribution['count'] / len(df)) * 100
    
    return distribution


def create_subsector_distribution(df: pd.DataFrame, sector: Optional[str] = None) -> pd.DataFrame:
    """
    Cria distribuição de ativos por subsetor.
    
    Args:
        df: DataFrame com coluna 'subsetor'
        sector: Se especificado, filtra por setor
    
    Returns:
        DataFrame com contagem por subsetor
    """
    if df.empty or 'subsetor' not in df.columns:
        return pd.DataFrame()
    
    filtered_df = df if sector is None else df[df['setor'] == sector]
    
    distribution = filtered_df['subsetor'].value_counts().reset_index()
    distribution.columns = ['subsetor', 'count']
    distribution['percentage'] = (distribution['count'] / len(filtered_df)) * 100
    
    if sector:
        distribution['setor'] = sector
    
    return distribution


def create_segment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria distribuição de ativos por segmento de listagem.
    
    Args:
        df: DataFrame com coluna 'segmento_listagem'
    
    Returns:
        DataFrame com contagem por segmento
    """
    if df.empty or 'segmento_listagem' not in df.columns:
        return pd.DataFrame()
    
    distribution = df['segmento_listagem'].value_counts().reset_index()
    distribution.columns = ['segmento', 'count']
    distribution['percentage'] = (distribution['count'] / len(df)) * 100
    
    return distribution


def create_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria distribuição de ativos por tipo.
    
    Args:
        df: DataFrame com coluna 'tipo'
    
    Returns:
        DataFrame com contagem por tipo
    """
    if df.empty or 'tipo' not in df.columns:
        return pd.DataFrame()
    
    distribution = df['tipo'].value_counts().reset_index()
    distribution.columns = ['tipo', 'count']
    distribution['percentage'] = (distribution['count'] / len(df)) * 100
    
    return distribution


def get_top_liquid_tickers(df: pd.DataFrame, n: int = 20) -> List[str]:
    """
    Retorna os N tickers mais líquidos.
    
    Args:
        df: DataFrame com coluna 'avg_volume_30d'
        n: Número de tickers a retornar
    
    Returns:
        Lista de tickers
    """
    if df.empty or 'avg_volume_30d' not in df.columns:
        return []
    
    top_liquid = df.nlargest(n, 'avg_volume_30d')
    return top_liquid['ticker'].tolist()


def get_diversified_selection(df: pd.DataFrame, n_per_sector: int = 3,
                              liquidity_weight: float = 0.7) -> List[str]:
    """
    Seleciona ativos de forma diversificada por setor.
    
    Args:
        df: DataFrame com ativos
        n_per_sector: Número de ativos por setor
        liquidity_weight: Peso da liquidez na seleção (0-1)
    
    Returns:
        Lista de tickers selecionados
    """
    if df.empty or 'setor' not in df.columns:
        return []
    
    selected_tickers = []
    
    # Para cada setor
    for sector in df['setor'].unique():
        sector_df = df[df['setor'] == sector].copy()
        
        if 'avg_volume_30d' in sector_df.columns:
            # Ordenar por liquidez
            sector_df = sector_df.sort_values('avg_volume_30d', ascending=False)
        
        # Selecionar top N do setor
        top_n = sector_df.head(n_per_sector)
        selected_tickers.extend(top_n['ticker'].tolist())
    
    logger.info(f"Seleção diversificada: {len(selected_tickers)} ativos de {df['setor'].nunique()} setores")
    
    return selected_tickers


def get_sector_concentration(tickers: List[str], universe_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula concentração por setor de uma lista de tickers.
    
    Args:
        tickers: Lista de tickers
        universe_df: DataFrame com metadados
    
    Returns:
        Dicionário {setor: percentual}
    """
    if not tickers or universe_df.empty:
        return {}
    
    selected_df = universe_df[universe_df['ticker'].isin(tickers)]
    
    if selected_df.empty or 'setor' not in selected_df.columns:
        return {}
    
    sector_counts = selected_df['setor'].value_counts()
    sector_pct = (sector_counts / len(selected_df)) * 100
    
    return sector_pct.to_dict()


def validate_sector_diversification(tickers: List[str], universe_df: pd.DataFrame,
                                   max_sector_pct: float = 40.0) -> Tuple[bool, Dict[str, float]]:
    """
    Valida se a seleção atende critérios de diversificação setorial.
    
    Args:
        tickers: Lista de tickers
        universe_df: DataFrame com metadados
        max_sector_pct: Percentual máximo permitido por setor
    
    Returns:
        Tuple (is_valid, sector_concentration)
    """
    concentration = get_sector_concentration(tickers, universe_df)
    
    if not concentration:
        return False, {}
    
    max_concentration = max(concentration.values())
    is_valid = max_concentration <= max_sector_pct
    
    if not is_valid:
        logger.warning(f"Concentração setorial excede limite: {max_concentration:.1f}% > {max_sector_pct}%")
    
    return is_valid, concentration


def suggest_additional_tickers(current_tickers: List[str], universe_df: pd.DataFrame,
                               target_count: int = 10, prioritize_sectors: Optional[List[str]] = None) -> List[str]:
    """
    Sugere tickers adicionais para melhorar diversificação.
    
    Args:
        current_tickers: Tickers já selecionados
        universe_df: DataFrame com universo
        target_count: Número alvo de tickers
        prioritize_sectors: Setores a priorizar
    
    Returns:
        Lista de tickers sugeridos
    """
    if universe_df.empty:
        return []
    
    # Remover tickers já selecionados
    available_df = universe_df[~universe_df['ticker'].isin(current_tickers)].copy()
    
    if available_df.empty:
        return []
    
    # Calcular concentração atual
    current_concentration = get_sector_concentration(current_tickers, universe_df)
    
    # Identificar setores sub-representados
    all_sectors = universe_df['setor'].unique()
    underrepresented_sectors = [s for s in all_sectors if current_concentration.get(s, 0) < 10]
    
    if prioritize_sectors:
        underrepresented_sectors = [s for s in underrepresented_sectors if s in prioritize_sectors]
    
    suggested = []
    remaining = target_count - len(current_tickers)
    
    # Selecionar de setores sub-representados
    for sector in underrepresented_sectors:
        if len(suggested) >= remaining:
            break
        
        sector_df = available_df[available_df['setor'] == sector]
        
        if not sector_df.empty:
            # Priorizar por liquidez se disponível
            if 'avg_volume_30d' in sector_df.columns:
                sector_df = sector_df.sort_values('avg_volume_30d', ascending=False)
            
            suggested.extend(sector_df.head(1)['ticker'].tolist())
    
    # Completar com mais líquidos se necessário
    if len(suggested) < remaining:
        if 'avg_volume_30d' in available_df.columns:
            available_df = available_df.sort_values('avg_volume_30d', ascending=False)
        
        additional = available_df[~available_df['ticker'].isin(suggested)].head(remaining - len(suggested))
        suggested.extend(additional['ticker'].tolist())
    
    logger.info(f"Sugeridos {len(suggested)} tickers adicionais")
    
    return suggested[:remaining]


def create_filter_ui(universe_df: pd.DataFrame, key_prefix: str = "") -> AssetFilter:
    """
    Cria interface de filtros no Streamlit.
    
    Args:
        universe_df: DataFrame com universo de ativos
        key_prefix: Prefixo para keys dos widgets
    
    Returns:
        AssetFilter configurado
    """
    asset_filter = AssetFilter(universe_df)
    
    st.markdown("### 🎯 Filtros de Seleção")
    
    # Abas para diferentes tipos de filtro
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Setor/Segmento", "💧 Liquidez", "🔍 Busca", "⚡ Rápidos"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Setores**")
            all_sectors = asset_filter.get_unique_values('setor')
            selected_sectors = st.multiselect(
                "Selecione setores:",
                options=all_sectors,
                key=f"{key_prefix}_sectors",
                help="Filtre por setores específicos da economia"
            )
            
            if selected_sectors:
                asset_filter.filter_by_sector(selected_sectors)
        
        with col2:
            st.markdown("**Subsetores**")
            all_subsectors = asset_filter.get_unique_values('subsetor')
            selected_subsectors = st.multiselect(
                "Selecione subsetores:",
                options=all_subsectors,
                key=f"{key_prefix}_subsectors",
                help="Refine por subsetores específicos"
            )
            
            if selected_subsectors:
                asset_filter.filter_by_subsector(selected_subsectors)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Segmento de Listagem**")
            all_segments = asset_filter.get_unique_values('segmento_listagem')
            selected_segments = st.multiselect(
                "Selecione segmentos:",
                options=all_segments,
                key=f"{key_prefix}_segments",
                help="Novo Mercado, Nível 1, Nível 2, etc."
            )
            
            if selected_segments:
                asset_filter.filter_by_segment(selected_segments)
        
        with col4:
            st.markdown("**Tipo de Ação**")
            all_types = asset_filter.get_unique_values('tipo')
            selected_types = st.multiselect(
                "Selecione tipos:",
                options=all_types,
                key=f"{key_prefix}_types",
                help="ON (Ordinária), PN (Preferencial), UNIT, etc."
            )
            
            if selected_types:
                asset_filter.filter_by_type(selected_types)
    
    with tab2:
        st.markdown("**Critérios de Liquidez**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_volume = st.number_input(
                "Volume médio mínimo (30d):",
                min_value=0,
                value=10000,
                step=10000,
                key=f"{key_prefix}_min_volume",
                help="Volume médio diário mínimo nos últimos 30 dias"
            )
        
        with col2:
            min_sessions = st.number_input(
                "Sessões mínimas negociadas:",
                min_value=0,
                max_value=30,
                value=5,
                step=1,
                key=f"{key_prefix}_min_sessions",
                help="Número mínimo de dias com negociação nos últimos 30 dias"
            )
        
        if min_volume > 0 or min_sessions > 0:
            asset_filter.filter_by_liquidity(min_volume, min_sessions)
        
        only_traded = st.checkbox(
            "Apenas ativos negociados nos últimos 30 dias",
            value=True,
            key=f"{key_prefix}_only_traded",
            help="Filtra apenas ativos com negociação recente"
        )
        
        if only_traded:
            asset_filter.filter_by_traded_status(True)
    
    with tab3:
        st.markdown("**Busca por Nome ou Ticker**")
        
        search_term = st.text_input(
            "Digite para buscar:",
            key=f"{key_prefix}_search",
            placeholder="Ex: PETR, Petrobras, Vale...",
            help="Busca em nomes e tickers"
        )
        
        if search_term:
            asset_filter.filter_by_search(search_term)
    
    with tab4:
        st.markdown("**Seleções Rápidas**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔥 Top 20 Liquidez", key=f"{key_prefix}_top20", use_container_width=True):
                top_tickers = get_top_liquid_tickers(universe_df, 20)
                asset_filter.filter_by_tickers(top_tickers)
                st.success(f"✅ Selecionados {len(top_tickers)} ativos mais líquidos")
        
        with col2:
            if st.button("🎲 Diversificado", key=f"{key_prefix}_diversified", use_container_width=True):
                diversified_tickers = get_diversified_selection(universe_df, n_per_sector=3)
                asset_filter.filter_by_tickers(diversified_tickers)
                st.success(f"✅ Selecionados {len(diversified_tickers)} ativos diversificados")
        
        with col3:
            if st.button("🔄 Limpar Filtros", key=f"{key_prefix}_reset", use_container_width=True):
                asset_filter.reset_filters()
                st.rerun()
    
    # Resumo dos filtros
    summary = asset_filter.get_filter_summary()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total no Universo", f"{summary['total_universe']}")
    
    with col2:
        st.metric("Após Filtros", f"{summary['total_filtered']}")
    
    with col3:
        st.metric("% Mantido", f"{summary['filtered_pct']:.1f}%")
    
    if summary['applied_filters']:
        with st.expander("📋 Filtros Aplicados", expanded=False):
            for filter_desc in summary['applied_filters']:
                st.text(f"• {filter_desc}")
    
    return asset_filter


def export_filtered_tickers(asset_filter: AssetFilter, filename: str = "filtered_tickers.csv") -> bool:
    """
    Exporta tickers filtrados para CSV.
    
    Args:
        asset_filter: Filtro aplicado
        filename: Nome do arquivo
    
    Returns:
        True se sucesso
    """
    try:
        df = asset_filter.get_filtered_dataframe()
        df.to_csv(filename, index=False)
        logger.info(f"Tickers filtrados exportados para {filename}")
        return True
    except Exception as e:
        logger.error(f"Erro ao exportar tickers: {e}")
        return False
