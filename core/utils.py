"""
core/utils.py
Funções utilitárias e helpers
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)


def get_use_mock_flag() -> bool:
    """
    Retorna o estado do flag use_mock de forma segura.
    
    Returns:
        bool: True se deve usar dados simulados
    """
    return st.session_state.get('use_mock_data', False)


def set_use_mock_flag(value: bool):
    """
    Define o flag use_mock.
    
    Args:
        value: True para usar dados simulados
    """
    st.session_state.use_mock_data = value
    logger.info(f"Modo de dados alterado: {'Simulado' if value else 'Real'}")


def show_data_mode_indicator():
    """
    Mostra indicador visual do modo de dados atual.
    """
    use_mock = get_use_mock_flag()
    
    if use_mock:
        st.info("🎲 **Modo Simulado Ativo** - Os dados são gerados aleatoriamente para demonstração e testes")
    else:
        st.info("📡 **Modo Real** - Tentando obter dados reais via yfinance")


def check_yfinance_availability() -> bool:
    """
    Verifica se yfinance está funcionando.
    
    Returns:
        bool: True se yfinance está disponível
    """
    try:
        import yfinance as yf
        
        # Tentar download simples
        test = yf.download('PETR4.SA', period='5d', progress=False, show_errors=False)
        
        return not test.empty
    
    except Exception as e:
        logger.error(f"yfinance não disponível: {e}")
        return False


def ensure_session_state_initialized():
    """
    Garante que todas as variáveis de sessão necessárias existam.
    """
    defaults = {
        'selected_tickers': [],
        'universe_df': None,
        'filtered_universe_df': None,
        'liquidity_applied': False,
        'price_data': None,
        'dividend_data': {},
        'expected_returns': None,
        'cov_matrix': None,
        'efficient_frontier': None,
        'optimized_portfolios': {},
        'specialized_portfolios': {},
        'recommended_portfolio': None,
        'share_quantities': {},
        'dividend_metrics': None,
        'use_mock_data': False,
        'yfinance_checked': False,
        'yfinance_works': False,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def format_currency(value: float, currency: str = "R$") -> str:
    """
    Formata valor como moeda.
    
    Args:
        value: Valor numérico
        currency: Símbolo da moeda
    
    Returns:
        String formatada
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    return f"{currency} {value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como percentual.
    
    Args:
        value: Valor decimal (0.15 = 15%)
        decimals: Casas decimais
    
    Returns:
        String formatada
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Formata número com separadores.
    
    Args:
        value: Valor numérico
        decimals: Casas decimais
    
    Returns:
        String formatada
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    return f"{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisão segura que retorna default se denominador for zero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se divisão não for possível
    
    Returns:
        Resultado da divisão ou default
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default


def validate_tickers_selected() -> bool:
    """
    Valida se há tickers selecionados.
    
    Returns:
        bool: True se há tickers selecionados
    """
    if not st.session_state.get('selected_tickers'):
        st.warning("⚠️ Nenhum ativo selecionado. Por favor, selecione ativos primeiro.")
        return False
    
    return True


def validate_data_loaded() -> bool:
    """
    Valida se os dados foram carregados.
    
    Returns:
        bool: True se dados estão carregados
    """
    if st.session_state.get('price_data') is None or st.session_state.price_data.empty:
        st.warning("⚠️ Dados não carregados. Por favor, carregue os dados primeiro.")
        return False
    
    return True


def get_period_info() -> dict:
    """
    Retorna informações sobre o período de análise.
    
    Returns:
        dict: Informações do período
    """
    start = st.session_state.get('period_start')
    end = st.session_state.get('period_end')
    
    if start and end:
        days = (end - start).days
        years = days / 365.25
        
        return {
            'start': start,
            'end': end,
            'days': days,
            'years': years,
            'trading_days': int(days * 0.7)  # Aproximação
        }
    
    return {}


def create_download_link(data, filename: str, label: str = "Download"):
    """
    Cria link de download para dados.
    
    Args:
        data: Dados para download (DataFrame, string, etc)
        filename: Nome do arquivo
        label: Texto do botão
    """
    import pandas as pd
    
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=True)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    elif isinstance(data, str):
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime='text/plain'
        )
    else:
        st.error("Tipo de dado não suportado para download")


def log_user_action(action: str, details: dict = None):
    """
    Registra ação do usuário para debugging.
    
    Args:
        action: Descrição da ação
        details: Detalhes adicionais
    """
    log_msg = f"User action: {action}"
    
    if details:
        log_msg += f" | Details: {details}"
    
    logger.info(log_msg)


def show_error_with_details(error: Exception, context: str = ""):
    """
    Mostra erro com detalhes para o usuário.
    
    Args:
        error: Exceção capturada
        context: Contexto do erro
    """
    st.error(f"❌ Erro: {str(error)}")
    
    if context:
        st.caption(f"Contexto: {context}")
    
    with st.expander("Ver detalhes técnicos"):
        st.code(f"{type(error).__name__}: {str(error)}")
        
        import traceback
        st.code(traceback.format_exc())


def create_metric_card_html(title: str, value: str, delta: str = None, 
                           icon: str = "📊", help_text: str = None) -> str:
    """
    Cria HTML para card de métrica customizado.
    
    Args:
        title: Título da métrica
        value: Valor principal
        delta: Variação (opcional)
        icon: Emoji do ícone
        help_text: Texto de ajuda
    
    Returns:
        HTML string
    """
    delta_html = f'<p style="color: #00FF88; font-size: 0.9rem; margin: 0.5rem 0 0 0;">{delta}</p>' if delta else ''
    help_html = f'<p style="color: #B0B0B0; font-size: 0.8rem; margin-top: 0.5rem;">{help_text}</p>' if help_text else ''
    
    html = f"""
    <div style="
        background: rgba(38, 39, 48, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h4 style="margin: 0; color: #B0B0B0; font-size: 0.9rem;">{title}</h4>
        </div>
        <p style="font-size: 2rem; font-weight: bold; color: #00D9FF; margin: 0.5rem 0;">
            {value}
        </p>
        {delta_html}
        {help_html}
    </div>
    """
    
    return html


import pandas as pd
import numpy as np

# Importar para as funções acima funcionarem
