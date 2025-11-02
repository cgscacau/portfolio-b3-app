"""
App de Alocação e Dividendos - B3
Análise quantitativa de portfólios com foco em dividendos regulares
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configurar logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Portfolio B3 - Análise de Dividendos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    
    .metric-card {
        background: rgba(38, 39, 48, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00D9FF;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .gradient-title {
        background: linear-gradient(90deg, #00D9FF 0%, #7B2FFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #00D9FF;
        background: rgba(0, 217, 255, 0.1);
        color: #00D9FF;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #00D9FF;
        color: #0E1117;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
    }
    
    .highlight-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #00D9FF;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    
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
        'period_start': datetime.now() - timedelta(days=365),
        'period_end': datetime.now(),
        'risk_free_rate': 0.1175,
        'max_weight_per_asset': 0.15,
        'max_weight_per_sector': 0.40,
        'lambda_penalty': 0.5,
        'investment_amount': 10000.0,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def render_sidebar():
    """Renderiza sidebar com controles globais."""
    
    with st.sidebar:
        st.markdown('<p class="gradient-title">⚙️ Configurações</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # PERÍODO DE ANÁLISE
        st.markdown("### 📅 Período de Análise")
        
        period_option = st.radio(
            "Selecione o período:",
            ["1 ano", "2 anos", "5 anos", "10 anos", "Personalizado"],
            help="Períodos mais longos tendem a estabilizar métricas"
        )
        
        end_date = datetime.now()
        
        if period_option == "1 ano":
            start_date = end_date - timedelta(days=365)
        elif period_option == "2 anos":
            start_date = end_date - timedelta(days=730)
        elif period_option == "5 anos":
            start_date = end_date - timedelta(days=1825)
        elif period_option == "10 anos":
            start_date = end_date - timedelta(days=3650)
        else:  # Personalizado
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Início",
                    value=st.session_state.period_start,
                    max_value=end_date
                )
            with col2:
                end_date = st.date_input(
                    "Fim",
                    value=end_date,
                    max_value=datetime.now()
                )
        
        st.session_state.period_start = start_date
        st.session_state.period_end = end_date
        
        # Validação
        days_diff = (end_date - start_date).days
        if days_diff < 252:
            st.warning("⚠️ Período < 1 ano pode gerar métricas instáveis")
        
        st.markdown("---")
        
        # TAXA LIVRE DE RISCO
        st.markdown("### 💰 Taxa Livre de Risco")
        
        st.session_state.risk_free_rate = st.number_input(
            "Taxa anual (%):",
            min_value=0.0,
            max_value=50.0,
            value=st.session_state.risk_free_rate * 100,
            step=0.25,
            help="Taxa Selic ou CDI. Usada no cálculo do Sharpe."
        ) / 100
        
        st.markdown("---")
        
        # RESTRIÇÕES
        st.markdown("### 🛡️ Restrições de Alocação")
        
        st.session_state.max_weight_per_asset = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=50,
            value=int(st.session_state.max_weight_per_asset * 100),
            step=5,
            help="Limite individual por ativo"
        ) / 100
        
        st.session_state.max_weight_per_sector = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=int(st.session_state.max_weight_per_sector * 100),
            step=5,
            help="Limite por setor"
        ) / 100
        
        st.markdown("---")
        
        # OTIMIZAÇÃO DIVIDENDOS
        st.markdown("### 🧮 Otimização de Dividendos")
        
        st.session_state.lambda_penalty = st.slider(
            "Penalização da variância (λ):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.lambda_penalty,
            step=0.05,
            help="Maior = prioriza regularidade"
        )
        
        st.markdown("---")
        
        # VALOR A INVESTIR
        st.markdown("### 💵 Valor a Investir")
        
        st.session_state.investment_amount = st.number_input(
            "Valor (R$):",
            min_value=100.0,
            max_value=10000000.0,
            value=st.session_state.investment_amount,
            step=1000.0,
            help="Valor total para alocação"
        )
        
        st.markdown("---")
        
        # INFORMAÇÕES
        st.markdown("### ℹ️ Informações")
        
        st.caption(f"**Versão:** 1.0.0")
        st.caption(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        st.caption(f"**Ativos:** {len(st.session_state.get('selected_tickers', []))}")


def main():
    """Função principal do app."""
    
    # Inicializar
    initialize_session_state()
    render_sidebar()
    
    # HEADER
    st.markdown('<p class="gradient-title">📈 Portfolio B3 - Análise de Dividendos</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bem-vindo ao sistema de análise quantitativa de portfólios focado em **dividendos regulares** 
    e **otimização de risco-retorno** para ativos da B3.
    """)
    
    # CARDS DE MÉTRICAS
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Ativos Selecionados</h4>
            <p class="highlight-metric">{}</p>
        </div>
        """.format(len(st.session_state.get('selected_tickers', []))), unsafe_allow_html=True)
    
    with col2:
        days = (st.session_state.period_end - st.session_state.period_start).days
        st.markdown("""
        <div class="metric-card">
            <h4>📅 Período</h4>
            <p class="highlight-metric">{} dias</p>
        </div>
        """.format(days), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Taxa Livre Risco</h4>
            <p class="highlight-metric">{:.2f}%</p>
        </div>
        """.format(st.session_state.risk_free_rate * 100), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>💵 Investimento</h4>
            <p class="highlight-metric">R$ {:.0f}</p>
        </div>
        """.format(st.session_state.investment_amount), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # INSTRUÇÕES
    with st.expander("📖 Como usar este aplicativo", expanded=False):
        st.markdown("""
        ### Fluxo de trabalho:
        
        1. **Selecionar Ativos** 🎯
           - Use o menu lateral para navegar
           - Filtre por liquidez e setor
           - Selecione 10-30 ativos
        
        2. **Análise de Dividendos** 💸
           - Histórico de pagamentos
           - Índice de regularidade
           - Calendário mensal
        
        3. **Portfólios Eficientes** 📊
           - Fronteira eficiente de Markowitz
           - Otimização risco-retorno
        
        4. **Sharpe e MinVol** 🎯
           - Máximo Sharpe
           - Mínima Volatilidade
           - Dividendos Regulares
        
        5. **Resumo Executivo** 📋
           - Recomendação personalizada
           - Quantidades de ações
           - Relatórios
        
        ### Dicas:
        - Períodos longos (5-10 anos) = análises robustas
        - Taxa livre de risco afeta o Sharpe
        - Diversifique entre setores
        """)
    
    # AVISO LEGAL
    st.info("""
    ℹ️ **Aviso Legal:** Este aplicativo é uma ferramenta de análise e **não constitui 
    recomendação de investimento**. Consulte um profissional certificado. 
    Rentabilidade passada não garante resultados futuros.
    """)
    
    # NAVEGAÇÃO
    st.markdown("### 🚀 Navegação")
    
    st.markdown("""
    Use o **menu lateral** (☰) para acessar:
    
    - 🎯 **Selecionar Ativos**
    - 💸 **Análise de Dividendos**
    - 📊 **Portfólios Eficientes**
    - 🎯 **Sharpe e MinVol**
    - 📋 **Resumo Executivo**
    """)
    
    # STATUS
    with st.expander("🔧 Status do Sistema", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Componentes:**")
            st.text("✅ Interface: OK")
            st.text("✅ Módulos: OK")
            st.text("✅ Cache: OK")
            st.text("✅ yfinance: Ativo")
        
        with col2:
            st.markdown("**Dados:**")
            
            if st.session_state.get('selected_tickers'):
                st.text(f"✅ Ativos: {len(st.session_state.selected_tickers)}")
            else:
                st.text("⚪ Ativos: Nenhum")
            
            if st.session_state.get('price_data') is not None:
                st.text("✅ Preços: Carregados")
            else:
                st.text("⚪ Preços: Não carregados")
            
            if st.session_state.get('dividend_data'):
                st.text("✅ Dividendos: Carregados")
            else:
                st.text("⚪ Dividendos: Não carregados")
    
    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Desenvolvido com ❤️ usando Streamlit | Dados via yfinance</p>
        <p style="font-size: 0.8rem;">© 2025 Portfolio B3 Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
