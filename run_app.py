"""
App de Alocação e Dividendos - B3
Análise quantitativa de portfólios com foco em dividendos regulares
"""

import streamlit as st
import logging
from datetime import datetime
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

# CSS customizado para tema futurista
st.markdown("""
    <style>
    /* Estilos globais */
    .main {
        padding: 2rem;
    }
    
    /* Cards com efeito glassmorphism */
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
    
    /* Títulos com gradiente */
    .gradient-title {
        background: linear-gradient(90deg, #00D9FF 0%, #7B2FFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Tooltips customizados */
    .tooltip-icon {
        color: #00D9FF;
        cursor: help;
        margin-left: 0.5rem;
    }
    
    /* Botões com efeito neon */
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
    
    /* Tabelas com hover effect */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(14, 17, 23, 0.95);
    }
    
    /* Métricas destacadas */
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
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    if 'period_start' not in st.session_state:
        st.session_state.period_start = None
    
    if 'period_end' not in st.session_state:
        st.session_state.period_end = None
    
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 0.1175  # Selic aproximada
    
    if 'max_weight_per_asset' not in st.session_state:
        st.session_state.max_weight_per_asset = 0.15
    
    if 'max_weight_per_sector' not in st.session_state:
        st.session_state.max_weight_per_sector = 0.40
    
    if 'lambda_penalty' not in st.session_state:
        st.session_state.lambda_penalty = 0.5
    
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 10000.0
    
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    
    if 'dividend_data' not in st.session_state:
        st.session_state.dividend_data = None
    
    if 'optimized_portfolios' not in st.session_state:
        st.session_state.optimized_portfolios = {}

def render_sidebar():
    """Renderiza sidebar com controles globais."""
    with st.sidebar:
        st.markdown('<p class="gradient-title">⚙️ Configurações</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Período de análise
        st.markdown("### 📅 Período de Análise")
        
        period_option = st.radio(
            "Selecione o período:",
            ["1 ano", "2 anos", "5 anos", "10 anos", "Personalizado"],
            help="Períodos mais longos tendem a estabilizar métricas; períodos curtos refletem regime recente."
        )
        
        from datetime import datetime, timedelta
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
                    "Data Início",
                    value=end_date - timedelta(days=365),
                    max_value=end_date
                )
            with col2:
                end_date = st.date_input(
                    "Data Fim",
                    value=end_date,
                    max_value=datetime.now()
                )
        
        st.session_state.period_start = start_date
        st.session_state.period_end = end_date
        
        # Validação do período
        days_diff = (end_date - start_date).days
        if days_diff < 252:
            st.warning("⚠️ Período menor que 1 ano pode gerar métricas instáveis")
        
        st.markdown("---")
        
        # Taxa livre de risco
        st.markdown("### 💰 Taxa Livre de Risco")
        st.session_state.risk_free_rate = st.number_input(
            "Taxa anual (%):",
            min_value=0.0,
            max_value=50.0,
            value=st.session_state.risk_free_rate * 100,
            step=0.25,
            help="Taxa Selic ou CDI como referência. Usada no cálculo do Índice Sharpe."
        ) / 100
        
        st.markdown("---")
        
        # Restrições de alocação
        st.markdown("### 🛡️ Restrições de Alocação")
        
        st.session_state.max_weight_per_asset = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=50,
            value=int(st.session_state.max_weight_per_asset * 100),
            step=5,
            help="Limite individual por ativo para evitar concentração excessiva."
        ) / 100
        
        st.session_state.max_weight_per_sector = st.slider(
            "Peso máximo por setor (%):",
            min_value=10,
            max_value=100,
            value=int(st.session_state.max_weight_per_sector * 100),
            step=5,
            help="Limite por setor para diversificação setorial."
        ) / 100
        
        st.markdown("---")
        
        # Parâmetros de otimização
        st.markdown("### 🧮 Otimização de Dividendos")
        
        st.session_state.lambda_penalty = st.slider(
            "Penalização da variância (λ):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.lambda_penalty,
            step=0.05,
            help="Quanto maior, mais prioriza regularidade dos dividendos vs. yield total."
        )
        
        st.markdown("---")
        
        # Valor a investir
        st.markdown("### 💵 Valor a Investir")
        
        st.session_state.investment_amount = st.number_input(
            "Valor (R$):",
            min_value=100.0,
            max_value=10000000.0,
            value=st.session_state.investment_amount,
            step=1000.0,
            help="Valor total para alocação no portfólio recomendado."
        )
        
        st.markdown("---")
        
        # Informações do sistema
        st.markdown("### ℹ️ Informações")
        st.caption(f"**Versão:** 1.0.0")
        st.caption(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        st.caption(f"**Ativos selecionados:** {len(st.session_state.selected_tickers)}")

def main():
    """Função principal do app."""
    initialize_session_state()
    render_sidebar()
    
    # Header principal
    st.markdown('<p class="gradient-title">📈 Portfolio B3 - Análise de Dividendos</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bem-vindo ao sistema de análise quantitativa de portfólios focado em **dividendos regulares** 
    e **otimização de risco-retorno** para ativos da B3.
    """)
    
    # Cards de métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Ativos Selecionados</h4>
            <p class="highlight-metric">{}</p>
        </div>
        """.format(len(st.session_state.selected_tickers)), unsafe_allow_html=True)
    
    with col2:
        days = (st.session_state.period_end - st.session_state.period_start).days
        st.markdown("""
        <div class="metric-card">
            <h4>📅 Período de Análise</h4>
            <p class="highlight-metric">{} dias</p>
        </div>
        """.format(days), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Taxa Livre de Risco</h4>
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
    
    # Instruções de uso
    with st.expander("📖 Como usar este aplicativo", expanded=False):
        st.markdown("""
        ### Fluxo de trabalho recomendado:
        
        1. **Selecionar Ativos** 🎯
           - Navegue até a página "Selecionar Ativos"
           - Filtre por setor/segmento ou selecione manualmente
           - Apenas ativos negociados nos últimos 30 dias são exibidos
        
        2. **Análise de Dividendos** 💸
           - Visualize histórico de dividendos
           - Analise regularidade dos pagamentos
           - Veja calendário mensal projetado
        
        3. **Portfólios Eficientes** 📊
           - Explore a fronteira eficiente de Markowitz
           - Compare diferentes estratégias de alocação
        
        4. **Sharpe e MinVol** 🎯
           - Carteiras otimizadas específicas
           - Máximo Sharpe vs. Mínima Volatilidade
        
        5. **Resumo Executivo** 📋
           - Recomendação final personalizada
           - Quantidades exatas de ações a comprar
           - Exportação de relatórios
        
        ### Dicas importantes:
        - Ajuste os parâmetros no painel lateral conforme seu perfil
        - Períodos mais longos (5-10 anos) geram análises mais robustas
        - A taxa livre de risco afeta diretamente o cálculo do Sharpe
        - Restrições de concentração protegem contra risco idiossincrático
        """)
    
    # Avisos importantes
    st.info("""
    ℹ️ **Aviso Legal:** Este aplicativo é uma ferramenta de análise quantitativa e não constitui 
    recomendação de investimento. Sempre consulte um profissional certificado antes de tomar 
    decisões financeiras. Rentabilidade passada não garante resultados futuros.
    """)
    
    # Navegação rápida
    st.markdown("### 🚀 Navegação Rápida")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🎯 Selecionar Ativos", use_container_width=True):
            st.switch_page("app/pages/01_Selecionar_Ativos.py")
    
    with col2:
        if st.button("💸 Análise Dividendos", use_container_width=True):
            st.switch_page("app/pages/02_Análise_de_Dividendos.py")
    
    with col3:
        if st.button("📊 Portfólios", use_container_width=True):
            st.switch_page("app/pages/03_Portfólios_Eficientes.py")
    
    with col4:
        if st.button("🎯 Sharpe/MinVol", use_container_width=True):
            st.switch_page("app/pages/04_Sharpe_e_MinVol.py")
    
    with col5:
        if st.button("📋 Resumo", use_container_width=True):
            st.switch_page("app/pages/05_Resumo_Executivo.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Desenvolvido com ❤️ usando Streamlit | Dados via yfinance</p>
        <p style="font-size: 0.8rem;">© 2025 Portfolio B3 Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
