"""
App de Alocação e Dividendos - B3
Análise quantitativa de portfólios com foco em dividendos regulares
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from core import utils

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
    
    /* Alertas customizados */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa variáveis de sessão."""
    
    # Usar função do utils para garantir todas as variáveis
    utils.ensure_session_state_initialized()
    
    # Valores padrão específicos
    if 'period_start' not in st.session_state:
        st.session_state.period_start = datetime.now() - timedelta(days=365)
    
    if 'period_end' not in st.session_state:
        st.session_state.period_end = datetime.now()
    
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


def check_yfinance_on_startup():
    """Verifica disponibilidade do yfinance na inicialização."""
    
    if not st.session_state.get('yfinance_checked', False):
        
        with st.spinner("🔍 Verificando disponibilidade do yfinance..."):
            yf_works = utils.check_yfinance_availability()
            
            st.session_state.yfinance_works = yf_works
            st.session_state.yfinance_checked = True
            
            if not yf_works:
                st.session_state.use_mock_data = True
                
                st.warning("""
                    ⚠️ **yfinance não está disponível no momento.**
                    
                    O aplicativo está configurado para usar **dados simulados** automaticamente.
                    
                    **Dados simulados são adequados para:**
                    - Testar a interface e funcionalidades
                    - Entender o fluxo de análise
                    - Demonstrações
                    
                    **Para análise real:**
                    - Tente novamente mais tarde
                    - Verifique sua conexão com a internet
                    - O yfinance pode estar temporariamente indisponível
                """)
                
                logger.warning("yfinance não disponível - usando modo simulado")
            else:
                st.success("✅ yfinance disponível - dados reais habilitados")
                logger.info("yfinance disponível")


def render_sidebar():
    """Renderiza sidebar com controles globais."""
    
    with st.sidebar:
        st.markdown('<p class="gradient-title">⚙️ Configurações</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========================================
        # MODO DE OPERAÇÃO
        # ========================================
        st.markdown("### 🔧 Modo de Operação")
        
        # Verificar se yfinance está disponível
        yf_available = st.session_state.get('yfinance_works', False)
        
        if not yf_available:
            st.error("📡 yfinance indisponível")
            st.caption("Usando dados simulados obrigatoriamente")
            use_mock_data = True
            st.session_state.use_mock_data = True
        else:
            use_mock_data = st.toggle(
                "Usar Dados Simulados",
                value=st.session_state.get('use_mock_data', False),
                key="use_mock_toggle",
                help="Ative para usar dados simulados mesmo com yfinance disponível"
            )
            
            st.session_state.use_mock_data = use_mock_data
        
        if use_mock_data:
            st.warning("⚠️ Modo simulado ativo")
            st.caption("Dados gerados aleatoriamente")
        else:
            st.info("📡 Modo real ativo")
            st.caption("Usando yfinance")
        
        st.markdown("---")
        
        # ========================================
        # PERÍODO DE ANÁLISE
        # ========================================
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
        
        # Validação do período
        days_diff = (end_date - start_date).days
        if days_diff < 252:
            st.warning("⚠️ Período < 1 ano pode gerar métricas instáveis")
        
        st.markdown("---")
        
        # ========================================
        # TAXA LIVRE DE RISCO
        # ========================================
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
        
        # ========================================
        # RESTRIÇÕES DE ALOCAÇÃO
        # ========================================
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
            help="Limite por setor para diversificação"
        ) / 100
        
        st.markdown("---")
        
        # ========================================
        # OTIMIZAÇÃO DE DIVIDENDOS
        # ========================================
        st.markdown("### 🧮 Otimização de Dividendos")
        
        st.session_state.lambda_penalty = st.slider(
            "Penalização da variância (λ):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.lambda_penalty,
            step=0.05,
            help="Maior = prioriza regularidade vs. yield total"
        )
        
        st.markdown("---")
        
        # ========================================
        # VALOR A INVESTIR
        # ========================================
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
        
        # ========================================
        # INFORMAÇÕES DO SISTEMA
        # ========================================
        st.markdown("### ℹ️ Informações")
        
        st.caption(f"**Versão:** 1.0.0")
        st.caption(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        st.caption(f"**Ativos:** {len(st.session_state.get('selected_tickers', []))}")
        
        # Status do yfinance
        if st.session_state.get('yfinance_checked'):
            if st.session_state.get('yfinance_works'):
                st.caption("**yfinance:** ✅ Disponível")
            else:
                st.caption("**yfinance:** ❌ Indisponível")


def main():
    """Função principal do app."""
    
    # Inicializar
    initialize_session_state()
    
    # Verificar yfinance na primeira execução
    check_yfinance_on_startup()
    
    # Renderizar sidebar
    render_sidebar()
    
    # ========================================
    # HEADER PRINCIPAL
    # ========================================
    st.markdown('<p class="gradient-title">📈 Portfolio B3 - Análise de Dividendos</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bem-vindo ao sistema de análise quantitativa de portfólios focado em **dividendos regulares** 
    e **otimização de risco-retorno** para ativos da B3.
    """)
    
    # ========================================
    # CARDS DE MÉTRICAS RÁPIDAS
    # ========================================
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
    
    # ========================================
    # INSTRUÇÕES DE USO
    # ========================================
    with st.expander("📖 Como usar este aplicativo", expanded=False):
        st.markdown("""
        ### Fluxo de trabalho recomendado:
        
        1. **Selecionar Ativos** 🎯
           - Use o menu lateral para navegar até "Selecionar Ativos"
           - Filtre por setor/segmento ou selecione manualmente
           - Apenas ativos líquidos são recomendados
        
        2. **Análise de Dividendos** 💸
           - Visualize histórico de dividendos
           - Analise regularidade dos pagamentos
           - Veja calendário mensal projetado
        
        3. **Portfólios Eficientes** 📊
           - Explore a fronteira eficiente de Markowitz
           - Compare diferentes estratégias de alocação
        
        4. **Sharpe e MinVol** 🎯
           - Carteiras otimizadas específicas
           - Máximo Sharpe vs. Mínima Volatilidade vs. Dividendos Regulares
        
        5. **Resumo Executivo** 📋
           - Recomendação final personalizada
           - Quantidades exatas de ações a comprar
           - Exportação de relatórios
        
        ### Dicas importantes:
        
        - **Ajuste os parâmetros** no painel lateral conforme seu perfil
        - **Períodos mais longos** (5-10 anos) geram análises mais robustas
        - **Taxa livre de risco** afeta diretamente o cálculo do Sharpe
        - **Restrições de concentração** protegem contra risco idiossincrático
        
        ### Modo de dados:
        
        - **Dados Reais:** Obtidos via yfinance (pode haver falhas)
        - **Dados Simulados:** Gerados aleatoriamente para testes e demonstração
        """)
    
    # ========================================
    # AVISOS IMPORTANTES
    # ========================================
    st.info("""
    ℹ️ **Aviso Legal:** Este aplicativo é uma ferramenta de análise quantitativa e **não constitui 
    recomendação de investimento**. Sempre consulte um profissional certificado antes de tomar 
    decisões financeiras. Rentabilidade passada não garante resultados futuros.
    """)
    
    # Aviso adicional se estiver em modo simulado
    if st.session_state.get('use_mock_data', False):
        st.warning("""
        ⚠️ **Modo Simulado Ativo:** Os dados exibidos são gerados aleatoriamente e **não representam 
        a realidade do mercado**. Use apenas para testar funcionalidades e entender o fluxo de análise.
        """)
    
    # ========================================
    # NAVEGAÇÃO RÁPIDA
    # ========================================
    st.markdown("### 🚀 Navegação Rápida")
    
    st.markdown("""
    Use o **menu lateral** (☰) para navegar entre as páginas:
    
    - 🎯 **Selecionar Ativos** - Escolha os ativos para análise
    - 💸 **Análise de Dividendos** - Histórico e regularidade
    - 📊 **Portfólios Eficientes** - Fronteira de Markowitz
    - 🎯 **Sharpe e MinVol** - Otimizações específicas
    - 📋 **Resumo Executivo** - Recomendação final
    """)
    
    # ========================================
    # STATUS DO SISTEMA
    # ========================================
    with st.expander("🔧 Status do Sistema", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Componentes:**")
            st.text("✅ Interface: OK")
            st.text("✅ Módulos Core: OK")
            st.text("✅ Cache: OK")
            
            if st.session_state.get('yfinance_works'):
                st.text("✅ yfinance: Disponível")
            else:
                st.text("❌ yfinance: Indisponível")
        
        with col2:
            st.markdown("**Dados Carregados:**")
            
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
            
            if st.session_state.get('optimized_portfolios') or st.session_state.get('specialized_portfolios'):
                total_portfolios = len(st.session_state.get('optimized_portfolios', {})) + \
                                 len(st.session_state.get('specialized_portfolios', {}))
                st.text(f"✅ Portfólios: {total_portfolios}")
            else:
                st.text("⚪ Portfólios: Nenhum")
    
    # ========================================
    # FOOTER
    # ========================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Desenvolvido com ❤️ usando Streamlit | Dados via yfinance (ou simulados)</p>
        <p style="font-size: 0.8rem;">© 2025 Portfolio B3 Analytics</p>
        <p style="font-size: 0.7rem;">Este é um projeto educacional e de demonstração</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
