"""
🎯 Portfolio B3 - Análise de Investimentos
Aplicação para seleção, análise e otimização de portfólios de ativos da B3
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Configurar path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from core.init import init_all

# Configuração da página inicial
st.set_page_config(
    page_title="Portfolio B3 - Home",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
init_all()


def main():
    """Página principal (Home)"""
    
    # Header
    st.title("🎯 Portfolio B3 - Análise de Investimentos")
    st.markdown("### Plataforma completa para análise e otimização de portfólios da B3")
    
    st.markdown("---")
    
    # Introdução
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 👋 Bem-vindo!
        
        Esta aplicação oferece ferramentas profissionais para análise de investimentos na B3:
        
        - **Seleção inteligente** de ativos por setor, segmento e liquidez
        - **Análise de dividendos** com histórico, regularidade e calendário mensal
        - **Otimização de portfólios** usando Teoria Moderna de Markowitz
        - **Comparação de estratégias** (Sharpe Máximo vs Mínima Volatilidade)
        - **Resumo executivo** com recomendações personalizadas
        """)
    
    with col2:
        st.info("""
        **📊 Status do Sistema**
        
        ✅ Dados: Yahoo Finance  
        ✅ Cache: Ativo  
        ✅ Páginas: 5  
        
        **🎨 Tema**  
        Dark Mode Profissional
        """)
    
    st.markdown("---")
    
    # Fluxo de trabalho
    st.header("🔄 Fluxo de Trabalho")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Selecionar
        
        📊 **Selecionar Ativos**
        
        - Universo B3 completo
        - Filtros por setor
        - Verificação de liquidez
        - Seleção inteligente
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Analisar
        
        💰 **Análise de Dividendos**
        
        - Dividend Yield
        - Regularidade
        - Calendário mensal
        - Projeções de renda
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Otimizar
        
        📈 **Portfólios Eficientes**
        
        - Fronteira eficiente
        - Markowitz
        - Sharpe Máximo
        - Mínima Volatilidade
        """)
    
    with col4:
        st.markdown("""
        ### 4️⃣ Comparar
        
        ⚖️ **Sharpe vs MinVol**
        
        - Comparação lado a lado
        - Performance histórica
        - Drawdown
        - Métricas ajustadas
        """)
    
    with col5:
        st.markdown("""
        ### 5️⃣ Decidir
        
        📋 **Resumo Executivo**
        
        - Recomendação final
        - Alocação sugerida
        - Quantidades
        - Export PDF
        """)
    
    st.markdown("---")
    
    # Estatísticas da sessão
    st.header("📊 Status da Sessão Atual")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_selecionados = len(st.session_state.get('selected_tickers', []))
        st.metric(
            "Ativos Selecionados",
            num_selecionados,
            help="Ativos marcados para análise"
        )
    
    with col2:
        num_portfolio = len(st.session_state.get('portfolio_tickers', []))
        st.metric(
            "Ativos no Portfólio",
            num_portfolio,
            help="Ativos salvos no portfólio"
        )
    
    with col3:
        periodo_dias = (st.session_state.period_end - st.session_state.period_start).days
        st.metric(
            "Período de Análise",
            f"{periodo_dias} dias",
            help="Janela temporal configurada"
        )
    
    with col4:
        tem_analise = st.session_state.get('analise_dividendos_completa', False)
        status_analise = "✅ Completa" if tem_analise else "⏳ Pendente"
        st.metric(
            "Análise de Dividendos",
            status_analise,
            help="Status da última análise"
        )
    
    st.markdown("---")
    
    # Início rápido
    st.header("🚀 Início Rápido")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Primeiros Passos
        
        1. **Vá para "Selecionar Ativos"** no menu lateral
        2. **Configure os filtros** (setor, tipo, liquidez)
        3. **Selecione seus ativos** usando botões rápidos ou manualmente
        4. **Salve no portfólio** para usar nas análises
        5. **Navegue pelas outras páginas** para análises detalhadas
        
        ### 💡 Dicas
        
        - Comece com 10-20 ativos para melhor performance
        - Use "Top Liquidez" para ativos mais negociados
        - Use "Top DY" para foco em dividendos
        - Diversifique entre setores diferentes
        - Verifique liquidez antes de investir
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Recursos
        
        **Documentação**
        - [README.md](https://github.com/seu-repo) - Guia completo
        - Tooltips em cada página
        - Expandir "ℹ️ Como usar" em cada página
        
        **Dados**
        - Fonte: Yahoo Finance
        - Atualização: Tempo real
        - Cache: 1 hora
        
        **Suporte**
        - Issues no GitHub
        - Documentação inline
        """)
    
    st.markdown("---")
    
    # Avisos importantes
    st.header("⚠️ Avisos Importantes")
    
    st.warning("""
    **Disclaimer Legal:**
    
    - Esta aplicação é apenas para fins educacionais e informativos
    - Não constitui recomendação de investimento
    - Consulte um profissional certificado antes de investir
    - Rentabilidade passada não garante resultados futuros
    - Investimentos em renda variável envolvem riscos
    """)
    
    st.markdown("---")
    
    # Informações técnicas
    with st.expander("🔧 Informações Técnicas"):
        st.markdown("""
        ### Tecnologias Utilizadas
        
        - **Frontend**: Streamlit 1.31.0
        - **Dados**: yfinance, requests
        - **Análise**: pandas, numpy
        - **Otimização**: scipy
        - **Visualização**: plotly
        
        ### Estrutura do Projeto
        
        ```
        portfolio-b3-app/
        ├── run_app.py              # Página inicial (esta)
        ├── pages/
        │   ├── 01_Selecionar_Ativos.py
        │   ├── 02_Análise_de_Dividendos.py
        │   ├── 03_Portfólios_Eficientes.py
        │   ├── 04_Sharpe_e_MinVol.py
        │   └── 05_Resumo_Executivo.py
        ├── core/
        │   ├── init.py             # Inicialização global
        │   ├── data.py             # Download de dados
        │   ├── cache.py            # Sistema de cache
        │   └── metrics.py          # Cálculos de métricas
        └── assets/
            └── b3_universe.csv     # Universo de ativos (opcional)
        ```
        
        ### Cache e Performance
        
        - Dados de preços: Cache de 1 hora
        - Verificação de liquidez: Cache por sessão
        - Análises: Persistem até mudança de parâmetros
        
        ### Limitações Conhecidas
        
        - Yahoo Finance pode ter rate limiting
        - Dados podem ter atrasos de ~15 minutos
        - Alguns ativos podem não ter histórico completo
        - Performance reduz com >50 ativos
        """)
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        **Portfolio B3** | Versão 1.0.0  
        Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """)
    
    with col2:
        if st.button("🔄 Resetar Sessão", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col3:
        if st.button("📊 Ir para Seleção", type="primary", use_container_width=True):
            st.switch_page("pages/01_Selecionar_Ativos.py")


if __name__ == "__main__":
    main()
