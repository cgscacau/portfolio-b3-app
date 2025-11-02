"""
Análise de Dividendos
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core import data

st.set_page_config(
    page_title="Análise de Dividendos",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# INICIALIZAÇÃO DO SESSION STATE
# ==========================================
def init_session_state():
    """Inicializa todas as variáveis do session state"""
    
    # Período
    if 'period_start' not in st.session_state:
        st.session_state.period_start = datetime.now() - timedelta(days=365)
    
    if 'period_end' not in st.session_state:
        st.session_state.period_end = datetime.now()
    
    # Tickers
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []
    
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    
    # Dados
    if 'dividend_data' not in st.session_state:
        st.session_state.dividend_data = None
    
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None

# CHAMAR IMEDIATAMENTE
init_session_state()

# ==========================================
# RESTO DO CÓDIGO
# ==========================================

def load_dividend_data():
    """Carrega dados de dividendos"""
    
    # Agora estas variáveis existem com certeza
    start_date = st.session_state.period_start
    end_date = st.session_state.period_end
    tickers = st.session_state.portfolio_tickers
    
    if not tickers:
        st.warning("⚠ Nenhum ativo selecionado. Vá para 'Selecionar Ativos' primeiro.")
        return
    
    st.info(f"Carregando dados de {len(tickers)} ativos...")
    
    # Carregar preços
    with st.spinner("Carregando preços..."):
        try:
            prices_df = data.get_price_history(tickers, start_date, end_date, use_cache=True)
            
            if prices_df.empty:
                st.error("❌ Erro ao carregar preços")
                return
            
            st.session_state.price_data = prices_df
            st.success(f"✓ Preços carregados: {len(prices_df)} dias")
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar preços: {str(e)}")
            return
    
    # Carregar dividendos
    with st.spinner("Carregando dividendos..."):
        try:
            all_dividends = {}
            
            for ticker in tickers:
                divs = data.get_dividends(ticker, start_date, end_date)
                if not divs.empty:
                    all_dividends[ticker] = divs
            
            if all_dividends:
                st.session_state.dividend_data = all_dividends
                total_divs = sum(len(df) for df in all_dividends.values())
                st.success(f"✓ Dividendos carregados: {total_divs} pagamentos")
            else:
                st.warning("⚠ Nenhum dividendo encontrado no período")
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar dividendos: {str(e)}")


def show_dividend_summary():
    """Mostra resumo de dividendos"""
    
    if st.session_state.dividend_data is None:
        st.info("Carregue os dados primeiro")
        return
    
    st.header("📊 Resumo de Dividendos")
    
    # Processar dados
    total_pagamentos = 0
    total_valor = 0
    
    for ticker, df in st.session_state.dividend_data.items():
        total_pagamentos += len(df)
        total_valor += df['valor'].sum()
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ativos com Dividendos", len(st.session_state.dividend_data))
    
    with col2:
        st.metric("Total de Pagamentos", total_pagamentos)
    
    with col3:
        st.metric("Valor Total", f"R$ {total_valor:,.2f}")
    
    # Tabela detalhada
    st.subheader("Detalhes por Ativo")
    
    detalhes = []
    for ticker, df in st.session_state.dividend_data.items():
        detalhes.append({
            'Ativo': ticker,
            'Pagamentos': len(df),
            'Total': df['valor'].sum(),
            'Média': df['valor'].mean(),
            'Último': df['valor'].iloc[-1] if not df.empty else 0
        })
    
    df_detalhes = pd.DataFrame(detalhes)
    
    st.dataframe(
        df_detalhes.style.format({
            'Total': 'R$ {:,.2f}',
            'Média': 'R$ {:,.2f}',
            'Último': 'R$ {:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )


def main():
    """Função principal"""
    
    st.title("💰 Análise de Dividendos")
    st.markdown("Análise completa de dividendos: histórico, regularidade e projeções.")
    st.markdown("---")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção de período
        st.subheader("Período")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start = st.date_input(
                "Data Inicial",
                value=st.session_state.period_start,
                key="start_date_input"
            )
        
        with col2:
            end = st.date_input(
                "Data Final",
                value=st.session_state.period_end,
                key="end_date_input"
            )
        
        # Atualizar session state
        st.session_state.period_start = datetime.combine(start, datetime.min.time())
        st.session_state.period_end = datetime.combine(end, datetime.min.time())
        
        st.markdown("---")
        
        # Botão carregar
        if st.button("📥 Carregar Dados", type="primary", use_container_width=True):
            load_dividend_data()
    
    # Conteúdo principal
    if not st.session_state.portfolio_tickers:
        st.warning("⚠ Nenhum ativo no portfólio. Vá para 'Selecionar Ativos' primeiro.")
        st.stop()
    
    # Mostrar ativos selecionados
    st.info(f"📊 {len(st.session_state.portfolio_tickers)} ativos no portfólio")
    
    with st.expander("Ver ativos"):
        st.write(st.session_state.portfolio_tickers)
    
    st.markdown("---")
    
    # Mostrar resumo se houver dados
    if st.session_state.dividend_data:
        show_dividend_summary()
    else:
        st.info("👈 Use o botão 'Carregar Dados' na barra lateral")


if __name__ == "__main__":
    main()
